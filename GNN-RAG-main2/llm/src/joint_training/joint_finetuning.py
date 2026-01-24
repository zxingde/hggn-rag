import sys
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import logging
import pickle  # 【核心修改】新增 pickle 导入
import transformers
import numpy as np
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
# 如果原本的 loader 不支持 pkl，我们会在 main 里直接加载
from align_kg.data_loader import load_multiple_datasets, load_new_tokens
from project.GraphProjector import GraphProjector


# ==============================================================================
#  1. Tokenizer 调整工具 (保持不变)
# ==============================================================================
def smart_tokenizer_and_embedding_resize(new_tokens, special_tokens_dict, tokenizer, model):
    if len(new_tokens) > 0:
        tokenizer.add_tokens(new_tokens, special_tokens=True)
    if len(special_tokens_dict) > 0:
        tokenizer.add_special_tokens(special_tokens_dict)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing token embeddings to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-len(new_tokens)].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-len(new_tokens)].mean(dim=0, keepdim=True)

        input_embeddings[-len(new_tokens):] = input_embeddings_avg
        output_embeddings[-len(new_tokens):] = output_embeddings_avg


# ==============================================================================
#  2. 模型包装类 (GraphLLMForTraining)
#  【核心修改】：将硬编码的 6 改为动态读取 Projector 的 num_tokens (3)
# ==============================================================================
class GraphLLMForTraining(nn.Module):
    def __init__(self, base_model, projector):
        super().__init__()
        self.base_model = base_model
        self.projector = projector

        # 【修改】获取 Projector 定义的 token 数量 (例如 3)
        self.num_tokens = getattr(projector, "num_tokens", 3)
        print(f"GraphLLMForTraining initialized with num_graph_tokens={self.num_tokens}")

    def forward(self, input_ids, attention_mask, labels=None, graph_features=None, **kwargs):
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        if graph_features is not None:
            # 确保类型匹配
            graph_features = graph_features.to(inputs_embeds.dtype)

            # 1. 通过 Projector (输入 [Batch, N, 50] -> 输出 [Batch, 3, 4096])
            projected_feat = self.projector(graph_features)

            # 2. 拼接在文本 Embedding 前面
            inputs_embeds = torch.cat([projected_feat, inputs_embeds], dim=1)

            # 3. 扩展 Attention Mask
            batch_size = attention_mask.shape[0]
            # 【修改】使用 self.num_tokens (3) 而不是硬编码的 6
            prefix_mask = torch.ones((batch_size, self.num_tokens), device=attention_mask.device,
                                     dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # 4. 扩展 Labels (用于计算 Loss，前缀部分忽略)
            if labels is not None:
                prefix_labels = torch.full((batch_size, self.num_tokens), -100, device=labels.device, dtype=torch.long)
                labels = torch.cat([prefix_labels, labels], dim=1)

                # 【新增 DEBUG 代码】 只打印第一个 batch 的第一个样本
                # 这里的 batch_idx 是为了防止刷屏，可以设个全局变量或者只打印一次
                if not hasattr(self, "_debug_printed"):
                    self._debug_printed = True
                    print("\n" + "=" * 50)
                    print("【DEBUG】正在检查第一个 Batch 的 Labels...")
                    print(f"Total Label Length: {labels.shape[1]}")

                    # 统计有多少个非 -100 的有效 Label
                    valid_count = (labels[0] != -100).sum().item()
                    print(f"Valid Labels Count (非-100数量): {valid_count}")

                    if valid_count > 0:
                        # 打印出具体的有效 Label 值
                        valid_labels = labels[0][labels[0] != -100]
                        print(f"Valid Labels Content: {valid_labels}")
                        # 尝试解码回文本看看是啥 (假设你有 tokenizer，没有也没关系，看数字就行)
                    else:
                        print("❌ 严重错误：有效 Label 数量为 0！模型在用空气计算 Loss！")

                    print("=" * 50 + "\n")

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


# ==============================================================================
#  3. 数据组装器 (GraphDataCollator)
#  【核心修改】：适配新的 .pkl 格式 (list of dicts) 并实现动态 Padding
# ==============================================================================
class GraphDataCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        graph_features_batch = []
        clean_features = []

        # --- 第一步：解析图特征并提取向量 ---
        for feature in features:
            # 【修改】读取新字段 'gnn_path_data' (这是我们生成的 .pkl 里的字段)
            # 结构: [{'path':..., 'feature': array([..]), 'hit':..}, ...]
            path_data = feature.get("gnn_path_data", [])

            # 提取 feature 向量
            vectors = []
            if isinstance(path_data, list):
                for item in path_data:
                    if isinstance(item, dict) and 'feature' in item:
                        vectors.append(item['feature'])

            # 兜底：如果完全没路径，给一个全 0 向量 (1, 50) 防止报错
            if not vectors:
                vectors = [np.zeros(50, dtype=np.float32)]

            # 确保是 list of arrays
            graph_features_batch.append(vectors)

            # 清理非 Tensor 字段，避免 base_collator 报错
            new_feature = {
                k: v for k, v in feature.items()
                if k in ['input_ids', 'attention_mask', 'labels']
            }
            clean_features.append(new_feature)

        # --- 第二步：处理 Padding (变长 -> 定长) ---
        # 找出当前 Batch 里路径最多的样本 (Max Sequence Length)
        max_paths_in_batch = max(len(v) for v in graph_features_batch)

        padded_graph_batch = []
        for vectors in graph_features_batch:
            current_len = len(vectors)
            pad_len = max_paths_in_batch - current_len

            # 转换为 list
            vec_list = [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]

            # 补 0 向量
            if pad_len > 0:
                zero_vec = [0.0] * 50
                vec_list.extend([zero_vec] * pad_len)

            padded_graph_batch.append(vec_list)

        # --- 第三步：调用基础 Collator 处理文本 ---
        batch = self.base_collator(clean_features)

        # --- 第四步：塞回图特征 ---
        # 最终形状: [Batch_Size, Max_Paths, 50]
        # Projector 会在内部处理这个变长输入，并输出固定的 [Batch, 3, 4096]
        batch["graph_features"] = torch.tensor(padded_graph_batch, dtype=torch.float32)

        return batch


# ==============================================================================
#  4. 参数定义
# ==============================================================================
@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data (.pkl)."})
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    graph_feat_path: Optional[str] = field(default=None)  # 废弃，因为已经在 pkl 里了
    rel_dict_path: list[str] = field(default=None)
    add_rel_token: bool = field(default=False)
    use_peft: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj")
    # 【修改】增加 GNN 相关参数
    gnn_input_dim: int = field(default=50)
    gnn_hidden_dim: int = field(default=4096)
    num_graph_tokens: int = field(default=3, metadata={"help": "Number of tokens to represent the graph."})


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_align")
    model_max_length: int = field(default=2048)
    remove_unused_columns: bool = field(default=False)


# ==============================================================================
#  5. 训练主函数
# ==============================================================================
def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    training_args.ddp_find_unused_parameters = False

    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 初始化投影层
    # 【核心修改】传入 num_tokens=3
    print(
        f"Initializing GraphProjector: {script_args.gnn_input_dim} -> {model.config.hidden_size} (Tokens: {script_args.num_graph_tokens})")

    projector = GraphProjector(
        gnn_input_dim=script_args.gnn_input_dim,
        llm_hidden_dim=model.config.hidden_size,
        num_tokens=script_args.num_graph_tokens  # 3
    )

    projector.to(model.device).to(model.dtype)
    for param in projector.parameters():
        param.requires_grad = True

    # 3. 处理 Token
    special_tokens_dict = dict()
    if tokenizer.pad_token is None: special_tokens_dict['pad_token'] = '<PAD>'
    new_tokens = ['<SEP>', '<PATH>', '</PATH>']
    smart_tokenizer_and_embedding_resize(new_tokens, special_tokens_dict, tokenizer, model)

    # 4. 配置 LoRA
    if script_args.use_peft:
        model.enable_input_require_grads()
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        targets = script_args.lora_target_modules.split(",") if isinstance(script_args.lora_target_modules,
                                                                           str) else script_args.lora_target_modules
        peft_config = LoraConfig(
            r=script_args.lora_r, lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout, target_modules=targets,
            bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # 5. 封装外壳
    model = GraphLLMForTraining(model, projector)

    # 6. 加载数据
    print(f"Loading datasets from {script_args.data_path_list} ...")

    # 强制转换: 无论传入的是 list 还是 str，都先拿到路径字符串
    raw_path = script_args.data_path_list
    if isinstance(raw_path, list):
        # 有时候 argparse 会把单个字符串也包成 list，取第一个
        data_path_str = raw_path[0]
    else:
        data_path_str = raw_path

    print(f"DEBUG: Checking file path: {data_path_str}")

    # 判定逻辑
    if str(data_path_str).endswith('.pkl'):
        print("✅ Detected PKL file, loading directly...")
        with open(data_path_str, 'rb') as f:
            raw_data = pickle.load(f)  # 先加载成 list

        # 【核心修复】把 list 转换成 HuggingFace Dataset 对象
        train_dataset = Dataset.from_list(raw_data)

        print(f"✅ Loaded {len(train_dataset)} samples from PKL.")
    else:
        print("⚠️ Warning: Not a PKL file, falling back to legacy loader...")
        # 回退到旧逻辑
        train_dataset = load_multiple_datasets(
            script_args.data_path_list,
            graph_feat_path=script_args.graph_feat_path,
            shuffle=True
        )

    # 7. 组装 Data Collator
    base_collator = DataCollatorForCompletionOnlyLM("[/INST]", tokenizer=tokenizer)
    graph_collator = GraphDataCollator(base_collator)

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=training_args.model_max_length,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        data_collator=graph_collator,
    )

    print("Starting training...")
    trainer.train()

    # 9. 保存结果
    if not os.path.exists(training_args.output_dir): os.makedirs(training_args.output_dir)
    print(f"Saving graph projector to {training_args.output_dir}/graph_projector.bin")
    torch.save(model.projector.state_dict(), os.path.join(training_args.output_dir, "graph_projector.bin"))

    if script_args.use_peft:
        model.base_model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()