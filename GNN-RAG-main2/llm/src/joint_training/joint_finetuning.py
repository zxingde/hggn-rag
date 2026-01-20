import sys
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 修正导入路径
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from align_kg.data_loader import load_multiple_datasets, load_new_tokens
from project.GraphProjector import GraphProjector


# --- 1. 模型包装类 (保持不变) ---
class GraphLLMForTraining(nn.Module):
    def __init__(self, base_model, projector):
        super().__init__()
        self.base_model = base_model
        self.projector = projector

    def forward(self, input_ids, attention_mask, labels=None, graph_features=None, **kwargs):
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # 图特征注入逻辑
        if graph_features is not None:
            # 确保类型匹配 (bfloat16/float16)
            graph_features = graph_features.to(inputs_embeds.dtype)

            projected_feat = self.projector(graph_features)
            inputs_embeds = torch.cat([projected_feat, inputs_embeds], dim=1)

            batch_size = attention_mask.shape[0]
            prefix_mask = torch.ones((batch_size, 6), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full((batch_size, 6), -100, device=labels.device, dtype=labels.long())
                labels = torch.cat([prefix_labels, labels], dim=1)

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


# --- 2. 新增：自定义 Collator (解决报错的核心) ---
class GraphDataCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        graph_features_batch = []
        clean_features = []

        for feature in features:
            # A. 提取并移除图特征
            gf = feature.get("graph_features")
            if gf is None:
                # 兜底：如果没有特征，给全0 (6x50)
                gf = [[0.0] * 50] * 6
            graph_features_batch.append(gf)

            # B. 过滤掉导致报错的非 Tensor 列 (text, id, 等)
            # 只保留 base_collator 能处理的字段
            new_feature = {
                k: v for k, v in feature.items()
                if k in ['input_ids', 'attention_mask', 'labels']
            }
            clean_features.append(new_feature)

        # C. 调用原来的 collator 处理 Padding 和 Label Masking
        batch = self.base_collator(clean_features)

        # D. 将图特征转为 Tensor 并塞回 batch
        # [Batch, 6, 50]
        batch["graph_features"] = torch.tensor(graph_features_batch, dtype=torch.float32)

        return batch


# --- 参数定义 ---
@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    graph_feat_path: Optional[str] = field(default=None, metadata={"help": "Path to .pkl file."})
    rel_dict_path: list[str] = field(default=None)
    add_rel_token: bool = field(default=False)
    use_peft: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,v_proj")


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_align")
    model_max_length: int = field(default=2048)
    remove_unused_columns: bool = field(default=False)  # 必须为 False


# --- 训练主函数 ---
def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 强制关闭列过滤
    training_args.remove_unused_columns = False

    # 1. 加载模型 (去掉了 device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Llama2 需要指定 pad_token

    # 2. 初始化投影层
    projector = GraphProjector(gnn_dim=50, llm_dim=model.config.hidden_size)
    projector.to(model.device).to(model.dtype)
    for param in projector.parameters():
        param.requires_grad = True

    # 3. 处理 Token
    special_tokens_dict = dict()
    if tokenizer.pad_token is None: special_tokens_dict['pad_token'] = '<PAD>'
    new_tokens = ['<SEP>', '<PATH>', '</PATH>']
    if script_args.add_rel_token:
        new_tokens = load_new_tokens(new_tokens, script_args.rel_dict_path)
    smart_tokenizer_and_embedding_resize(new_tokens, special_tokens_dict, tokenizer, model)

    # 4. 配置 LoRA
    if script_args.use_peft:
        if isinstance(script_args.lora_target_modules, str):
            targets = script_args.lora_target_modules.split(",")
        else:
            targets = script_args.lora_target_modules
        peft_config = LoraConfig(
            r=script_args.lora_r, lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout, target_modules=targets,
            bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    # 5. 封装外壳
    model = GraphLLMForTraining(model, projector)

    # 6. 加载数据
    train_dataset = load_multiple_datasets(
        script_args.data_path_list,
        graph_feat_path=script_args.graph_feat_path,
        shuffle=True
    )

    # 7. 组装 Data Collator
    # 先创建基础的 Collator (负责 Masking)
    base_collator = DataCollatorForCompletionOnlyLM("[/INST]", tokenizer=tokenizer)
    # 再套上我们的 Graph 过滤器
    graph_collator = GraphDataCollator(base_collator)

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=training_args.model_max_length,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        data_collator=graph_collator,  # 使用自定义的 collator
    )

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