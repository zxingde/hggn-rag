import sys
import os
from peft import LoraConfig, TaskType

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from utils import *
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from align_kg.data_loader import load_multiple_datasets, load_new_tokens
from peft import AutoPeftModelForCausalLM, LoraConfig

import datasets
datasets.disable_progress_bar()

import torch.nn as nn
from project.GraphProjector import GraphProjector


class GraphLLMForTraining(nn.Module):
    def __init__(self, base_model, projector):
        super().__init__()
        self.base_model = base_model  # 这里的 base_model 是加载了 LoRA 的 Llama
        self.projector = projector

    def forward(self, input_ids, attention_mask, labels=None, graph_features=None, **kwargs):
        # 1. 将文本 ID 转为 Embedding
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # 2. 如果有图特征，进行注入
        if graph_features is not None:
            # 投影：[Batch, 6, 50] -> [Batch, 6, 4096]
            projected_feat = self.projector(graph_features.to(self.base_model.dtype))

            # 拼接：放在文本前面 [Batch, 6 + Seq_Len, 4096]
            inputs_embeds = torch.cat([projected_feat, inputs_embeds], dim=1)

            # 扩展 Attention Mask：为前面的 6 个向量补 1
            batch_size = attention_mask.shape[0]
            prefix_mask = torch.ones((batch_size, 6), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # 扩展 Labels：如果是在训练，labels 也要补 -100（表示这 6 个位置不计算 loss）
            if labels is not None:
                prefix_labels = torch.full((batch_size, 6), -100, device=labels.device, dtype=labels.long())
                labels = torch.cat([prefix_labels, labels], dim=1)

        # 3. 调用原模型的 forward，注意这次传入的是 inputs_embeds
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    # 为了让 Trainer 能访问到 base_model 的属性（如 config）
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(
        metadata={"help": "Path to the training data."}
    )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"}
    )
    rel_dict_path: list[str] = field(
        default=None, metadata={"help": "Path to the relation dictionary."}
    )
    add_rel_token: Optional[bool] = field(
        default=False, metadata={"help": "Wether to add relation token or not"}
    )
    prompt_path: str = field(
        default="prompts/llama2.txt",
        metadata={"help": "Path to the prompt template"},
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(default="q_proj,v_proj",
                                     metadata={"help": "Comma separated list of target modules."})
    # --- 新增下面这个参数 ---
    graph_feat_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the graph features .pkl file."}
    )
    # -----------------------

@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    ddp_find_unused_parameters: bool = field(default=False)

    def train():
        parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
        script_args, training_args = parser.parse_args_into_dataclasses()

        # 1. 加载基础大模型
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_auth_token=True,
        )

        # 2. 【核心修复】立即初始化投影层（必须在大模型加载后、封装前定义）
        # 注意：这里修正了导入路径，从 project.GraphProjector 导入
        from project.GraphProjector import GraphProjector
        print(f"Initializing GraphProjector with dim: 50 -> {model.config.hidden_size}")
        projector = GraphProjector(gnn_dim=50, llm_dim=model.config.hidden_size)
        projector.to(model.device).to(model.dtype)

        # 显式开启投影层的梯度，确保它会被训练
        for param in projector.parameters():
            param.requires_grad = True

        # 3. 加载 Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    # Add new tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    new_tokens = [SEP, BOP, EOP]
    if script_args.add_rel_token:
        new_tokens = load_new_tokens(new_tokens, script_args.rel_dict_path)
    smart_tokenizer_and_embedding_resize(new_tokens, special_tokens_dict, tokenizer, model)

    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # 4. 配置 PEFT (LoRA)
    peft_config = None
    if script_args.use_peft:
        if isinstance(script_args.lora_target_modules, str):
            targets = script_args.lora_target_modules.split(",")
        else:
            targets = script_args.lora_target_modules

        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 5. 【核心修复】封装图注入外壳 (现在 projector 已经定义好了，不会报错了)
    # 把它移出 PEFT 块，确保无论用不用 LoRA 都能注入图特征
    model = GraphLLMForTraining(model, projector)

    # 6. 设置训练参数
    # 关键：手动关闭列过滤，否则数据里的 graph_features 会被删掉导致训练报错
    training_args.remove_unused_columns = False

    # 7. 加载数据集
    train_dataset = load_multiple_datasets(
        script_args.data_path_list,
        graph_feat_path=script_args.graph_feat_path,
        shuffle=True
    )
    # --- 新增：初始化投影层 ---
    # 维度要与你的 GNN 特征 (50) 和 Llama 隐藏层 (model.config.hidden_size) 对齐
    from project.GraphProjector import GraphProjector  # 确保能引用到你写的类
    projector = GraphProjector(gnn_dim=50, llm_dim=model.config.hidden_size)
    projector.to(model.device).to(model.dtype)

    # 显式开启投影层的梯度，确保它会被训练
    for param in projector.parameters():
        param.requires_grad = True
    # -----------------------

    # Load datasets
    train_dataset = load_multiple_datasets(
        script_args.data_path_list,
        graph_feat_path=script_args.graph_feat_path,  # 传入刚才定义的路径
        shuffle=True
    )

    # Prepare instruct tuning
    response_template = "[/INST]"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=training_args.model_max_length,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field = "text",
        data_collator=data_collator,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    trainer.train(resume_from_checkpoint=checkpoint)
    # 6. 【核心修复】清理冗余的保存逻辑，确保投影层被保存
    print(f"Saving combined model and projector to {training_args.output_dir}")
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # 保存投影层 (从包装类中取出)
    projector_save_path = os.path.join(training_args.output_dir, "graph_projector.bin")
    torch.save(model.projector.state_dict(), projector_save_path)

    # 保存 LLM 部分 (LoRA 或全量)
    if script_args.use_peft:
        # 注意：model 是 GraphLLMForTraining，model.base_model 才是 PEFT 模型
        model.base_model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
