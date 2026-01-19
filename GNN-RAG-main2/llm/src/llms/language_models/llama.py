from transformers import pipeline, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer
from project.GraphProjector import GraphProjector
from transformers import AutoModelForCausalLM
import os

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')


    def __init__(self, args):
        self.args = args
        self.gnn_dim = 50
        self.llm_dim = 4096
        self.maximun_token = 4096 - 100
        self.graph_projector = GraphProjector(self.gnn_dim, self.llm_dim)
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs, use_fast=False, token="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")
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self, **model_kwargs):
        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            use_fast=False,
            token="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"
        )

        # 3. 加载模型主体（替代之前的 pipeline）
        print("Loading model for inference: ", self.args.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            token="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",
            device_map="auto",
            torch_dtype=self.DTYPE.get(self.args.dtype, torch.float16),
            **model_kwargs
        )

        # 4. 确保投影层与模型设备和精度一致
        self.graph_projector.to(device=self.model.device, dtype=self.model.dtype)
        self.graph_projector.eval()

    @torch.inference_mode()
    def generate_sentence(self, llm_input, graph_feat=None):
        """
        核心：实现 Embedding 拼接逻辑
        """
        # 1. 文本转 Embedding
        inputs = self.tokenizer(llm_input, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        attention_mask = inputs.attention_mask

        # 2. 如果有特征，执行注入
        if graph_feat is not None:
            # 确保特征格式正确
            if not isinstance(graph_feat, torch.Tensor):
                graph_feat = torch.tensor(graph_feat, device=self.model.device, dtype=self.model.dtype)

            # 投影并增加 Batch 维度: [6, 50] -> [1, 6, 4096]
            projected_feat = self.graph_projector(graph_feat).unsqueeze(0)

            # 在序列长度维度(dim=1)进行拼接
            inputs_embeds = torch.cat([projected_feat, inputs_embeds], dim=1)

            # 扩展 Attention Mask
            prefix_mask = torch.ones((1, 6), device=self.model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 3. 调用底层的 generate 接口
        generate_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False
        )

        # 4. 解码输出
        return self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)