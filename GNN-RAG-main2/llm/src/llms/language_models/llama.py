from transformers import pipeline, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer
import project.GraphProjector
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
        self.maximun_token = 4096 - 100
        self.graph_projector = GraphProjector(self.gnn_dim, self.llm_dim)
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs, use_fast=False, token="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")
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,  
        use_fast=False, token="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")

        # --- 新增：尝试加载训练好的投影层权重 ---
        projector_path = os.path.join(self.args.model_path, "graph_projector.bin")
        if os.path.exists(projector_path):
            print(f"Loading trained graph projector from {projector_path}...")
            state_dict = torch.load(projector_path, map_with_ptr=True, map_location='cpu')
            self.graph_projector.load_state_dict(state_dict)
        else:
            print("Warning: No trained graph projector found, using initialized weights.")

        # 确保投影层状态正确
        self.graph_projector.to(device=self.model.device, dtype=self.model.dtype)
        self.graph_projector.eval()  # 推理模式
        # ------------------------------------
        target_dtype = self.DTYPE.get(self.args.dtype, torch.bfoat16)
        self.graph_projector.to(device="cuda", dtype=target_dtype)
        #model_kwargs.update({'use_auth_token': True})
        print("model: ", self.args.model_path)
        self.generator = pipeline("text-generation", token="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", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))

    @torch.inference_mode()
    def generate_sentence(self, llm_input, graph_feat=None):
        """
        llm_input: 文本 Prompt 字符串
        graph_feat: 该问题对应的 (6, 50) 图特征 (可以是 list 或 tensor)
        """
        # 1. 处理文本：转为 Embedding [1, seq_len, 4096]
        inputs = self.tokenizer(llm_input, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        attention_mask = inputs.attention_mask

        # 2. 如果提供了图特征，进行注入
        if graph_feat is not None:
            # 确保是 Tensor 且精度设备一致
            if not isinstance(graph_feat, torch.Tensor):
                graph_feat = torch.tensor(graph_feat, device=self.model.device, dtype=self.model.dtype)

            # 投影图特征 [6, 50] -> [6, 4096]
            # 增加 Batch 维度变为 [1, 6, 4096]
            projected_feat = self.graph_projector(graph_feat).unsqueeze(0)

            # 拼接：将图特征放在文本 Embedding 的最前面
            # 结果维度: [1, 6 + seq_len, 4096]
            inputs_embeds = torch.cat([projected_feat, inputs_embeds], dim=1)

            # 更新 Attention Mask：前缀的 6 个位置也要设为 1
            prefix_mask = torch.ones((1, 6), device=self.model.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 3. 调用底层的 generate 接口
        # 注意：传入 inputs_embeds 时，不需要传 input_ids
        generate_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False,  # 保持确定性输出，便于调试
        )

        # 4. 解码并返回结果
        return self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)