from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer

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
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs, use_fast=False, token="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")
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,  
        use_fast=False, token="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")
        #model_kwargs.update({'use_auth_token': True})
        print("model: ", self.args.model_path)
        # self.generator = pipeline("text-generation", token="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", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
        # 2. 【核心修改】加载裸模型，而不是 pipeline
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            device_map="auto",
            torch_dtype=self.DTYPE.get(self.args.dtype, torch.float16),
            token="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",
            **model_kwargs
        )
        self.model.eval()  # 设为评估模式

    # @torch.inference_mode()
    # def generate_sentence(self, llm_input):
    #     outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
    #     return outputs[0]['generated_text'] # type: ignore
    @torch.inference_mode()
    def generate_sentence(self, llm_input, soft_prompts=None):  # 【新增参数】
        """
        Args:
            llm_input (str): 文本 Prompt
            soft_prompts (Tensor): 你的 GNN 投影 Token [1, N, 4096] (可选)
        """
        # 1. 把文本变成 ID
        inputs = self.tokenizer(llm_input, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 2. 把 ID 变成向量 (Embeddings)
        # 只有变成了向量，才能跟你的 soft_prompts 拼接
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # 3. 【核心修改】拼接 Soft Prompt
        if soft_prompts is not None:
            # 确保 soft_prompts 在同一个设备上
            soft_prompts = soft_prompts.to(self.model.device)
            if soft_prompts.dtype != inputs_embeds.dtype:
                soft_prompts = soft_prompts.to(inputs_embeds.dtype)

            # 拼接到前面: [Soft_Prompts, Text_Prompts]
            inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)

            # 注意：如果加了 soft prompts，attention_mask 也要变长
            # 简单做法：全设为 1 (如果不涉及 padding)
            # 或者：
            prefix_mask = torch.ones(
                (soft_prompts.shape[0], soft_prompts.shape[1]),
                dtype=inputs.attention_mask.dtype,
                device=self.model.device
            )
            attention_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)
        else:
            attention_mask = inputs.attention_mask

        # 4. 调用 generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,  # 传入拼接后的向量
            attention_mask=attention_mask,
            max_new_tokens=self.args.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 5. 解码结果 (注意 outputs 包含输入，可能需要截断)
        # 使用 inputs_embeds 生成时，huggingface 不会返回输入的 token id，通常只返回新生成的
        # 但稳妥起见，我们直接解码整个 output
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 如果你发现 decoded_text 包含了输入 Prompt，你需要手动切掉
        # 简单处理：
        if decoded_text.startswith(llm_input):
            decoded_text = decoded_text[len(llm_input):]

        return decoded_text.strip()