import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from src.models.projector import GraphProjector  # 引用刚才写的


class HGNN_RAG_Model(nn.Module):
    def __init__(self, llm_path, freeze_llm=True):
        super().__init__()
        print(f"�� [Model] Loading LLM from: {llm_path}")

        # 加载 LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 获取 LLM 维度 (例如 4096)
        embed_dim = self.llm.config.hidden_size
        print(f"�� [Model] Projector Target Dim: {embed_dim}")

        # 初始化 Projector
        self.projector = GraphProjector(input_dim=50, output_dim=embed_dim)
        # 确保 Projector 也是 float16 (和 LLM 对齐)
        self.projector = self.projector.to(dtype=self.llm.dtype).cuda()

        # 冻结 LLM
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("❄️ [Model] LLM parameters frozen.")

    def forward(self, input_ids, attention_mask, labels, graph_feats, graph_mask):
        # 1. 文本 Embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 2. 图特征 Embedding
        # 确保类型一致 (half)
        graph_feats = graph_feats.to(inputs_embeds.dtype).cuda()
        graph_mask = graph_mask.cuda()

        # [Batch, 1, 4096]
        graph_token = self.projector(graph_feats, graph_mask)

        # 3. 拼接: [Graph_Token, Text_Tokens...]
        inputs_embeds = torch.cat([graph_token, inputs_embeds], dim=1)

        # 4. 修正 Attention Mask (前面加1)
        b_size = attention_mask.shape[0]
        ones = torch.ones((b_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([ones, attention_mask], dim=1)

        # 5. 修正 Labels (前面加-100忽略)
        ignore = torch.full((b_size, 1), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([ignore, labels], dim=1)

        # 6. LLM Forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs