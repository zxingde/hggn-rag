import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphProjector(nn.Module):
    def __init__(self, input_dim=50, output_dim=4096):
        super().__init__()
        # 1. 投影层：先把 50维 升到 4096维
        self.linear = nn.Linear(input_dim, output_dim)

        # 2. 注意力查询向量 (Learnable Query)
        # 模型会学习"什么样的路径是重要的"
        self.query = nn.Parameter(torch.randn(output_dim))

        self.act = nn.SiLU()  # Llama/Mistral 常用激活函数

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Max_Paths, 50]
            mask: [Batch, Max_Paths] (1有效, 0无效)
        Returns:
            out: [Batch, 1, 4096] (作为1个Token)
        """
        # [B, N, 50] -> [B, N, 4096]
        h = self.act(self.linear(x))

        # --- Attention Pooling ---
        # 计算每条路径的分数: [B, N, 4096] * [4096] -> [B, N]
        scores = torch.matmul(h, self.query)

        # 处理 Mask: padding 部分给负无穷，softmax后为0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 归一化权重: [B, N] -> [B, N, 1]
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)

        # 加权求和: [B, N, 1] * [B, N, 4096] -> Sum -> [B, 4096]
        out = torch.sum(h * attn_weights, dim=1)

        return out.unsqueeze(1)  # [B, 1, 4096]