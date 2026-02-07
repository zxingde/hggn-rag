import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphProjector(nn.Module):
    def __init__(self, input_dim=50, output_dim=4096):
        super().__init__()
        # 1. 投影层
        self.linear = nn.Linear(input_dim, output_dim)

        # 2. 注意力查询向量
        # 优化 1: 使用较小的初始化范围，防止 Softmax 饱和
        self.query = nn.Parameter(torch.randn(output_dim) * 0.02)

        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        # x: [B, N, 50]
        # mask: [B, N]

        # [B, N, 50] -> [B, N, 4096]
        h = self.act(self.linear(x))

        # --- Attention Pooling ---
        # scores: [B, N]
        scores = torch.matmul(h, self.query)

        # 处理 Mask
        if mask is not None:
            # �� 核心修改：改回 -1e4。不要用 min_value，太小会导致 NaN 梯度。
            scores = scores.masked_fill(mask == 0, -1e4)

        # 优化 2: 防止 softmax 在全 Mask 情况下输出 NaN
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]

        # 加权求和
        out = torch.sum(h * attn_weights, dim=1)  # [B, 4096]

        return out.unsqueeze(1)  # [B, 1, 4096]