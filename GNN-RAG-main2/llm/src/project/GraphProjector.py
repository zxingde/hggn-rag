import torch
import torch.nn as nn

class GraphProjector(nn.Module):
    def __init__(self, gnn_dim=50, llm_dim=4096):
        super().__init__()
        # 第一层：升维到一个中间层
        self.projector = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim // 2),
            nn.GELU(),
            nn.Linear(llm_dim // 2, llm_dim),
            # 增加 LayerNorm 有助于稳定注入后的 Embedding 分布
            nn.LayerNorm(llm_dim)
        )

    def forward(self, x):
        # x: [Batch, 6, 50] -> [Batch, 6, 4096]
        return self.projector(x)