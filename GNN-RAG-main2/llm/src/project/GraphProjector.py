import torch
import torch.nn as nn


class GraphProjector(nn.Module):
    def __init__(self, gnn_dim, llm_dim):
        super().__init__()
        # 1. 【关键】归一化层：彻底解决 Norm=65 vs Norm=1 的问题
        # 无论 GNN 出来的数值多大，先压成标准分布
        self.norm = nn.LayerNorm(gnn_dim)

        # 2. 双层 MLP：增强特征的语义转换能力
        # 结构：Linear -> GELU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim),
            nn.GELU(),  # 激活函数
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, num_nodes, gnn_dim]

        # 先做归一化
        x = self.norm(x)

        # 再做投影
        x = self.mlp(x)
        return x