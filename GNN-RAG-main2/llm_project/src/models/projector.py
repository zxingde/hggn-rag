import torch
import torch.nn as nn


class StructureProjector(nn.Module):
    def __init__(self, gnn_dim=50, llm_dim=4096, num_tokens=2):
        """
        Args:
            gnn_dim (int): GNN 输出的特征维度 (例如 50)
            llm_dim (int): LLM 的 Embedding 维度 (例如 4096)
            num_tokens (int): 你希望输出的固定 Token 数量 (这里是 2)
        """
        super().__init__()
        self.num_tokens = num_tokens

        # 1. 特征维度对齐: 先把 50维 映射到 4096维
        # 这一步是逐节点进行的 (Pointwise Linear)
        self.input_proj = nn.Linear(gnn_dim, llm_dim)

        # 2. 定义核心组件: 可学习的 Query 向量 (The "Probes")
        # 形状: [1, 2, 4096] -> 这就是你想要的“固定的2个Token”的种子
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, llm_dim))

        # 3. Cross-Attention 层
        # Query 是固定的 2 个向量，Key/Value 是你不固定数量的节点特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=8,  # 8个头关注不同方面
            batch_first=True  # 输入形状是 [Batch, Seq, Dim]
        )

        # 4. 输出前的归一化，有助于训练稳定
        self.norm = nn.LayerNorm(llm_dim)

        # 初始化参数 (可选，但推荐)
        nn.init.normal_(self.query_tokens, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, partial_node_features):
        """
        Args:
            partial_node_features: 选出来的部分节点特征
            Shape: [Batch_Size, Dynamic_Num_Nodes, gnn_dim]

            注意：Dynamic_Num_Nodes 可以是变长的。
            如果 Batch 内样本的节点数不同 (比如样本1有3个节点，样本2有5个)，
            你需要把它们 Pad 到相同长度，然后传入 key_padding_mask (见下文进阶用法)，
            或者你的 Batch Size = 1 就完全不用管 Padding。

        Returns:
            fixed_tokens: [Batch_Size, 2, llm_dim] -> 固定的 2 个 Token
        """
        batch_size = partial_node_features.shape[0]

        # 步骤 1: 投影 GNN 特征 (Feature Projection)
        # Input: [B, N, 50] -> Output: [B, N, 4096]
        # N 是动态的，Linear 层不在乎 N 是多少
        key_value = self.input_proj(partial_node_features)

        # 步骤 2: 扩展 Query (Expand Queries)
        # 让 Query 适应当前的 Batch Size
        # [1, 2, 4096] -> [B, 2, 4096]
        queries = self.query_tokens.expand(batch_size, -1, -1)

        # 步骤 3: Cross Attention 聚合 (Aggregation)
        # Query (2个) 去"查询" Key (N个) 的信息
        # 运算结果维度只取决于 Query 的长度
        # Output: [B, 2, 4096]
        out, _ = self.cross_attn(
            query=queries,
            key=key_value,
            value=key_value
        )

        # 步骤 4: 归一化
        return self.norm(out)