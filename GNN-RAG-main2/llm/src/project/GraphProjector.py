import torch
import torch.nn as nn


class GraphProjector(nn.Module):
    def __init__(self, gnn_input_dim, llm_hidden_dim, num_tokens=3, num_transformer_layers=2):
        super().__init__()
        self.num_tokens = num_tokens  # 3
        self.llm_hidden_dim = llm_hidden_dim

        # 1. 投影层 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(gnn_input_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

        # 2. Transformer 编码器 (保留推理能力!)
        # 我们使用标准的 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_hidden_dim,
            nhead=8,  # 8头注意力，足够捕捉不同关系的交互
            dim_feedforward=llm_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-Norm 结构，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 初始化权重，防止初始梯度过大
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, graph_features):
        """
        graph_features: [Batch, Max_Paths, 50]
        """
        batch_size, seq_len, dim = graph_features.shape
        device = graph_features.device
        dtype = graph_features.dtype
        target_len = self.num_tokens  # 3

        # 0. 【保险丝】输入清洗：防止上游传过来 NaN
        if torch.isnan(graph_features).any():
            graph_features = torch.nan_to_num(graph_features, nan=0.0)

        # =========================================================
        # 1. 筛选策略：Top-K 截取 (不使用 Mask，物理截取)
        # =========================================================
        # 我们需要从 N 条路径里选出 3 条。
        # 策略：计算模长 (Norm)，选模长最大的前 3 个。
        # (因为 0 向量的模长是 0，肯定会被排到后面去)

        path_norm = torch.norm(graph_features, p=2, dim=-1)  # [Batch, N]

        # 排序：从大到小
        _, sorted_indices = torch.sort(path_norm, dim=1, descending=True)

        # 只要前 3 个索引
        # 如果实际路径少于 3 个，这里取模或者 clamp 可能会复杂。
        # 简单方案：直接对 graph_features 进行物理排序
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_features = torch.gather(graph_features, 1, expanded_indices)  # [Batch, N, 50]

        # 物理截取前 3 个
        if seq_len >= target_len:
            selected_features = sorted_features[:, :target_len, :]
        else:
            # 如果一共就不到 3 条路径，那就全拿，剩下的补 0
            diff = target_len - seq_len
            zeros = torch.zeros((batch_size, diff, dim), device=device, dtype=dtype)
            selected_features = torch.cat([sorted_features, zeros], dim=1)

        # 现在的 selected_features 形状固定为 [Batch, 3, 50]
        # 且不仅保留了最有价值的路径，还去掉了大部分 0 向量

        # =========================================================
        # 2. 投影 + Transformer 交互
        # =========================================================
        projected_features = self.mlp(selected_features)  # [Batch, 3, 4096]

        # 【关键点】不传 mask！
        # 让 Transformer 自己去学。如果第 3 条路径是补的 0，Transformer
        # 会发现它与其他路径没有相关性，Attention 权重自然会很低。
        # 这样做绝对不会除以 0，绝对不会 NaN。
        transformer_out = self.transformer(projected_features)

        return transformer_out