import torch
import torch.nn as nn


class GraphProjector(nn.Module):
    def __init__(self, gnn_input_dim, llm_hidden_dim, num_tokens=3, num_transformer_layers=2):
        super().__init__()

        self.num_tokens = num_tokens  # 最终输出 3 个 token
        self.llm_hidden_dim = llm_hidden_dim

        # 1. 两层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(gnn_input_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_hidden_dim,
            nhead=8,
            dim_feedforward=llm_hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

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
        batch_size, max_paths, dim = graph_features.shape
        device = graph_features.device

        # =========================================================
        # 步骤 1: 识别哪些是有效路径 (非0)
        # =========================================================
        # 计算 L2 范数，大于 0 的就是有效数据
        # [Batch, Max_Paths]
        path_norm = torch.norm(graph_features, p=2, dim=-1)
        is_valid = path_norm > 1e-5  # True=有效, False=0向量

        # =========================================================
        # 步骤 2: “物理删除”模拟 —— 排序挤压 (Sorting)
        # =========================================================
        # 这一步至关重要！它把所有有效数据移到了最前面，0数据被赶到了最后面
        # 效果等同于：[A, 0, B, 0] -> [A, B, 0, 0]
        # 这样 Transformer 只会处理前两个，后面会被 Mask 掉

        sorted_mask, sorted_indices = torch.sort(is_valid, dim=1, descending=True)

        # 根据排序后的索引，重排原始特征
        # 此时 graph_features 变成了：前半部分全是有效数据，后半部分全是 0
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_features = torch.gather(graph_features, 1, expanded_indices)

        # =========================================================
        # 步骤 3: MLP 投影
        # =========================================================
        # 此时进去的 sorted_features，前面已经是紧凑的有效向量了
        projected_features = self.mlp(sorted_features)

        # =========================================================
        # 步骤 4: Transformer (带 Mask)
        # =========================================================
        # 我们生成一个 padding_mask，告诉 Transformer：
        # “虽然矩阵是对齐的，但后面那些被我排过去的 0，你完全不要看，不要计算注意力。”
        # True = 忽略 (Mask), False = 保留
        padding_mask = ~sorted_mask

        # 这里的 Transformer 计算时，前排的有效向量只会互相 Attention
        # 根本不会理会后面那些 0 向量，实现了“逻辑上的删除”
        transformer_out = self.transformer(projected_features, src_key_padding_mask=padding_mask)

        # =========================================================
        # 步骤 5: 截取前 3 个
        # =========================================================
        # 因为我们排过序了，前 3 个一定是最有效的
        final_output = transformer_out[:, :self.num_tokens, :]

        # 强制补 0 (如果有效数量 < 3，比如只有1个有效，那第2、3个位置强制置0)
        final_mask = padding_mask[:, :self.num_tokens]
        final_output = final_output.masked_fill(final_mask.unsqueeze(-1), 0.0)

        return final_output