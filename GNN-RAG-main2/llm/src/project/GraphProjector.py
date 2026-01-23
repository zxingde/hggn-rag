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
        dtype = graph_features.dtype

        # 1. 识别有效路径 (True/False)
        path_norm = torch.norm(graph_features, p=2, dim=-1)
        is_valid = path_norm > 1e-5

        # 2. 排序 (Sort)
        # 【技巧】先转 long 排序(避开CUDA bug)，再转回 bool 给 Transformer 用
        sorted_mask_int, sorted_indices = torch.sort(is_valid.long(), dim=1, descending=True)
        sorted_mask_bool = sorted_mask_int.bool()  # True=有效, False=0向量

        # 根据索引重排特征
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_features = torch.gather(graph_features, 1, expanded_indices)

        # 3. MLP
        projected_features = self.mlp(sorted_features)

        # 4. Transformer
        # 生成 Padding Mask: True 表示要被忽略 (即 sorted_mask_bool 为 False 的部分)
        padding_mask = ~sorted_mask_bool

        transformer_out = self.transformer(projected_features, src_key_padding_mask=padding_mask)

        # =========================================================
        # 5. 截取 + 维度强制对齐 (核心修复)
        # =========================================================
        curr_seq_len = transformer_out.shape[1]

        if curr_seq_len >= self.num_tokens:
            # 路径足够多，直接截取前 3 个
            final_output = transformer_out[:, :self.num_tokens, :]
            final_mask = padding_mask[:, :self.num_tokens]
        else:
            # 路径不够 (比如只有1条)，需要硬凑到 3 条
            # 1. 拿出现有的
            final_output = transformer_out

            # 2. 生成全 0 的补丁
            diff = self.num_tokens - curr_seq_len
            zeros = torch.zeros((batch_size, diff, self.llm_hidden_dim), device=device, dtype=dtype)

            # 3. 拼上去 -> [Batch, 3, 4096]
            final_output = torch.cat([final_output, zeros], dim=1)

            # 4. Mask 也要补 (补上的部分全是 True-忽略)
            ones_mask = torch.ones((batch_size, diff), device=device, dtype=torch.bool)
            final_mask = torch.cat([padding_mask, ones_mask], dim=1)

        # 6. 最终清洗 (把 Mask 掉的位置强制置 0)
        final_output = final_output.masked_fill(final_mask.unsqueeze(-1), 0.0)

        return final_output