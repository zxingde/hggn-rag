# modules/kg_reasoning/hyper_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        """
        初始化一个 **加权** HyperGNN 层。
        """
        super(HyperGNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 定义两个线性变换层
        self.node_to_hyperedge_mlp = nn.Linear(in_features, out_features)
        self.hyperedge_to_node_mlp = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_features, batch_hyperedges, hyperedge_weights=None):
        """
        加权 HyperGNN 的前向传播。
        """
        batch_updated_features = []
        for i in range(node_features.size(0)):
            current_node_features = node_features[i]
            hyperedges = batch_hyperedges[i]
            num_nodes = current_node_features.size(0)
            num_hyperedges = len(hyperedges)  # 获取当前图的实际超边数量

            if num_hyperedges == 0 or num_nodes == 0:
                batch_updated_features.append(current_node_features.unsqueeze(0))
                continue

            # --- 1. 构建关联矩阵 ---
            incidence_matrix_rows = []
            incidence_matrix_cols = []
            for edge_idx, edge in enumerate(hyperedges):
                for node_idx in edge:
                    if node_idx < num_nodes:
                        incidence_matrix_rows.append(node_idx)
                        incidence_matrix_cols.append(edge_idx)

            if not incidence_matrix_rows:
                batch_updated_features.append(current_node_features.unsqueeze(0))
                continue

            H = torch.sparse_coo_tensor(
                [incidence_matrix_rows, incidence_matrix_cols],
                torch.ones(len(incidence_matrix_rows)),
                (num_nodes, num_hyperedges)
            ).to(node_features.device)

            # --- 2. 节点 -> 超边 聚合 ---
            hyperedge_features = torch.sparse.mm(H.t(), current_node_features)
            hyperedge_features = self.dropout(F.relu(self.node_to_hyperedge_mlp(hyperedge_features)))

            # --- 3. (核心修改) 应用动态权重 ---
            if hyperedge_weights is not None:
                # <-- 修改点：只取当前图实际需要的权重数量
                current_weights = hyperedge_weights[i, :num_hyperedges]  # 从 (976,) 切片为 (293,)
                # <-- 修改结束

                hyperedge_features = hyperedge_features * current_weights.unsqueeze(1)

            # --- 4. 超边 -> 节点 聚合 ---
            updated_node_features = torch.sparse.mm(H, hyperedge_features)
            updated_node_features = self.dropout(F.relu(self.hyperedge_to_node_mlp(updated_node_features)))

            # 添加残差连接
            updated_node_features = updated_node_features + current_node_features
            batch_updated_features.append(updated_node_features.unsqueeze(0))

        return torch.cat(batch_updated_features, dim=0)