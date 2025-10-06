import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        """
        初始化一个 HyperGNN 层。
        in_features: 输入的节点特征维度
        out_features: 输出的节点特征维度
        """
        super(HyperGNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 定义两个线性变换层
        self.node_to_hyperedge_mlp = nn.Linear(in_features, out_features)
        self.hyperedge_to_node_mlp = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_features, batch_hyperedges):
        """
        HyperGNN 的前向传播。

        输入:
            node_features (Tensor): 批次中所有节点的特征, 形状 (batch_size, num_nodes, feature_dim)
            batch_hyperedges (list[list[tuple]]): 我们之前构建的超边列表
        输出:
            updated_node_features (Tensor): 经过 HyperGNN 更新后的节点特征
        """

        batch_updated_features = []
        # 逐个处理 batch 中的每个图
        for i in range(node_features.size(0)):
            current_node_features = node_features[i]  # (num_nodes, feature_dim)
            hyperedges = batch_hyperedges[i]

            num_nodes = current_node_features.size(0)

            if not hyperedges or num_nodes == 0:
                batch_updated_features.append(current_node_features.unsqueeze(0))
                continue

            # --- 1. 构建关联矩阵 (Incidence Matrix) ---
            # 这个矩阵描述了节点和超边之间的关系
            # 矩阵的行代表节点，列代表超边
            incidence_matrix_rows = []
            incidence_matrix_cols = []

            for edge_idx, edge in enumerate(hyperedges):
                for node_idx in edge:
                    if node_idx < num_nodes:  # 安全检查，防止索引越界
                        incidence_matrix_rows.append(node_idx)
                        incidence_matrix_cols.append(edge_idx)

            # 如果没有有效的节点-超边关系，则跳过
            if not incidence_matrix_rows:
                batch_updated_features.append(current_node_features.unsqueeze(0))
                continue

            # 使用稀疏矩阵来节省内存
            H = torch.sparse_coo_tensor(
                [incidence_matrix_rows, incidence_matrix_cols],
                torch.ones(len(incidence_matrix_rows)),
                (num_nodes, len(hyperedges))
            ).to(node_features.device)

            # --- 2. 节点 -> 超边 聚合 ---
            # (H^T @ X_v)
            # H.T 是 (num_edges, num_nodes)
            # current_node_features 是 (num_nodes, feature_dim)
            # 结果 hyperedge_features 是 (num_edges, feature_dim)
            # 这一步相当于对每个超边，将其所有内部节点的特征相加
            hyperedge_features = torch.sparse.mm(H.t(), current_node_features)

            # 对超边特征进行非线性变换和dropout
            hyperedge_features = self.dropout(F.relu(self.node_to_hyperedge_mlp(hyperedge_features)))

            # --- 3. 超边 -> 节点 聚合 ---
            # (H @ X_e)
            # H 是 (num_nodes, num_edges)
            # hyperedge_features 是 (num_edges, feature_dim)
            # 结果 updated_node_features 是 (num_nodes, feature_dim)
            # 这一步相当于对每个节点，将其所在的所有超边的特征相加
            updated_node_features = torch.sparse.mm(H, hyperedge_features)

            # 对更新后的节点特征进行非线性变换和dropout
            updated_node_features = self.dropout(F.relu(self.hyperedge_to_node_mlp(updated_node_features)))

            # 添加残差连接 (Residual Connection)，防止梯度消失，让模型更稳定
            updated_node_features = updated_node_features + current_node_features

            batch_updated_features.append(updated_node_features.unsqueeze(0))

        # 将 list 合并回一个 batch tensor
        return torch.cat(batch_updated_features, dim=0)