import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean, scatter_add

class PyGWeightedHGNNLayer(MessagePassing):
    """
    基于 PyG 实现的、带动态权重的多层 HGNN 推理模块 (框架)。
    """
    def __init__(self, in_dim, out_dim, num_layers, dropout, instruction_dim):
        super(PyGWeightedHGNNLayer, self).__init__(aggr=None) # 手动聚合
        print(f"Initializing PyGWeightedHGNNLayer with {num_layers} layers.")
        # 这里可以先定义必要的层，例如 MLP 等，但内部逻辑暂时为空
        self.entity_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        # 占位：定义多层 MLP (此处仅为示例结构)
        self.node_to_he_mlps = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_layers)])
        self.he_to_node_mlps = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_layers)])
        self.hyperedge_weight_mlp = nn.Linear(in_dim + instruction_dim, 1) # 动态权重 MLP

    def forward(self, init_node_features_flat, current_instructions_avg, pyg_batch):
        """
        HGNN 前向传播 (完整实现)。
        输入:
            init_node_features_flat: [TotalNodes, D] - 初始 Flattened 节点嵌入 B
            current_instructions_avg: [B, D_ins] - 当前迭代的平均指令
            pyg_batch: PyG Batch 对象 (包含 hyperedge_index, batch, num_hyperedges)
        输出:
            node_features: [TotalNodes, D] - 更新后的 Flattened 节点嵌入 B
        """
        # print("Running PyGWeightedHGNNLayer forward pass...") # (调试时可以取消注释)

        hyperedge_index = pyg_batch.hyperedge_index
        node_batch_ptr = pyg_batch.batch
        he_counts_per_graph = pyg_batch.num_hyperedges  # [B]
        total_hyperedges = pyg_batch.num_hyperedges.sum().item()

        # 安全检查: 如果没有超边或连接，直接返回初始特征
        if total_hyperedges == 0 or hyperedge_index.numel() == 0:
            # print("No hyperedges found, returning initial features.") # (调试时可以取消注释)
            return init_node_features_flat

        # --- 1. 计算全局超边索引 global_he_idx ---
        he_offsets = torch.cat([
            torch.tensor([0], device=he_counts_per_graph.device),
            torch.cumsum(he_counts_per_graph, dim=0)[:-1]
        ])
        # 确定每个连接(边)属于哪个图
        connection_batch_ptr = node_batch_ptr[hyperedge_index[0]]
        # 计算全局索引
        global_he_idx = hyperedge_index[1] + he_offsets[connection_batch_ptr]
        # --- 结束计算全局索引 ---

        # --- 2. 分散指令到超边 ---
        he_batch_ptr = torch.repeat_interleave(
            torch.arange(len(he_counts_per_graph), device=init_node_features_flat.device),
            repeats=he_counts_per_graph
        )
        # 确保 he_batch_ptr 的长度与 total_hyperedges 匹配 (可能在空图时出错)
        if len(he_batch_ptr) != total_hyperedges:
            print(
                f"警告: he_batch_ptr 长度 ({len(he_batch_ptr)}) 与 total_hyperedges ({total_hyperedges}) 不匹配。检查 he_counts_per_graph。")
            # 可能需要更复杂的处理来跳过没有超边的图的指令
            # 暂时假设不会发生或影响不大
            if len(he_batch_ptr) > total_hyperedges:
                he_batch_ptr = he_batch_ptr[:total_hyperedges]  # 尝试截断
            # 如果更短，问题更严重，暂时忽略

        # 检查 current_instructions_avg 的维度是否足够
        if current_instructions_avg.shape[0] < (he_batch_ptr.max().item() + 1):
            raise IndexError(
                f"指令索引 ({he_batch_ptr.max().item()}) 超出了指令张量维度 ({current_instructions_avg.shape[0]})")

        scattered_instructions = current_instructions_avg[he_batch_ptr]
        # --- 结束分散指令 ---

        # --- 3. 开始 HGNN 多层推理 ---
        node_features = init_node_features_flat  # 始终从初始嵌入开始

        for k in range(self.num_layers):
            # a. 节点 -> 超边聚合 (聚合得到 he_features_aggregated)
            # 安全检查: 确保 hyperedge_index[0] 不会超出 node_features 的边界
            max_node_idx_in_he = hyperedge_index[0].max()
            if max_node_idx_in_he >= node_features.shape[0]:
                raise IndexError(
                    f"HGNN Error (Layer {k}): node index in hyperedge_index ({max_node_idx_in_he}) is out of bounds for node_features dimension 0 ({node_features.shape[0]}).")

            he_features_aggregated = scatter_mean(
                node_features[hyperedge_index[0]],  # 源: 节点特征
                global_he_idx,  # 索引: 全局超边 ID
                dim=0,
                dim_size=total_hyperedges  # 输出大小: 全局超边数
            )
            # 应用 MLP
            he_features_mlp = self.dropout(F.relu(self.node_to_he_mlps[k](he_features_aggregated)))

            # b. 计算动态权重 (使用聚合后的 he_features_aggregated)
            # 检查维度是否匹配
            if he_features_aggregated.shape[0] != scattered_instructions.shape[0]:
                # 处理可能的维度不匹配问题 (之前已经有初步处理)
                if he_features_aggregated.shape[0] < scattered_instructions.shape[0]:
                    padding_size = scattered_instructions.shape[0] - he_features_aggregated.shape[0]
                    padding = torch.zeros(padding_size, he_features_aggregated.shape[1],
                                          device=he_features_aggregated.device, dtype=he_features_aggregated.dtype)
                    he_features_aggregated_padded = torch.cat([he_features_aggregated, padding], dim=0)
                    print(f"Warning: Padded he_features_aggregated in weight calculation (Layer {k})")
                else:  # he_features_aggregated 更长，通常不应该发生
                    he_features_aggregated_padded = he_features_aggregated[:scattered_instructions.shape[0]]
                    print(f"Warning: Truncated he_features_aggregated in weight calculation (Layer {k})")

                combined_reps = torch.cat([he_features_aggregated_padded, scattered_instructions], dim=1)
            else:
                combined_reps = torch.cat([he_features_aggregated, scattered_instructions], dim=1)

            weights = torch.sigmoid(self.hyperedge_weight_mlp(combined_reps))

            # c. 应用权重到 MLP 处理后的特征上
            weighted_he_features = he_features_mlp * weights

            # d. 超边 -> 节点聚合 (聚合加权后的 weighted_he_features)
            # 安全检查: 确保 global_he_idx 不会超出 weighted_he_features 的边界
            max_global_he_idx = global_he_idx.max()
            if max_global_he_idx >= total_hyperedges:
                raise IndexError(
                    f"HGNN Error (Layer {k}): global_he_idx ({max_global_he_idx}) >= total_hyperedges ({total_hyperedges}) for scatter_add source.")

            source_features = weighted_he_features[global_he_idx]

            updated_node_features = scatter_add(
                source_features,
                hyperedge_index[0],  # 索引: 全局节点 ID
                dim=0,
                dim_size=node_features.size(0)  # 输出大小: 全局节点数
            )
            # 应用 MLP
            updated_node_features = self.dropout(F.relu(self.he_to_node_mlps[k](updated_node_features)))

            # e. 残差连接
            node_features = node_features + updated_node_features
        # --- HGNN 多层推理结束 ---

        # print(f"Finished PyGWeightedHGNNLayer forward pass. Output shape: {node_features.shape}") # (调试时可以取消注释)
        return node_features  # 返回最终更新的 Flattened 特征 B