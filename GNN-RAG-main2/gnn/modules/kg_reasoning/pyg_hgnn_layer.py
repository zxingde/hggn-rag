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

        # 安全检查
        if total_hyperedges == 0 or pyg_batch.hyperedge_index.numel() == 0:
            return init_node_features_flat

        # --- MODIFICATION START ---
        # --- 1. 直接使用 PyG Batching 后的索引 ---
        node_indices = pyg_batch.hyperedge_index[0]  # 全局节点索引 (已被 PyG 处理)
        hyperedge_target_indices = pyg_batch.hyperedge_index[1]  # <--- 假设这是 PyG 处理后的全局超边索引

        # --- （关键）运行时检查：确保 hyperedge_target_indices 在 [0, total_hyperedges - 1] 范围内 ---
        max_target_idx = hyperedge_target_indices.max().item()
        min_target_idx = hyperedge_target_indices.min().item()
        if min_target_idx < 0 or max_target_idx >= total_hyperedges:
            raise IndexError(
                f"HGNN Error: PyG's hyperedge_index[1] (min={min_target_idx}, max={max_target_idx}) "
                f"is out of bounds for total_hyperedges ({total_hyperedges}). Data loading is flawed."
            )
        # --- 检查结束 ---
        # --- MODIFICATION END ---

        # --- 2. 分散指令到超边 ---
        # 我们需要知道每个 *全局* 超边属于哪个图 (0..B-1)
        # `he_batch_ptr` 的原始计算是正确的，它的大小是 total_hyperedges
        he_batch_ptr = torch.repeat_interleave(
            torch.arange(len(he_counts_per_graph), device=init_node_features_flat.device),
            repeats=he_counts_per_graph
        )
        if len(he_batch_ptr) != total_hyperedges:  # 保持长度检查
            print(f"Warning: he_batch_ptr length ({len(he_batch_ptr)}) != total_hyperedges ({total_hyperedges}).")
            if len(he_batch_ptr) > total_hyperedges: he_batch_ptr = he_batch_ptr[:total_hyperedges]

        # 分散指令 (检查索引范围)
        max_batch_id_needed = he_batch_ptr.max().item() if total_hyperedges > 0 else -1
        if max_batch_id_needed >= current_instructions_avg.shape[0]:
            raise IndexError(
                f"Instruction index ({max_batch_id_needed}) out of bounds ({current_instructions_avg.shape[0]})")

        if total_hyperedges > 0:
            scattered_instructions = current_instructions_avg[he_batch_ptr]  # Shape: [total_hyperedges, D_ins]
        else:
            scattered_instructions = torch.empty((0, current_instructions_avg.shape[1]),
                                                 device=current_instructions_avg.device,
                                                 dtype=current_instructions_avg.dtype)
        # --- 指令分散结束 ---

        # --- 3. 开始 HGNN 多层推理 ---
        node_features = init_node_features_flat

        for k in range(self.num_layers):
            # a. Node -> Hyperedge aggregation
            max_node_idx_in_he = node_indices.max()
            if max_node_idx_in_he >= node_features.shape[0]:
                raise IndexError(
                    f"HGNN Error (Layer {k}): node index in hyperedge_index[0] ({max_node_idx_in_he}) >= node_features dim 0 ({node_features.shape[0]}).")

            # ---> 使用 PyG 的 hyperedge_target_indices 进行聚合 <---
            he_features_aggregated = scatter_mean(
                node_features[node_indices],  # 源: 节点特征 [num_connections, D]
                hyperedge_target_indices,  # 索引: PyG 处理后的全局超边索引 [num_connections]
                dim=0,
                dim_size=total_hyperedges  # 输出大小: 全局超边数
            )  # 输出 Shape: [total_hyperedges, D]

            he_features_mlp = self.dropout(F.relu(self.node_to_he_mlps[k](he_features_aggregated)))

            # b. Calculate dynamic weights (逻辑不变)
            # (维度检查和处理保持不变)
            if he_features_aggregated.shape[0] != scattered_instructions.shape[0]:
                # ... (之前的 padding/truncating 逻辑) ...
                if he_features_aggregated.shape[0] < scattered_instructions.shape[0]:
                    padding_size = scattered_instructions.shape[0] - he_features_aggregated.shape[0]
                    padding = torch.zeros(padding_size, he_features_aggregated.shape[1],
                                          device=he_features_aggregated.device, dtype=he_features_aggregated.dtype)
                    he_features_aggregated_padded = torch.cat([he_features_aggregated, padding], dim=0)
                else:
                    he_features_aggregated_padded = he_features_aggregated[:scattered_instructions.shape[0]]
                combined_reps = torch.cat([he_features_aggregated_padded, scattered_instructions], dim=1)
            else:
                combined_reps = torch.cat([he_features_aggregated, scattered_instructions], dim=1)
            weights = torch.sigmoid(self.hyperedge_weight_mlp(combined_reps))  # Shape: [total_hyperedges, 1]

            # c. Apply weights
            weighted_he_features = he_features_mlp * weights  # Shape: [total_hyperedges, D]

            # d. Hyperedge -> Node aggregation
            # ---> 使用 PyG 的 hyperedge_target_indices 作为源索引 <---
            # 从加权后的全局超边特征中，根据连接关系取出对应的特征
            source_features = weighted_he_features[hyperedge_target_indices]  # Shape: [num_connections, D]

            updated_node_features = scatter_add(
                source_features,  # 源: 连接对应的超边特征
                node_indices,  # 索引: 目标节点索引
                dim=0,
                dim_size=node_features.size(0)  # 输出大小: 全局节点数
            )  # 输出 Shape: [total_nodes, D]
            updated_node_features = self.dropout(F.relu(self.he_to_node_mlps[k](updated_node_features)))

            # e. Residual connection
            node_features = node_features + updated_node_features
        # --- HGNN end ---

        return node_features