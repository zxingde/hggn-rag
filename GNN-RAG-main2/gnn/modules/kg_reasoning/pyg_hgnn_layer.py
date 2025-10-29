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
        hyperedge_index = pyg_batch.hyperedge_index # Shape [2, num_connections]
        node_batch_ptr = pyg_batch.batch # 节点所属图的 batch vector [total_nodes]
        he_counts_per_graph = pyg_batch.num_hyperedges  # 每个图的超边数 [B], Tensor
        total_hyperedges = he_counts_per_graph.sum().item() # 总超边数, int

        # 安全检查
        if total_hyperedges == 0 or hyperedge_index.numel() == 0:
            return init_node_features_flat

        # 1. 计算超边偏移量
        he_offsets = torch.cat([
            torch.tensor([0], device=he_counts_per_graph.device),
            torch.cumsum(he_counts_per_graph, dim=0)[:-1] # [0, n0, n0+n1, ...] Shape: [B]
        ])

        # 2. 确定每个连接(节点-超边对)属于哪个图
        node_indices = hyperedge_index[0] # 全局节点索引 (已被 PyG 处理), Shape [num_connections]
        connection_batch_ptr = node_batch_ptr[node_indices] # Shape: [num_connections]

        # 3. 计算全局超边索引
        global_he_idx = pyg_batch.global_he_idx
        # --- (关键) 运行时检查：确认预计算的 global_he_idx 是否有效 ---
        # (这个检查现在应该总是通过，除非 get_batch 中计算错误)
        max_global_he_idx_val = global_he_idx.max().item()
        min_global_he_idx_val = global_he_idx.min().item()
        if min_global_he_idx_val < 0 or max_global_he_idx_val >= total_hyperedges:
            raise IndexError(
                f"HGNN Error: Pre-calculated global_he_idx (min={min_global_he_idx_val}, max={max_global_he_idx_val}) "
                f"is out of bounds for total_hyperedges ({total_hyperedges}). Error in get_batch calculation."
            )
        # --- 检查结束 --

        # --- (关键) 运行时检查：确保 global_he_idx 在 [0, total_hyperedges - 1] 范围内 ---
        # (这个检查在我们之前的调试中确认是失败的，但现在 data loading 修复后应该通过)
        max_global_he_idx_val = global_he_idx.max().item()
        min_global_he_idx_val = global_he_idx.min().item()
        if min_global_he_idx_val < 0 or max_global_he_idx_val >= total_hyperedges:
             # 如果这个错误仍然出现，说明 dataset_load.py 的修复还不够或有其他问题
             raise IndexError(
                 f"HGNN Error after manual calculation: global_he_idx (min={min_global_he_idx_val}, max={max_global_he_idx_val}) "
                 f"is out of bounds for total_hyperedges ({total_hyperedges}). Data loading is still flawed."
             )
        # --- 检查结束 ---


        # --- 2. 分散指令到超边 ---
        # (这部分逻辑保持不变)
        he_batch_ptr = torch.repeat_interleave(
            torch.arange(len(he_counts_per_graph), device=init_node_features_flat.device),
            repeats=he_counts_per_graph
        )
        if len(he_batch_ptr) != total_hyperedges:
             print(f"Warning: he_batch_ptr length ({len(he_batch_ptr)}) != total_hyperedges ({total_hyperedges}).")
             if len(he_batch_ptr) > total_hyperedges: he_batch_ptr = he_batch_ptr[:total_hyperedges]
        max_batch_id_needed = he_batch_ptr.max().item() if total_hyperedges > 0 else -1
        if max_batch_id_needed >= current_instructions_avg.shape[0]:
             raise IndexError(f"Instruction index ({max_batch_id_needed}) out of bounds ({current_instructions_avg.shape[0]})")
        if total_hyperedges > 0:
            scattered_instructions = current_instructions_avg[he_batch_ptr]
        else:
            scattered_instructions = torch.empty((0, current_instructions_avg.shape[1]), device=current_instructions_avg.device, dtype=current_instructions_avg.dtype)
        # --- 指令分散结束 ---


        # --- 3. 开始 HGNN 多层推理 ---
        node_features = init_node_features_flat

        for k in range(self.num_layers):
            # a. Node -> Hyperedge aggregation
            max_node_idx_in_he = node_indices.max()
            if max_node_idx_in_he >= node_features.shape[0]:
                 raise IndexError(f"HGNN Error (Layer {k}): node index in hyperedge_index[0] ({max_node_idx_in_he}) >= node_features dim 0 ({node_features.shape[0]}).")

            # ---> 使用手动计算的 global_he_idx 进行聚合 <---
            he_features_aggregated = scatter_mean(
                 node_features[node_indices],         # 源: 节点特征 [num_connections, D]
                 global_he_idx,                       # 索引: 手动计算的全局超边索引 [num_connections]
                 dim=0,
                 dim_size=total_hyperedges            # 输出大小: 全局超边数
            )

            he_features_mlp = self.dropout(F.relu(self.node_to_he_mlps[k](he_features_aggregated)))

            # b. Calculate dynamic weights (逻辑不变)
            if he_features_aggregated.shape[0] != scattered_instructions.shape[0]:
                 if he_features_aggregated.shape[0] < scattered_instructions.shape[0]:
                     padding_size = scattered_instructions.shape[0] - he_features_aggregated.shape[0]
                     padding = torch.zeros(padding_size, he_features_aggregated.shape[1], device=he_features_aggregated.device, dtype=he_features_aggregated.dtype)
                     he_features_aggregated_padded = torch.cat([he_features_aggregated, padding], dim=0)
                 else:
                     he_features_aggregated_padded = he_features_aggregated[:scattered_instructions.shape[0]]
                 combined_reps = torch.cat([he_features_aggregated_padded, scattered_instructions], dim=1)
            else:
                 combined_reps = torch.cat([he_features_aggregated, scattered_instructions], dim=1)
            weights = torch.sigmoid(self.hyperedge_weight_mlp(combined_reps))

            # c. Apply weights
            weighted_he_features = he_features_mlp * weights

            # d. Hyperedge -> Node aggregation
            # ---> 使用手动计算的 global_he_idx 作为源索引 <---
            source_features = weighted_he_features[global_he_idx]

            updated_node_features = scatter_add(
                 source_features,                   # 源: 连接对应的超边特征
                 node_indices,                      # 索引: 目标节点索引
                 dim=0,
                 dim_size=node_features.size(0)     # 输出大小: 全局节点数
            )
            updated_node_features = self.dropout(F.relu(self.he_to_node_mlps[k](updated_node_features)))

            # e. Residual connection
            node_features = node_features + updated_node_features
        # --- HGNN end ---

        return node_features