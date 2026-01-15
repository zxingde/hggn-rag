
import torch
import torch.nn.functional as F
import torch.nn as nn


from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000

class hgnn_reason(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg,
                 hgnn_module=None,  # 接收实例化的 PyGWeightedHGNNLayer
                 fusion_module=None  # 接收实例化的 DiffusionFusion
                 ):
        super(hgnn_reason, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']

        self.use_posemb = args['pos_emb']

        self.hgnn_reasoning = hgnn_module
        self.diffusion_fusion = fusion_module

        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i),
                                nn.Linear(in_features=2 * (self.num_ins) * entity_dim + entity_dim,
                                          out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        self.lin_m = nn.Linear(in_features=(self.num_ins) * entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities,
                    # --- 新增参数 ---
                    pyg_hypergraph_batch=None,
                    node_mask=None,
                    init_entity_emb=None  # 传入最原始的 Padded 嵌入 A
                    # --- 结束新增 ---
                    , query_node_emb=None):

        # (我们之前用于测试的 raise Exception 已经删除)

        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities

        self.pyg_hypergraph_batch = pyg_hypergraph_batch
        self.node_mask = node_mask

        if init_entity_emb is not None and self.pyg_hypergraph_batch is not None:
            ptr = self.pyg_hypergraph_batch.ptr

            # 2. 从 Padded A (B, N_max, D) 中提取切片并拼接
            feature_slices = []
            for i in range(self.batch_size):
                num_nodes_in_this_graph = ptr[i + 1] - ptr[i]  # 计算第 i 个图的节点数
                # 从 (8, 2000, 50) 中切片 [i, :num_nodes, :]
                feature_slices.append(init_entity_emb[i, :num_nodes_in_this_graph, :])

            # 3. 拼接成 (11793, 50) 的张量
            self.init_entity_emb_flat = torch.cat(feature_slices, dim=0)

            # (可选) 安全检查
            if self.init_entity_emb_flat.shape[0] != self.pyg_hypergraph_batch.num_nodes:
                print(
                    f"警告: 节点特征张量维度 ({self.init_entity_emb_flat.shape[0]}) 与 pyg_batch.num_nodes ({self.pyg_hypergraph_batch.num_nodes}) 不匹配！")

            # --- 修正结束 ---

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        """
        Aggregates neighbor representations
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features

        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)

        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel) + pe) * fact_query)
        else:
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)

        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv

        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)

        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb_inv is not None:
            pe = pos_emb_inv(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel) + pe) * fact_query)
        else:
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)

        return neighbor_rep

    def combine(self, emb):
        """
        Combines instruction-specific representations.
        """
        local_emb = torch.cat(emb, dim=-1)
        local_emb = F.relu(self.lin_m(local_emb))

        score_func = self.score_func

        score_tp = score_func(self.linear_drop(local_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        return current_dist, local_emb

    def forward(self, current_dist, relational_ins, step=0, return_score=False):  # 参数不变
        """
        修改后的 forward: 在每一层内部完成 GNN -> HGNN -> Fusion -> 计算下一层分布
        返回: (next_layer_dist, fused_emb) -> (下一层 GNN 输入分布, 当前层融合特征)
        """
        # 获取当前层 j 需要的 GNN 层 (MLP等)
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # 使用本模块内部定义的 score_func
        score_func = self.score_func
        # print(f"--- ReasonGNNLayer step {step} ---") # 调试信息

        # --- 1. GNN 部分 (计算 GNN 更新后的特征 gnn_emb) ---
        neighbor_reps = []
        if self.use_posemb:
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else:
            pos_emb, pos_emb_inv = None, None

        # 计算 GNN 的邻居聚合 (不变)
        for j_ins in range(relational_ins.size(1)):
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:, j_ins, :], rel_linear, pos_emb)
            neighbor_reps.append(neighbor_rep)
            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:, j_ins, :], rel_linear, pos_emb_inv)
            neighbor_reps.append(neighbor_rep)
        neighbor_reps = torch.cat(neighbor_reps, dim=2)

        # 更新 GNN 的节点嵌入状态
        # 输入 self.local_entity_emb 是上一层 j-1 的 GNN 嵌入
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        gnn_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
        # 更新 GNN 内部状态，供下一层 j+1 或下一轮 t+1 使用 (取决于外部如何调用)
        self.local_entity_emb = gnn_emb
        # print(f"  GNN emb shape: {gnn_emb.shape}") # 调试信息

        # --- 2. HGNN 部分 (计算 HGNN 更新后的特征 hgnn_entity_emb_padded) ---
        hgnn_entity_emb_padded = torch.zeros_like(gnn_emb)  # 初始化为零 Padded 张量 A (8, 2000, 50)
        if self.hgnn_reasoning is not None and self.init_entity_emb_flat is not None:
            # current_instructions_avg = torch.mean(relational_ins, dim=1)  # HGNN 需要当前指令
            # === 改进点 1：循环处理每条指令 ===
            num_ins = relational_ins.size(1)
            hgnn_outputs_flat = []  # 存储每次HGNN调用的扁平化输出

            try:
                for ins_idx in range(num_ins):
                    # 1. 获取当前单个指令 [B, D]
                    current_instruction = relational_ins[:, ins_idx, :]

                    # 2. 使用当前指令调用HGNN
                    # 输入: 初始 Flattened 特征 B (11793, 50), *当前*指令
                    hgnn_entity_emb_flat_single = self.hgnn_reasoning(self.init_entity_emb_flat,
                                                                      current_instruction,
                                                                      self.pyg_hypergraph_batch)

                    # 3. 存储单次运行的扁平化结果
                    hgnn_outputs_flat.append(hgnn_entity_emb_flat_single)

                # 4. 聚合多次运行的结果
                # 堆叠: [num_ins, 11793, 50]
                all_hgnn_flat = torch.stack(hgnn_outputs_flat, dim=0)
                # 取平均: [11793, 50]
                hgnn_entity_emb_flat = torch.mean(all_hgnn_flat, dim=0)
                # === 改进结束 ===

                # 输出: hgnn_entity_emb_flat (Flattened 特征 B)
                # print(f"  HGNN emb_flat shape: {hgnn_entity_emb_flat.shape}") # 调试信息

                # --- 修正开始：将 (11793, 50) 转换回 Padded 格式 (8, 2000, 50) ---
                ptr = self.pyg_hypergraph_batch.ptr
                for i in range(self.batch_size):
                    # 计算第 i 个图的节点数
                    num_nodes_in_this_graph = ptr[i + 1] - ptr[i]
                    # 从 (11793, 50) 中切片 [ptr[i]:ptr[i+1], :]
                    flat_slice = hgnn_entity_emb_flat[ptr[i]:ptr[i + 1], :]
                    # 赋值回 (8, 2000, 50) 的 Padded 张量
                    hgnn_entity_emb_padded[i, :num_nodes_in_this_graph, :] = flat_slice
                # --- 修正结束 ---
            except Exception as e:
                print(f"错误发生在 HGNN 推理或格式转换 (step {step}): {e}")
                # 保留 hgnn_entity_emb_padded 为零或采取其他回退措施
        # else: 如果没有 HGNN 模块，hgnn_entity_emb_padded 保持为零
        # print(f"  HGNN emb_padded shape: {hgnn_entity_emb_padded.shape}") # 调试信息

        # --- 3. 融合部分 (融合 gnn_emb 和 hgnn_entity_emb_padded) ---
        if self.diffusion_fusion is not None:
            fused_emb = self.diffusion_fusion(gnn_emb, hgnn_entity_emb_padded)
            # fused_emb 是 Padded 特征 A'
            # print(f"  Fused emb shape: {fused_emb.shape}") # 调试信息
        else:
            # 如果没有融合模块，默认只使用 GNN 特征
            # print(f"  No fusion module, using GNN emb.") # 调试信息
            fused_emb = gnn_emb

        # --- 4. 计算下一层分布 (使用融合后的特征 fused_emb) ---
        layer_scores = score_func(self.linear_drop(fused_emb)).squeeze(dim=2)
        local_entity_mask_float = self.local_entity_mask  # 使用 GNN 内部的 float mask
        layer_scores = layer_scores + (1 - local_entity_mask_float) * VERY_NEG_NUMBER
        next_layer_dist = self.softmax_d1(layer_scores)  # 计算出的下一层 GNN 输入分布
        # print(f"  Next layer dist shape: {next_layer_dist.shape}") # 调试信息

        # 返回下一层需要的分布，以及当前层融合后的特征 (用于外部的指令更新)
        # return_score 参数不再需要，因为分布总是在内部计算
        return next_layer_dist, fused_emb, gnn_emb, hgnn_entity_emb_padded