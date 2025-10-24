import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import math

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform, DiffusionFusion
from modules.hypergraph_construction.relation_community_constructor import RelationCommunityConstructor
# from modules.kg_reasoning.hyper_gnn import HyperGNNLayer # <--- 删除这一行
from torch_geometric.nn import MessagePassing # <--- 添加这一行
from torch_scatter import scatter_mean, scatter_add # <--- 添加这一行

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class PyGHyperGNNReasoningLayer(MessagePassing):
    """
    使用 PyG MessagePassing 和 torch_scatter 实现的
    完全向量化的、带动态权重的超图推理层。
    """

    def __init__(self, in_dim, out_dim, num_layers, dropout):
        super(PyGHyperGNNReasoningLayer, self).__init__(aggr=None)  # 我们手动聚合
        self.entity_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        # 每一层都有自己的MLP
        self.node_to_he_mlps = nn.ModuleList()
        self.he_to_node_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.node_to_he_mlps.append(nn.Linear(in_dim, out_dim))
            self.he_to_node_mlps.append(nn.Linear(in_dim, out_dim))

        # 动态权重 MLP
        self.hyperedge_weight_mlp = nn.Linear(in_dim * 2, 1)

    def forward(self, init_entity_emb, current_instructions_avg, pyg_batch):
        # pyg_batch 包含合并后的图数据
        # hyperedge_index [2, TotalConnections]: (row 0: node_idx, row 1: he_idx)
        hyperedge_index = pyg_batch.hyperedge_index
        # node_batch_ptr [TotalNodes]: [0, 0, ..., 1, 1, ..., B-1, B-1]
        node_batch_ptr = pyg_batch.batch

        # 1. 创建超边的批处理指针 (he_batch_ptr)
        # he_counts_per_graph [B]: [num_he_in_G0, num_he_in_G1, ...]
        he_counts_per_graph = pyg_batch.num_hyperedges
        he_batch_ptr = torch.repeat_interleave(
            torch.arange(len(he_counts_per_graph), device=init_entity_emb.device),
            repeats=he_counts_per_graph
        )  # Shape: [TotalHyperedges]

        # 2. 将指令 "分散" (Scatter) 到所有超边
        # current_instructions_avg [B, D] -> scattered_instructions [TotalHyperedges, D]
        scattered_instructions = current_instructions_avg[he_batch_ptr]

        # --- 开始HGNN多层推理 ---
        node_features = init_entity_emb  # 始终从初始嵌入开始
        total_hyperedges = pyg_batch.num_hyperedges.sum()
        for k in range(self.num_layers):
            # 3. 步骤 a: 节点 -> 超边 (Node-to-Hyperedge)
            # 收集所有节点特征，按其所属的超边ID (hyperedge_index[1]) 进行平均
            he_features = scatter_mean(
                node_features[hyperedge_index[0]],  # 源: 节点特征
                hyperedge_index[1],  # 索引: 超边ID
                dim=0, # 沿着节点维度聚合
                dim_size = total_hyperedges
            )  # Shape: [TotalHyperedges, D]

            he_features = self.dropout(F.relu(self.node_to_he_mlps[k](he_features)))

            # 4. 步骤 b: 计算并应用动态权重
            combined_reps = torch.cat([he_features, scattered_instructions], dim=1)
            weights = torch.sigmoid(self.hyperedge_weight_mlp(combined_reps))
            weighted_he_features = he_features * weights

            # 5. 步骤 c: 超边 -> 节点 (Hyperedge-to-Node)
            # 收集所有加权后的超边特征，按其连接的节点ID (hyperedge_index[0]) 进行求和
            updated_node_features = scatter_add(
                weighted_he_features[hyperedge_index[1]],  # 源: 加权的超边特征
                hyperedge_index[0],  # 索引: 节点ID
                dim=0,  # 沿着超边维度聚合
                dim_size=node_features.size(0)  # 确保输出张量大小正确
            )  # Shape: [TotalNodes, D]

            updated_node_features = self.dropout(F.relu(self.he_to_node_mlps[k](updated_node_features)))

            # 6. 残差连接
            node_features = node_features + updated_node_features

        return node_features


    def forward(self, batch, training=False):
        """
        修改后的 Forward 流程: ( 指令更新 -> ( (GNN || HGNN) -> Fusion -> 更新分布 ) x L层 ) x T次
        """
        # --- 步骤 1: 解包并转换输入数据 (不变) ---
        local_entity, query_entities, kb_adj_mat, pyg_hypergraph_batch, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        initial_dist = Variable(seed_dist, requires_grad=False)  # 初始种子分布
        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        batch_size = local_entity.size(0)

        # --- 步骤 2: 初始化推理环境 (获取初始指令和嵌入) (不变) ---
        self.init_reason(curr_dist=initial_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins

        # --- 步骤 3: 准备 PyG 和格式转换 (不变) ---
        pyg_hypergraph_batch = pyg_hypergraph_batch.to(self.device)
        node_mask = (local_entity != self.num_entity)
        # HGNN 始终使用初始的 Flattened 特征 B 作为节点输入
        init_entity_emb_flat = self.init_entity_emb[node_mask]

        # --- 步骤 4: 迭代推理 ---
        # current_fused_emb 用于指令更新，在每次 t 循环的 j 循环结束后更新
        self.current_fused_emb = self.init_entity_emb
        self.layer_input_dist = self.initial_dist
        # current_dist 用于 GNN 输入和最终预测，在每次 j 循环内部更新
        self.current_dist = self.initial_dist
        self.dist_history = [self.current_dist]  # 记录每次外层迭代 t 结束时的最终分布

        for t in range(self.num_iter):  # 外层循环 T (更新指令)

            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            current_instructions_avg = torch.mean(relation_ins, dim=1)  # (B, D) - 用于 HGNN


            for j in range(self.num_gnn):  # 内层循环 L (图卷积层)

                # --- 4a: GNN 推理 ---
                # 输入: 上一层的分布 layer_input_dist
                # 输出: gnn_dist (本层计算出的分布), gnn_emb (本层更新后的 Padded 特征 A)
                _, gnn_emb = self.reasoning(self.layer_input_dist, relation_ins, step=j)

                # --- 4b: HGNN 推理 ---
                # 输入: 初始的 Flattened 特征 B, 当前的指令
                # 输出: hgnn_entity_emb_flat (本层更新后的 Flattened 特征 B)
                hgnn_entity_emb_flat = self.hgnn_reasoning(self.current_fused_emb,
                                                           current_instructions_avg,
                                                           pyg_hypergraph_batch)

                # --- 4c: 转换 HGNN 输出回 Padded 格式 A ---
                hgnn_entity_emb_padded = torch.zeros_like(self.init_entity_emb)
                hgnn_entity_emb_padded[node_mask] = hgnn_entity_emb_flat

                # --- 4d: 融合 GNN 和 HGNN 的 *特征* ---
                self.current_fused_emb = self.diffusion_fusion(gnn_emb, hgnn_entity_emb_padded)
                # current_fused_emb 是第 j 层融合后的 Padded 特征 A'

                # --- 4e: 使用 *当层融合后* 的特征计算 *下一层* 的分布 (cul 函数) ---
                layer_scores = self.final_score_func(self.linear_drop(self.current_fused_emb)).squeeze(dim=-1)
                local_entity_mask = node_mask.float()
                layer_scores = layer_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER
                # 更新 layer_input_dist，供下一层 j+1 的 GNN 使用
                layer_input_dist = F.softmax(layer_scores, dim=1)

            # --- 内层循环 j 结束 ---
            # 记录这次外层迭代最终的分布 (即最后一层 j 计算出的 layer_input_dist)
            current_dist = layer_input_dist  # 保存最后一层的分布结果
            self.dist_history.append(current_dist)

            # --- 4f: 指令更新 (使用最后 L 层融合后的特征 current_fused_emb) ---
            for j_ins in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j_ins))
                q = reform(self.instruction.instructions[j_ins].squeeze(1),self.current_fused_emb, query_entities,
                           node_mask.float())
                self.instruction.instructions[j_ins] = q.unsqueeze(1)

        # --- 外层迭代 t 结束 ---

        # --- 步骤 5: 最终答案预测 ---
        pred_dist = self.dist_history[-1]  # 使用最后一次迭代 t 的最终分布

        # --- 步骤 6: 计算损失并返回 (不变) ---
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]

        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list