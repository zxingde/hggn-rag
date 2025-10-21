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
from modules.kg_reasoning.hyper_gnn import HyperGNNLayer

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class HyperGNNReasoningLayer(nn.Module):
    """
    封装HGNN推理逻辑的模块，包括动态权重计算和HGNN层。
    """

    def __init__(self, args, entity_dim, num_layers, dropout):
        super(HyperGNNReasoningLayer, self).__init__()
        self.entity_dim = entity_dim

        # 1. 定义HGNN层
        self.hgnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hgnn_layers.append(
                HyperGNNLayer(in_features=entity_dim,
                              out_features=entity_dim,
                              dropout=dropout)
            )

        # 2. 定义动态权重MLP
        self.hyperedge_weight_mlp = nn.Linear(self.entity_dim * 2, 1)

    def forward(self, init_entity_emb, current_instructions_avg, batch_hyperedges, max_num_hyperedges, batch_size,
                device):
        """
        init_entity_emb: (B, N, D) - 初始节点嵌入
        current_instructions_avg: (B, D) - 当前迭代的平均指令
        batch_hyperedges: list[list[tuple]] - 预先构建的超边
        """

        # --- 步骤 5: 动态超边权重计算 ---
        if max_num_hyperedges > 0:
            padded_weights = torch.zeros(batch_size, max_num_hyperedges).to(device)
            for i in range(batch_size):
                hyperedges = batch_hyperedges[i]
                if not hyperedges:
                    continue

                num_hyperedges = len(hyperedges)
                # a. 计算超边表示 (使用 初始 节点特征)
                current_node_embs = init_entity_emb[i]
                hyperedge_embs = []
                for h_edge in hyperedges:
                    valid_indices = [idx for idx in list(h_edge) if idx < current_node_embs.shape[0]]
                    if not valid_indices:
                        hyperedge_embs.append(torch.zeros(self.entity_dim).to(device))
                        continue
                    nodes_in_edge_embs = current_node_embs[valid_indices, :]
                    hyperedge_embs.append(torch.mean(nodes_in_edge_embs, dim=0))

                if not hyperedge_embs:  # 确保 hyperedge_embs 不为空
                    continue

                hyperedge_embs = torch.stack(hyperedge_embs)

                # b. 计算匹配分数
                current_instruction = current_instructions_avg[i].unsqueeze(0).expand(num_hyperedges, -1)
                combined_reps = torch.cat([hyperedge_embs, current_instruction], dim=1)
                scores = self.hyperedge_weight_mlp(combined_reps).squeeze(-1)

                # c. 生成权重
                weights = torch.sigmoid(scores)
                padded_weights[i, :num_hyperedges] = weights
            final_hyperedge_weights = padded_weights
        else:
            final_hyperedge_weights = None

        # --- 步骤 6: 加权HGNN推理 ---
        # 每次都从 初始 实体嵌入开始HNN推理
        hgnn_entity_emb = init_entity_emb
        for layer in self.hgnn_layers:
            hgnn_entity_emb = layer(hgnn_entity_emb, batch_hyperedges, final_hyperedge_weights)

        return hgnn_entity_emb

class ReaRevHGNN(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word, relation2id):
        """
        初始化 ReaRevHGNN 模型.
        """
        super(ReaRevHGNN, self).__init__(args, num_entity, num_relation, num_word)
        self.norm_rel = args['norm_rel']
        self.relation2id = relation2id
        self.layers(args)

        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

        relation_features = self.relation_embedding.weight.clone()
        self.private_module_def(args, num_entity, num_relation, relation_features)

        self.to(self.device)
        self.lin = nn.Linear(3 * self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))

    def layers(self, args):
        # (这部分与之前保持一致)
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.linear_dropout = args['linear_dropout']
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

        # =================================================================
        # ================ 2. 替换这个函数 ================================
        # =================================================================
    def private_module_def(self, args, num_entity, num_relation, relation_features):
        """
        定义模型使用的各个模块.
        """
        entity_dim = self.entity_dim
        # 原始GNN推理模块
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)

        # 指令编码器
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])

        # --- 模块修改与新增 ---
        # 1. 超图构建器 (使用新的基于关系的构造器)
        self.hypergraph_constructor = RelationCommunityConstructor(args, num_entity)

        # 2. HGNN推理模块 (已重构)
        self.num_hyper_gnn_layers = args.get('num_hyper_gnn_layers', self.num_gnn)
        self.hgnn_reasoning = HyperGNNReasoningLayer(args,
                                                     self.entity_dim,
                                                     self.num_hyper_gnn_layers,
                                                     self.linear_dropout)

        # 3. GNN-HNN 融合模块 (模块四)
        self.diffusion_fusion = DiffusionFusion(self.entity_dim)

        # 4. 最终打分层
        self.final_score_func = nn.Linear(in_features=self.entity_dim, out_features=1)

    # --- 省略 get_ent_init, get_rel_feature, calc_loss_label 等与之前版本一致的辅助函数 ---
    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        self.reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb, rel_features=rel_features,
                                   rel_features_inv=rel_features_inv, query_entities=query_entities)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity, edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)
        return local_entity_emb

    def get_rel_feature(self):
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            rel_features = self.self_att_r(rel_features, (self.rel_texts != self.instruction.pad_val).float())
            rel_features_inv = self.self_att_r(rel_features_inv, (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation + 1).float())
                rel_features_inv = self.self_att_r(rel_features_inv,
                                                   (self.rel_texts_inv != self.num_relation + 1).float())
        return rel_features, rel_features_inv

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss


    def forward(self, batch, training=False):
        """
        修改后的 Forward 流程: (GNN -> HGNN -> Fusion -> 指令更新) x T次
        """
        # --- 步骤 1: 解包并转换输入数据 ---
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)
        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        batch_size = local_entity.size(0)

        # --- 步骤 2: 初始化推理环境 (获取初始指令和嵌入) ---
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins

        # --- 步骤 3: 超图构建 (在循环外构建一次) ---
        batch_hyperedges = self.hypergraph_constructor(kb_adj_mat, local_entity, self.relation2id)
        max_num_hyperedges = max(len(h) for h in batch_hyperedges) if batch_hyperedges else 0

        # --- 步骤 4: 迭代推理 (GNN -> HGNN -> Fusion -> Update) ---
        gnn_entity_emb = None
        fused_entity_emb = None  # 存储最后一次迭代的融合嵌入
        gnn_current_dist = self.curr_dist  # 保持原始的种子分布 (用于GNN)

        for t in range(self.num_iter):

            # --- 4a: GNN推理 ---
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = gnn_current_dist  # GNN 每次都从种子分布开始 (ReaRev特性)
            for j in range(self.num_gnn):
                self.curr_dist, gnn_entity_emb = self.reasoning(self.curr_dist, relation_ins, step=j)

            # --- 4b: HGNN推理 ---
            current_instructions_avg = torch.mean(relation_ins, dim=1)  # (B, D)
            hgnn_entity_emb = self.hgnn_reasoning(self.init_entity_emb,  # HGNN 每次都从初始嵌入开始
                                                  current_instructions_avg,
                                                  batch_hyperedges,
                                                  max_num_hyperedges,
                                                  batch_size,
                                                  self.device)

            # --- 4c: 融合 GNN 和 HGNN ---
            fused_entity_emb = self.diffusion_fusion(gnn_entity_emb, hgnn_entity_emb)

            # --- 4d: 指令更新 (使用融合后的特征) ---
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), fused_entity_emb, query_entities,
                           (local_entity != self.num_entity).float())
                self.instruction.instructions[j] = q.unsqueeze(1)

        # --- 迭代循环结束 ---

        # --- 步骤 5: 最终答案预测 ---
        # 使用 *最后一次* 迭代产生的融合特征进行预测
        final_scores = self.final_score_func(self.linear_drop(fused_entity_emb)).squeeze(dim=-1)

        local_entity_mask = (local_entity != self.num_entity).float()
        final_scores = final_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = F.softmax(final_scores, dim=1)

        # --- 步骤 6: 计算损失并返回 ---
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