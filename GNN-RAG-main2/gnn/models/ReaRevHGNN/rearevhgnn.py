import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.reasongnn import ReasonGNNLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
# 导入新的模块
from modules.query_update import AttnEncoder, Fusion, QueryReform, DiffusionFusion
from modules.hypergraph_construction.adaptive_constructor import AdaptiveHypergraphConstructor
from modules.kg_reasoning.hyper_gnn import HyperGNNLayer

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class ReaRevHGNN(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        初始化 ReaRevHGNN 模型.
        """
        super(ReaRevHGNN, self).__init__(args, num_entity, num_relation, num_word)
        self.norm_rel = args['norm_rel']
        self.layers(args)

        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

        # 在这里初始化关系嵌入，以便传递给超图构造器
        # 注意：这部分逻辑在BaseModel的embedding_def中，super()调用时已执行
        relation_features = self.relation_embedding.weight.clone()

        self.private_module_def(args, num_entity, num_relation, relation_features)

        self.to(self.device)
        self.lin = nn.Linear(3 * self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))

    def layers(self, args):
        # 这部分与原始ReaRev保持一致
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

        # --- 新增模块 ---
        # 1. 超图构建器
        self.hypergraph_constructor = AdaptiveHypergraphConstructor(args, relation_features)

        # 2. HGNN层
        self.num_hyper_gnn_layers = args.get('num_hyper_gnn_layers', self.num_gnn)  # 层数与GNN保持一致
        self.hgnn_layers = nn.ModuleList()
        for _ in range(self.num_hyper_gnn_layers):
            self.hgnn_layers.append(
                HyperGNNLayer(in_features=self.entity_dim,
                              out_features=self.entity_dim,
                              dropout=self.linear_dropout)
            )

        # 3. 融合模块
        self.diffusion_fusion = DiffusionFusion(self.entity_dim)

        # 4. 最终打分层 (现在输入是融合后的特征)
        self.final_score_func = nn.Linear(in_features=self.entity_dim, out_features=1)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        初始化推理环境 (与原始ReaRev保持一致)
        """
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.init_entity_emb = self.local_entity_emb
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist

        self.reasoning.init_reason(
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            local_entity_emb=self.local_entity_emb,
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        # (与原始ReaRev保持一致)
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)
        return local_entity_emb

    def get_rel_feature(self):
        # (与原始ReaRev保持一致)
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
        # (与原始ReaRev保持一致)
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, training=False):
        """
        新的 Forward 流程: GNN -> HGNN -> Fusion -> Predict
        """
        # --- 步骤 1: 解包并转换输入数据 (与原始一致) ---
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist, g2l_maps_batch = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)
        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)

        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val
            query_mask = (q_input != pad_val).float()
        else:
            query_mask = (q_input != self.num_word).float()

        # --- 步骤 2: 初始化推理环境 (与原始一致) ---
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins

        # --- 步骤 3: 原始GNN推理 ---
        gnn_entity_emb = None
        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist
            for j in range(self.num_gnn):
                self.curr_dist, gnn_entity_emb = self.reasoning(self.curr_dist, relation_ins, step=j)

            # (指令更新部分保持不变)
            qs = []
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), gnn_entity_emb, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)

        # --- 步骤 4: 超图构建和HGNN推理 ---
        # 4.1 构建超图
        # 注意：这里需要传入全局到局部的映射，这在dataloader中生成，但模型本身不直接接收
        # 我们需要从dataloader获取或在模型中模拟这个映射
        # 暂时我们假设可以拿到 g2l_maps_batch
        # TODO: 需要从dataloader获取 g2l_maps_batch
        # 这里暂时用一个placeholder代替，实际需要从dataloader传入
        g2l_maps_batch = [{} for _ in range(local_entity.size(0))]
        batch_hyperedges = self.hypergraph_constructor(kb_adj_mat, self.init_entity_emb, g2l_maps_batch)

        # 4.2 HGNN推理
        hgnn_entity_emb = self.init_entity_emb  # 使用初始的实体嵌入
        for layer in self.hgnn_layers:
            hgnn_entity_emb = layer(hgnn_entity_emb, batch_hyperedges)

        # --- 步骤 5: 融合GNN和HGNN的输出 ---
        fused_entity_emb = self.diffusion_fusion(gnn_entity_emb, hgnn_entity_emb)

        # --- 步骤 6: 最终答案预测 ---
        final_scores = self.final_score_func(self.linear_drop(fused_entity_emb)).squeeze(dim=-1)

        # 应用mask并计算最终概率分布
        local_entity_mask = (local_entity != self.num_entity).float()
        final_scores = final_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = F.softmax(final_scores, dim=1)

        # --- 步骤 7: 计算损失并返回 (与原始一致) ---
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