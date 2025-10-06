import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.base_model import BaseModel
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.hypergraph_construction.constructor import HypergraphConstructor
from modules.kg_reasoning.hyper_gnn import HyperGNNLayer
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class HyperReasonerNet(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init HyperReasonerNet model.
        """
        super(HyperReasonerNet, self).__init__(args, num_entity, num_relation, num_word)
        self.linear_dropout = args['linear_dropout']
        self.norm_rel = args['norm_rel']
        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        # 1. 初始化所有层
        self.layers(args)

        # 2. 初始化所有模块
        self.private_module_def(args, num_entity, num_relation)

        # 3. 将模型移动到设备
        self.to(self.device)

    def layers(self, args):
        """定义模型需要的各种层"""
        entity_dim = self.entity_dim
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        self.score_func = nn.Linear(in_features=self.entity_dim, out_features=1)

    def private_module_def(self, args, num_entity, num_relation):
        """定义模型使用的各个模块"""
        entity_dim = self.entity_dim
        self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
        self.hypergraph_constructor = HypergraphConstructor(args)

        self.num_hyper_gnn_layers = args.get('num_hyper_gnn_layers', 2)
        self.hyper_gnn_layers = nn.ModuleList()
        for _ in range(self.num_hyper_gnn_layers):
            self.hyper_gnn_layers.append(
                HyperGNNLayer(in_features=self.entity_dim,
                              out_features=self.entity_dim,
                              dropout=self.linear_dropout)
            )

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        """获取实体的初始嵌入表示 (逻辑与ReaRev一致)"""
        if self.encode_type:
            return self.type_layer(local_entity=local_entity, edge_list=kb_adj_mat, rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            return self.entity_linear(local_entity_emb)

    def get_rel_feature(self):
        """获取关系的特征表示 (逻辑与ReaRev一致)"""
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features = self.relation_linear(rel_features)
            rel_features_inv = self.relation_linear(rel_features_inv)
        else:
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)

            mask = (self.rel_texts != self.instruction.pad_val).float()
            rel_features = self.self_att_r(rel_features, mask)
            rel_features_inv = self.self_att_r(rel_features_inv, mask)

        return rel_features, rel_features_inv

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """初始化GNN推理环境"""
        # (此处的self.instruction()调用的是bert_encoder.py中的forward方法)
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.local_entity = local_entity
        rel_features, rel_features_inv = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.dist_history = [curr_dist]
        self.seed_entities = curr_dist

    def forward(self, batch, training=False):
        # --- 步骤 1: 解包并转换输入数据 ---
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, _, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        # --- 步骤 2: 初始化推理环境 (获取嵌入等) ---
        self.init_reason(curr_dist=current_dist,
                         local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat,
                         q_input=q_input,
                         query_entities=query_entities)

        node_features = self.local_entity_emb

        # --- 步骤 3: 构建超图 ---
        batch_hyperedges = self.hypergraph_constructor(kb_adj_mat, local_entity)

        # --- 步骤 4: 通过 HyperGNN 更新节点特征 ---
        for layer in self.hyper_gnn_layers:
            node_features = layer(node_features, batch_hyperedges)

        # --- 步骤 5: 答案预测 ---
        final_scores = self.score_func(self.linear_drop(node_features)).squeeze(dim=-1)

        # 应用mask并计算最终概率分布
        local_entity_mask = (local_entity != self.num_entity).float()
        final_scores = final_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = F.softmax(final_scores, dim=1)

        # --- 步骤 6: 计算损失并返回 ---
        if training:
            # 使用 BaseModel 中的 get_loss 方法计算损失
            loss = self.get_loss(pred_dist, answer_dist)

            # 计算评估指标 h1 和 f1
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]

            # 在训练时，返回loss, 预测标签, 概率分布, 以及评估指标
            # (为了兼容trainer, pred可以暂时设为None)
            return loss, None, pred_dist, tp_list
        else:
            # 在推理模式下，只返回概率分布
            return None, None, pred_dist, None