import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.hgnn_reason import hgnn_reason
from modules.kg_reasoning.hgnn_layer import hgnn_layer
from modules.question_encoding.lstm_encoder import LSTMInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.query_update import AttnEncoder, Fusion, QueryReform, DiffusionFusion
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder, Fusion, QueryReform

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class ReaRevHGNN(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRevHGNN, self).__init__(args, num_entity, num_relation, num_word)
        # self.embedding_def()
        # self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)

        self.loss_type = args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']

        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3 * self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        # self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']

        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        # self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        # self.self_att_r_inv = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)

        return local_entity_emb

    def get_rel_feature(self):
        """
        Encode relation tokens to vectors.
        """
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

    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        num_hyper_gnn_layers = args.get('num_hyper_gnn_layers', 3)
        hgnn_module = hgnn_layer(
            in_dim=entity_dim,
            out_dim=entity_dim,
            num_layers=num_hyper_gnn_layers,
            dropout=self.linear_dropout,  # 复用 GNN 的 dropout
            instruction_dim=entity_dim
        )

        fusion_module = DiffusionFusion(d_hid=entity_dim)
        self.reasoning = hgnn_reason(
            args, num_entity, num_relation, entity_dim, self.alg,
            hgnn_module=hgnn_module,
            fusion_module=fusion_module
        )
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            # self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities,pyg_hypergraph_batch=None, node_mask=None):
        """
        Initializing Reasoning
        """
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        # 2*(8*50)  2*(8*16*1)
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
            local_entity_emb=self.local_entity_emb,  # 传递 GNN 自己的初始嵌入 (Padded A)
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities,
            # --- 新增传递的参数 ---
            pyg_hypergraph_batch=pyg_hypergraph_batch,  # 传递 PyG 对象
            node_mask=node_mask,  # 传递布尔 Mask
            init_entity_emb=self.init_entity_emb  # 传递最原始的 Padded 嵌入 A
            # --- 结束新增 ---
        )

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """
        local_entity, query_entities, kb_adj_mat, pyg_hypergraph_batch, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        initial_dist = Variable(seed_dist, requires_grad=False)  # 初始种子分布
        q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        pyg_hypergraph_batch = pyg_hypergraph_batch.to(self.device)
        batch_size = local_entity.size(0)

        current_dist = Variable(seed_dist, requires_grad=True)

        # q_input = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        # query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val  # tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()

        else:
            query_mask = (q_input != self.num_word).float()

        """
        Instruction generations
        """
        node_mask = (local_entity != self.num_entity)
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities,
                         pyg_hypergraph_batch = pyg_hypergraph_batch,
                         node_mask = node_mask
                         )
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        # relation_ins = torch.cat(self.instruction.instructions, dim=1)
        # query_emb = None
        self.dist_history.append(self.curr_dist)

        """
        BFS + GNN reasoning
        """

        for t in range(self.num_iter):
            total_cl_loss = 0
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist
            for j in range(self.num_gnn):
                self.curr_dist, global_rep, gnn_emb, hgnn_emb = self.reasoning(self.curr_dist, relation_ins, step=j)
                if training:
                    # 调用计算对比损失的函数
                    # 注意：确保你在类里已经添加了 compute_contrastive_loss 函数
                    step_cl_loss = self.compute_contrastive_loss(gnn_emb, hgnn_emb, node_mask)
                    total_cl_loss += step_cl_loss
                # ================================================================
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)

        """
        Answer Predictions
        """
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = 0
        # for pred_dist in self.dist_history:
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        # ==================== 【修改 4：将对比损失加入总 Loss】 ====================
        if training:
            # 确保你在 __init__ 中定义了 self.cl_weight (例如 0.1 或 0.5)
            # 如果没定义，这里暂时用常数 0.1 代替也可以
            weight = getattr(self, 'cl_weight', 0.1)
            loss = loss + weight * total_cl_loss
        # ======================================================================
        pred_dist = self.dist_history[-1]

        # ==================== 修改开始：提取 Top-K + Global ====================

        final_graph_sequence = None  # 初始化返回值

        if not training:  # 只有在推理/评估阶段我们需要提取这个向量
            # 1. 设定 K 值 (比如保存最重要的 10 个节点)
            K = 5
            # 确保 K 不超过当前图的实体数量
            curr_K = min(K, pred_dist.size(1))

            # 2. 选出 Top-K 索引
            # topk_indices: [batch_size, K]
            topk_scores, topk_indices = torch.topk(pred_dist, k=curr_K, dim=1)

            # 3. 提取 Top-K 实体的语义向量
            # self.local_entity_emb: [batch_size, max_local_entity, entity_dim]
            # 我们需要 gather 出来
            dim = self.local_entity_emb.size(-1)

            # 扩展索引维度以便 gather: [batch, K] -> [batch, K, dim]
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, dim)

            # 提取向量: [batch, K, dim]
            topk_node_vecs = torch.gather(self.local_entity_emb, 1, expanded_indices)

            # 4. 处理全局向量 (Global Rep)
            # global_rep 是循环里最后一次 reasoning 产出的
            # global_rep: [batch, dim] -> 变成 [batch, 1, dim]
            if 'global_rep' in locals() and global_rep is not None:
                global_vec = global_rep.unsqueeze(1)
            else:
                # 极端情况防爆：如果没有 global_rep，用全0填充
                global_vec = torch.zeros(batch_size, 1, dim).to(self.device)

            # 5. 拼接：[Global(1) + TopK(10)] -> [batch, 11, dim]
            final_graph_sequence = torch.cat([global_vec, topk_node_vecs], dim=1)

            # 6. Detach (断开梯度，只存数值)
            final_graph_sequence = final_graph_sequence.detach().cpu()

        # ==================== 修改结束 ====================

        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list,final_graph_sequence

    def compute_contrastive_loss(self, view1, view2, mask=None):
        """
        计算对比损失: 拉近 gnn_emb (view1) 和 hgnn_emb (view2) 的距离
        """
        # 简单的 Cosine 相似度对比
        # 如果你想效果更好，可以在 __init__ 里加投影层 self.proj = nn.Linear(...)

        z1 = F.normalize(view1, dim=-1)
        z2 = F.normalize(view2, dim=-1)

        # 计算正样本相似度 (同节点在两视图应相似)
        # [Batch, Nodes, Dim] * [Batch, Nodes, Dim] -> sum(-1) -> [Batch, Nodes]
        pos_sim = torch.sum(z1 * z2, dim=-1)

        # 损失：最小化负的相似度 (即最大化相似度)
        loss_per_node = -pos_sim

        # 只计算有效节点 (mask 掉 padding)
        if mask is not None:
            loss_per_node = loss_per_node * mask.float()
            return loss_per_node.sum() / (mask.float().sum() + 1e-9)
        else:
            return loss_per_node.mean()

