# modules/hypergraph_construction/relation_community_constructor.py

import torch
import torch.nn as nn
from collections import defaultdict
import networkx as nx
import community as community_louvain
import numpy as np


class RelationCommunityConstructor(nn.Module):
    def __init__(self, args, num_entity):  # <-- 修改点 1：在这里直接接收 num_entity
        """
        初始化基于关系的社区超图构建器。
        :param args: 模型参数
        :param num_entity: KG中的总实体数, 用于识别 padding
        """
        super(RelationCommunityConstructor, self).__init__()
        self.args = args
        self.num_entity = num_entity  # <-- 修改点 2：直接保存 num_entity，不再依赖 self.args
        print("Relation-based Community Hypergraph Constructor initialized.")

    def forward(self, kb_adj_mat, local_entity, relation2id):
        """
        为批次中的每个图构建超边。
        """
        batch_heads, batch_rels, batch_tails, batch_ids, _, _, _ = kb_adj_mat
        batch_size = local_entity.size(0)
        num_relations = len(relation2id)

        batch_hyperedges = []

        for i in range(batch_size):
            hyperedges_for_one_graph = set()
            current_batch_indices = np.where(batch_ids == i)[0]

            if len(current_batch_indices) == 0:
                batch_hyperedges.append([])
                continue

            # <-- 修改点 3：使用 self.num_entity 替换原来的 self.args['num_entity']
            max_local_entity_idx = local_entity[i].ne(self.num_entity).sum().item()

            if max_local_entity_idx == 0:
                batch_hyperedges.append([])
                continue

            graphs_by_relation = [nx.Graph() for _ in range(num_relations)]
            for index in current_batch_indices:
                head_local = batch_heads[index] % max_local_entity_idx
                tail_local = batch_tails[index] % max_local_entity_idx
                rel_id = batch_rels[index]

                if rel_id < num_relations:
                    graphs_by_relation[rel_id].add_edge(head_local, tail_local)

            for rel_graph in graphs_by_relation:
                if rel_graph.number_of_nodes() == 0:
                    continue

                partition = community_louvain.best_partition(rel_graph)

                communities = defaultdict(list)
                for node, community_id in partition.items():
                    communities[community_id].append(node)

                for community_id, nodes in communities.items():
                    if len(nodes) > 1:
                        hyperedges_for_one_graph.add(tuple(sorted(nodes)))

            batch_hyperedges.append(list(hyperedges_for_one_graph))

        return batch_hyperedges