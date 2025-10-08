import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np


class AdaptiveHypergraphConstructor(nn.Module):
    def __init__(self, args, relation_features):
        """
        初始化自适应超图构建器。
        :param args: 模型参数
        :param relation_features: (num_relations, feature_dim) 关系嵌入
        """
        super(AdaptiveHypergraphConstructor, self).__init__()
        self.num_relation_clusters = args.get('num_relation_clusters', 5)  # 关系聚类的簇数
        self.num_node_clusters = args.get('num_node_clusters', 10)  # 节点聚类的簇数
        self.relation_features = relation_features.detach().cpu().numpy()

        # 1. 对关系进行聚类
        self.relation_cluster_labels = self._cluster_relations()

    def _cluster_relations(self):
        """
        使用K-Means对关系嵌入进行聚类
        """
        kmeans = KMeans(n_clusters=self.num_relation_clusters, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(self.relation_features)
        print(f"关系聚类完成，共 {self.num_relation_clusters} 个簇。")
        return torch.tensor(cluster_labels)

    def forward(self, kb_adj_mat, local_entity_emb, local_entity_map):
        """
        输入:
            kb_adj_mat (tuple): 包含图连接信息的元组 (heads, rels, tails, ...)
            local_entity_emb (Tensor): 子图中的实体嵌入, 形状为 (batch_size, num_nodes, feature_dim)
            local_entity_map (list[dict]): 从全局ID到局部ID的映射列表
        输出:
            batch_hyperedges (list): 一个列表，每个元素是该 batch 中一个图的所有超边
        """
        batch_heads, batch_rels, batch_tails, batch_ids, _, _, _ = kb_adj_mat
        batch_size = local_entity_emb.size(0)
        batch_hyperedges = []

        for i in range(batch_size):
            # 获取当前图的节点嵌入和ID映射
            current_entity_emb = local_entity_emb[i].detach().cpu().numpy()
            g2l = local_entity_map[i]

            if current_entity_emb.shape[0] == 0:
                batch_hyperedges.append([])
                continue

            # 按关系簇将节点分组
            nodes_in_relation_cluster = defaultdict(set)

            # 找到属于当前batch的边
            current_batch_indices = np.where(batch_ids == i)[0]

            for index in current_batch_indices:
                head_local_id = batch_heads[index] % len(g2l)
                tail_local_id = batch_tails[index] % len(g2l)
                rel_id = batch_rels[index]

                # 获取关系所在的簇
                cluster_id = self.relation_cluster_labels[rel_id].item()
                nodes_in_relation_cluster[cluster_id].add(head_local_id)
                nodes_in_relation_cluster[cluster_id].add(tail_local_id)

            hyperedges_for_one_graph = set()

            # 在每个关系簇内进行节点聚类
            for cluster_id, nodes in nodes_in_relation_cluster.items():
                node_list = sorted(list(nodes))
                if len(node_list) < self.num_node_clusters:
                    # 如果节点数少于聚类数，直接作为一条超边
                    hyperedges_for_one_graph.add(tuple(node_list))
                    continue

                # 提取这些节点的嵌入
                node_embeddings_for_clustering = current_entity_emb[node_list]

                # 使用K-Means进行节点聚类
                kmeans = KMeans(n_clusters=self.num_node_clusters, random_state=0, n_init=10)
                node_cluster_labels = kmeans.fit_predict(node_embeddings_for_clustering)

                # 将每个节点簇作为一条超边
                nodes_in_node_cluster = defaultdict(list)
                for local_node_idx, node_cluster_id in enumerate(node_cluster_labels):
                    original_node_id = node_list[local_node_idx]
                    nodes_in_node_cluster[node_cluster_id].append(original_node_id)

                for _, hyperedge_nodes in nodes_in_node_cluster.items():
                    hyperedges_for_one_graph.add(tuple(sorted(hyperedge_nodes)))

            batch_hyperedges.append(list(hyperedges_for_one_graph))

        return batch_hyperedges