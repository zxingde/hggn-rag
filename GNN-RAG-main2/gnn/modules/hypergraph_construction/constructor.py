import torch
import torch.nn as nn
from collections import defaultdict
import random


class HypergraphConstructor(nn.Module):
    def __init__(self, args):
        """
        初始化超图构建器。
        这里我们从 args 中获取随机游走所需要的超参数。
        """
        super(HypergraphConstructor, self).__init__()

        # 从配置参数中读取随机游走所需的设置
        self.walk_length = args.get('hypergraph_walk_length', 3)  # 每次游走的长度
        self.num_walks = args.get('hypergraph_num_walks', 5)  # 每个节点作为起点的游走次数

        print(f"超图构建器初始化成功: 游走长度={self.walk_length}, 每个节点游走次数={self.num_walks}")

    def forward(self, kb_adj_mat, local_entity):
        """
        输入:
            kb_adj_mat (tuple): 包含图连接信息的元组 (heads, rels, tails, ...)
            local_entity (Tensor): 子图中的实体列表, 形状为 (batch_size, max_local_entity)
        输出:
            batch_hyperedges (list): 一个列表，每个元素是该 batch 中一个图的所有超边
        """
        print("--- 进入超图构建模块 ---")

        # 解包图的边信息
        batch_heads, _, batch_tails, batch_ids, _, _, _ = kb_adj_mat

        # --- 1. 构建邻接表 ---
        # 我们需要为 batch 中的每一个图单独构建邻接表
        adj_lists = [defaultdict(list) for _ in range(local_entity.size(0))]

        for head, tail, batch_id in zip(batch_heads, batch_tails, batch_ids):
            # 对于图中的每条边 (head -> tail)，我们都认为是无向的
            # 这样随机游走时可以双向进行
            adj_lists[batch_id][head].append(tail)
            adj_lists[batch_id][tail].append(head)

        batch_hyperedges = []
        # --- 2. 对 batch 中的每个图执行随机游走 ---
        for batch_id in range(local_entity.size(0)):
            adj = adj_lists[batch_id]
            nodes = list(adj.keys())  # 获取当前图的所有节点

            hyperedges_for_one_graph = set()  # 使用集合来自动去重

            if not nodes:
                batch_hyperedges.append([])
                continue

            # 从每个节点开始，执行 num_walks 次随机游走
            for _ in range(self.num_walks):
                for start_node in nodes:
                    walk = [start_node]  # 路径以起始节点开始
                    current_node = start_node

                    # 执行 walk_length 步的游走
                    for _ in range(self.walk_length - 1):
                        neighbors = adj.get(current_node)
                        if not neighbors:  # 如果当前节点没有邻居，则停止本次游走
                            break

                        # 从邻居中随机选择一个作为下一个节点
                        next_node = random.choice(neighbors)
                        walk.append(next_node)
                        current_node = next_node

                    # 将生成的路径（超边）排序后转换成元组，存入集合
                    hyperedges_for_one_graph.add(tuple(sorted(walk)))

            batch_hyperedges.append(list(hyperedges_for_one_graph))

        print(f"--- 超图构建完成，为 batch 0 构建了 {len(batch_hyperedges[0]) if batch_hyperedges else 0} 条超边 ---")

        return batch_hyperedges