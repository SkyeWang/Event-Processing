"""SAGNN 预处理：Ego/Cut 子图与位置编码（去 DGL 版本）。

原实现用 DGL 做三件事：保存图、提取节点诱导子图、给 PE 提供邻接矩阵。
本文件用 ``core.simple_graph.SimpleGraph`` 替换这些操作，子图与社区划分
逻辑保持与原代码一致。

``SubgraphsTransform`` 的输出字段：
- ``subgraphs_batch`` / ``subgraphs_nodes_mapper`` / ``subgraphs_edges_mapper``：
  Ego 子图在 batch 中的索引，供 PyG 拼接与 scatter 使用；
- ``ego_RWPE``：每个 Ego 子图节点的位置编码；
- ``cut_RWPE`` / ``subgraph_x_index``：Cut 子图的位置编码及对应的原图节点编号；
- ``glo_RWPE``：全图位置编码；
- ``hop_indicator``：每个节点到根节点的跳数。
"""

import itertools
import re

import networkx as nx
import torch
from torch_geometric.data import Data

from core.PE_init import init_positional_encoding, lap_positional_encoding
from core.simple_graph import SimpleGraph
from core.transform_utils.sampling import *  # noqa: F401,F403
from core.transform_utils.subgraph_extractors import *  # noqa: F401,F403


class SubgraphsData(Data):
    """带子图索引的 PyG Data；重写 ``__inc__``/``__cat_dim__`` 以正确 batch。"""

    def __inc__(self, key, value, *args, **kwargs):
        # PyG 在 batch 时会对“索引类”字段自动加偏移：
        # - subgraphs_batch：每个节点属于第几个子图，偏移为 1 + 最大子图编号；
        # - nodes_mapper / edges_mapper：指向原图节点/边，偏移为原图规模；
        # - combined_subgraphs：按子图节点数偏移。
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        self.g = None
        if bool(re.search("(combined_subgraphs)", key)):
            return getattr(self, key[: -len("combined_subgraphs")] + "subgraphs_nodes_mapper").size(0)
        elif bool(re.search("(subgraphs_batch)", key)):
            return 1 + getattr(self, key)[-1]
        elif bool(re.search("(nodes_mapper)|(selected_supernodes)", key)):
            return num_nodes
        elif bool(re.search("(edges_mapper)", key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # combined_subgraphs 的边索引按最后一维拼接，其余交给 PyG 默认逻辑
        if bool(re.search("(combined_subgraphs)", key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class SubgraphsTransform(object):
    """把 PyG ``Data`` 转成 SAGNN 需要的 ``SubgraphsData``。

    参数从 yacs config 读取：``subgraph.hops`` 控制 Ego 子图跳数，
    ``model.cut_times`` 控制 Cut 子图数量，``model.embedding_type`` 选择
    ``rand_walk`` 或 ``lap_pe``。
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_hops = cfg.subgraph.hops
        self.cut_times = cfg.model.cut_times
        self.fullgraph_pos_enc_dim = cfg.model.fullgraph_pos_enc_dim
        self.egograph_pos_enc_dim = cfg.model.egograph_pos_enc_dim
        self.cutgraph_pos_enc_dim = cfg.model.cutgraph_pos_enc_dim
        self.embedding_type = cfg.model.embedding_type

    def _encode(self, graph, dim):
        """按配置选择 Laplacian PE 或随机游走 PE。"""
        if self.embedding_type == "lap_pe":
            return lap_positional_encoding(graph, dim)
        return init_positional_encoding(graph, dim)

    def __call__(self, data):
        # 复制成 SubgraphsData，保留 x/edge_index/y 等原有字段
        data = SubgraphsData(**{k: v for k, v in data})
        graph = SimpleGraph(data.num_nodes, data.edge_index)

        # ---- Ego 子图：每个节点一个 k 跳子图 -------------------------
        # node_mask[i, j] 表示节点 j 是否在节点 i 的 k 跳子图中；
        # edge_mask[i, e] 表示边 e 是否被第 i 个子图包含。
        node_mask, edge_mask, hop_dense = extract_subgraphs(
            data.edge_index, data.num_nodes, self.num_hops
        )
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
            node_mask, edge_mask, hop_dense
        )

        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]

        # 对每个 Ego 子图单独计算 PE，再按子图顺序拼接；
        # 行顺序与 subgraphs_nodes_mapper 对齐。
        ego_rwpe = []
        for i in range(edge_mask.shape[0]):
            target_nodes = graph.nodes()[node_mask[i]]
            sub_g = graph.subgraph(target_nodes)
            ego_rwpe.append(self._encode(sub_g, self.egograph_pos_enc_dim))
        ego_rwpe = torch.cat(ego_rwpe, dim=0)

        global_rwpe = self._encode(graph, self.fullgraph_pos_enc_dim)

        # ---- Cut 子图：Girvan-Newman 社区划分 -----------------------
        graph_nx = graph.to_networkx()
        target_g_list = []
        comp = nx.algorithms.community.girvan_newman(graph_nx)
        connected_components = list(nx.algorithms.connected_components(graph_nx))
        if len(connected_components) >= self.cut_times:
            # 原图本身就不连通：每个连通分量直接作为一个 Cut 子图
            for item in connected_components:
                target_g_list.append(graph_nx.subgraph(item))
        else:
            # 否则用 GN 迭代切分，取社区数不超过 cut_times 的最后一次划分
            ggg = None
            limited = itertools.takewhile(lambda c: len(c) <= self.cut_times, comp)
            for communities in limited:
                ggg = tuple(sorted(c) for c in communities)
            if ggg is not None:
                for item in ggg:
                    target_g_list.append(graph_nx.subgraph(item))
        if not target_g_list:  # 无边图或没有社区可切时退化为全图
            target_g_list.append(graph_nx)

        cut_rwpe = []
        subgraph_x_index = []
        for g in target_g_list:
            # 记录 Cut 子图节点在原图中的编号，供模型 scatter 回原图
            subgraph_x_index.append(
                torch.tensor(list(g.nodes()), dtype=torch.long)
            )
            cut_rwpe.append(self._encode(SimpleGraph.from_networkx(g), self.cutgraph_pos_enc_dim))

        data.hop_indicator = hop_indicator
        data.cut_RWPE = torch.cat(cut_rwpe, dim=0)
        data.ego_RWPE = ego_rwpe
        data.glo_RWPE = global_rwpe
        data.subgraph_x_index = torch.cat(subgraph_x_index, dim=-1)
        data.__num_nodes__ = data.num_nodes
        return data


def to_sparse(node_mask, edge_mask, hop_indicator):
    """把稠密掩码转成 PyG 需要的稀疏索引。

    Args:
        node_mask: ``[N, N]`` bool，``node_mask[i, j]`` 表示 j 是否在 i 的子图中。
        edge_mask: ``[N, E]`` bool，表示边是否被某个子图包含。
        hop_indicator: ``[N, N]`` 跳数，可为 None。

    Returns:
        ``(subgraphs_nodes, subgraphs_edges, hop_indicator)``：
        - subgraphs_nodes: ``[2, nnz]``，第一行是子图编号，第二行是节点编号；
        - subgraphs_edges: ``[2, nnz]``，第一行是子图编号，第二行是边编号；
        - hop_indicator: 按子图节点顺序取出的跳数，或 None。
    """
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges, hop_indicator


def extract_subgraphs(edge_index, num_nodes, num_hops, sparse=False):
    """为每个节点提取 k 跳 Ego 子图。

    先由 ``k_hop_subgraph`` 得到节点掩码和跳数，再用“两端都在子图内”
    构造边掩码。``sparse=True`` 时直接返回 ``to_sparse`` 的索引格式。
    """
    node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes, num_hops)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]]
    if not sparse:
        return node_mask, edge_mask, hop_indicator
    return to_sparse(node_mask, edge_mask, hop_indicator)
