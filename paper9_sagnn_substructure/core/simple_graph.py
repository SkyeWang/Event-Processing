"""轻量图容器，替代 SAGNN 原实现中的 DGL。

只实现 ``core.transform`` 和 ``core.PE_init`` 用到的操作：

- ``number_of_nodes`` / ``nodes`` / ``in_degrees``：节点数、节点编号、入度；
- ``adjacency_matrix``：返回 SciPy CSR 邻接矩阵；
- ``subgraph``：节点诱导子图，按给定节点顺序重编号；
- ``to_networkx`` / ``from_networkx``：与 networkx 互转。

节点编号始终为 ``0 .. n-1``；子图重编号顺序与 DGL 的 ``node_subgraph``
保持一致，保证替换 DGL 后 Ego/Cut 子图的节点顺序不变。

注意：本类只服务于 SAGNN 的小图（MUTAG/ZINC），没有实现 DGL 的异构图、
消息传递、GPU 稀疏矩阵等完整能力。
"""

import networkx as nx
import numpy as np
import torch
from scipy import sparse as sp


class SimpleGraph:
    """最小图结构：节点数 + 边索引 + ``ndata`` 字典。"""

    def __init__(self, num_nodes, edge_index):
        # _n 统一存 int，避免 numpy/torch 的 int 类型混用
        self._n = int(num_nodes)
        # edge_index 统一成 [2, E] 的 long tensor，后续切片/索引都用它
        self.edge_index = edge_index.long().contiguous().view(2, -1)
        # 兼容 DGL 的 g.ndata['pos_enc'] 写法；SAGNN 只在 PE 里用到
        self.ndata = {}

    # ---- DGL 兼容的访问接口 -------------------------------------------
    def number_of_nodes(self):
        """返回节点数（DGL 风格接口）。"""
        return self._n

    def num_nodes(self):
        """返回节点数（PyG 风格接口）。"""
        return self._n

    def nodes(self):
        """返回 ``0 .. n-1`` 的 long tensor。"""
        return torch.arange(self._n, dtype=torch.long)

    def in_degrees(self):
        """返回入度。

        SAGNN 的图是无向图（边索引通常双向），因此入度等于度数；
        这里按列索引做 bincount，与 DGL ``in_degrees`` 的语义一致。
        """
        if self.edge_index.numel() == 0:
            return torch.zeros(self._n, dtype=torch.float)
        return torch.bincount(self.edge_index[1], minlength=self._n).float()

    def adjacency_matrix_scipy(self, return_edge_ids=False):
        """返回 SciPy CSR 邻接矩阵。

        ``return_edge_ids`` 只是为了兼容旧调用签名，当前实现不返回边 id。
        """
        return self._adjacency()

    def adjacency_matrix(self, scipy_fmt="csr"):
        """返回 SciPy CSR 邻接矩阵；``scipy_fmt`` 仅为兼容旧接口。"""
        return self._adjacency()

    def _adjacency(self):
        """由边索引构造 ``n x n`` 的 CSR 邻接矩阵。

        重复边会在 COO 转 CSR 时按值累加；SAGNN 的图没有重边，
        因此矩阵元素为 0/1。
        """
        if self.edge_index.numel() == 0:
            return sp.csr_matrix((self._n, self._n), dtype=np.float64)
        row = self.edge_index[0].cpu().numpy()
        col = self.edge_index[1].cpu().numpy()
        data = np.ones(row.shape[0], dtype=np.float64)
        return sp.coo_matrix((data, (row, col)), shape=(self._n, self._n)).tocsr()

    # ---- 图操作 -------------------------------------------------------
    def subgraph(self, nodes):
        """取节点诱导子图，并按 ``nodes`` 的顺序重编号。

        步骤：
        1. 用布尔掩码筛出两端都在 ``nodes`` 中的边；
        2. 建立旧节点 id -> 新节点 id 的映射（按 ``nodes`` 顺序）；
        3. 用映射重写边索引，返回新的 ``SimpleGraph``。

        该顺序与 DGL ``node_subgraph`` 一致，保证后续 RWPE 的行顺序不变。
        """
        nodes = torch.as_tensor(nodes, dtype=torch.long)
        if nodes.numel() == 0:
            return SimpleGraph(0, torch.empty(2, 0, dtype=torch.long))
        keep = torch.zeros(self._n, dtype=torch.bool)
        keep[nodes] = True
        edge_keep = keep[self.edge_index[0]] & keep[self.edge_index[1]]
        mapper = torch.full((self._n,), -1, dtype=torch.long)
        mapper[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
        new_edges = mapper[self.edge_index[:, edge_keep]]
        return SimpleGraph(nodes.numel(), new_edges)

    def to_networkx(self):
        """转成无向 ``networkx.Graph``；重复边会自动合并。"""
        g = nx.Graph()
        g.add_nodes_from(range(self._n))
        if self.edge_index.numel() > 0:
            edges = self.edge_index.t().cpu().numpy().tolist()
            g.add_edges_from((int(u), int(v)) for u, v in edges)
        return g

    @classmethod
    def from_networkx(cls, g):
        """从 ``networkx.Graph`` 构造。

        无向图会补成双向边，与 DGL ``from_networkx`` 的行为一致；
        节点按 ``list(g.nodes())`` 的顺序重编号。
        """
        nodes = list(g.nodes())
        mapper = {v: i for i, v in enumerate(nodes)}
        edges = []
        for u, v in g.edges():
            edges.append((mapper[u], mapper[v]))
            if not g.is_directed():
                edges.append((mapper[v], mapper[u]))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        return cls(len(nodes), edge_index)
