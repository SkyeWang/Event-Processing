"""SAGNN 的位置编码（去 DGL 版本）。

原实现用 DGL 的 ``adjacency_matrix_scipy`` 和 ``in_degrees`` 取邻接矩阵与度；
``core.simple_graph.SimpleGraph`` 提供了同名接口，因此下面的数学公式与
原论文/原代码保持一致，只是数据来源从 DGL 换成了 SciPy 稀疏矩阵。

两种编码：
- ``lap_positional_encoding``：拉普拉斯特征向量；
- ``init_positional_encoding``：随机游走返回概率（SAGNN 默认使用）。
"""

import numpy as np
import torch
from scipy import sparse as sp


def _degrees(g):
    """把图对象的入度转成 numpy float 数组，供 SciPy 对角矩阵使用。"""
    return g.in_degrees().cpu().numpy().astype(float)


def lap_positional_encoding(g, pos_enc_dim):
    """拉普拉斯特征向量位置编码。

    步骤：
    1. 取邻接矩阵 A 和度向量 d；
    2. 计算对称归一化拉普拉斯 ``L = I - D^{-1/2} A D^{-1/2}``；
    3. 对 L 做特征分解，按特征值升序取前 ``pos_enc_dim`` 个非平凡特征向量；
    4. 若特征向量个数不足，右侧补零。

    Args:
        g: ``SimpleGraph``，需提供 ``adjacency_matrix_scipy`` 和 ``in_degrees``。
        pos_enc_dim: 位置编码维度。

    Returns:
        ``[n, pos_enc_dim]`` 的 float tensor。
    """
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy().astype(float)
    # 度可能为 0（孤立点），clip 到 1 避免除零
    N = sp.diags(np.clip(_degrees(g), 1, None) ** -0.5, dtype=float)
    L = sp.eye(n) - N @ A @ N

    # 图规模小（MUTAG/ZINC），直接转稠密做特征分解
    eig_val, eig_vec = np.linalg.eig(L.toarray())
    idx = eig_val.argsort()
    eig_vec = np.real(eig_vec[:, idx])

    # 跳过第一个平凡特征向量（对应常数向量）
    pos_enc_emb = torch.from_numpy(eig_vec[:, 1:pos_enc_dim + 1]).float()
    if pos_enc_emb.shape[-1] < pos_enc_dim:
        offset = pos_enc_dim - pos_enc_emb.shape[-1]
        pos_enc_emb = torch.cat(
            (pos_enc_emb, torch.zeros(pos_enc_emb.shape[0], offset)), dim=-1
        )
    return pos_enc_emb


def init_positional_encoding(g, pos_enc_dim, type_init="rand_walk"):
    """随机游走返回概率位置编码（SAGNN 默认）。

    设转移矩阵 ``M = A @ D^{-1}``，则 ``(M^s)_{v,v}`` 表示从 v 出发走 s 步
    回到 v 的概率。把 s=1..S 的返回概率拼成向量，即为节点 v 的位置编码。

    Args:
        g: ``SimpleGraph``。
        pos_enc_dim: 返回概率的步数 S。
        type_init: 目前只支持 ``rand_walk``。

    Returns:
        ``[n, pos_enc_dim]`` 的 float tensor，并写入 ``g.ndata['pos_enc']``。
    """
    n = g.number_of_nodes()
    if type_init != "rand_walk":
        raise ValueError(f"Unsupported positional encoding: {type_init}")

    A = g.adjacency_matrix(scipy_fmt="csr")
    # D^{-1}：度 clip 到 1，避免孤立点除零
    Dinv = sp.diags(np.clip(_degrees(g), 1, None) ** -1.0, dtype=float)
    RW = A @ Dinv
    M = RW

    # PE[s] = diag(M^(s+1))，即 s 步返回概率
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(pos_enc_dim - 1):
        M_power = M_power @ M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    g.ndata["pos_enc"] = PE
    return PE
