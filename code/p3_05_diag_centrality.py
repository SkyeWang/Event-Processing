# -*- coding: utf-8 -*-
"""论文3 复现 · 诊断：作者 centrality_utils.py 在 Cora 上 closeness 全 0、katz 全 NaN 的根因。"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import igraph as ig

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
CORADIR = os.path.join(ROOT, "_appx", "p3", "代码验收", "cora_citeseer")
sys.path.insert(0, CORADIR)
os.chdir(CORADIR)

import torch                                 # noqa: E402
from torch_geometric.datasets import Planetoid   # noqa: E402

DATA = os.path.join(ROOT, "_data", "pyg")


class PlanetoidLocal(Planetoid):
    def __init__(self, root=DATA, name="Cora", *a, **kw):
        super().__init__(root=DATA, name=name, *a, **kw)


dataset = PlanetoidLocal()
data = dataset[0]
edge_list = [(int(s), int(d)) for s, d in data.edge_index.t().numpy()]
g = ig.Graph(edges=edge_list, directed=False)
g.vs["feature"] = data.x.numpy()
g.vs["label"] = data.y.numpy()
print(f"图：节点 {g.vcount()}，边 {g.ecount()}（PyG edge_index 含双向，去重后 {len(set(map(tuple, map(sorted, edge_list))))}）")

from centrality_utils import AdvancedCentralities  # noqa: E402

ac = AdvancedCentralities(g)

print("\n[closeness 根因]")
clo = ac.closeness_centrality()
clo = np.asarray(clo, dtype=float)
print(f"  原始 closure 值：min={clo.min():.6g} max={clo.max():.6g} 均值={clo.mean():.6g} 非零占比={np.mean(clo>0):.1%}")
sp = g.shortest_paths()
arr = np.asarray(sp, dtype=float)
print(f"  shortest_paths() 矩阵形状={arr.shape}，每行距离和：min={arr.sum(1).min():.4g} max={arr.sum(1).max():.4g}")
print(f"  距离矩阵中 inf 个数={np.isinf(arr).sum()}，0 的个数={np.sum(arr == 0)}")
row0 = arr[0]
print(f"  节点0 的距离行前 10 个：{row0[:10]}")
print(f"  → 说明：Cora 由 edge_index 构造后是【非连通图】，绝大多数节点对距离为 inf，"
      f"sum(p)==inf → (n-1)/inf == 0，故 closeness 原始值几乎全为 0；")
print(f"    再经 safe_normalize 后仍为 0（常数向量 min==max）。")
def safe_norm(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)
print(f"  归一化后：min={safe_norm(clo).min():.6g} max={safe_norm(clo).max():.6g}")

print("\n[katz 根因]")
n = g.vcount()
adj = np.asarray(g.get_adjacency().data, dtype=float)
w = np.linalg.eigvalsh(adj)
rho = float(np.abs(w).max())
print(f"  邻接矩阵谱半径 ρ(A)={rho:.4f}；α=0.1 → α·ρ={0.1*rho:.4f}"
      f"{' > 1，迭代发散' if 0.1*rho > 1 else ' <= 1，本应收敛'}")
x = np.ones(n) * 0.1
for i in range(1000):
    xn = 0.1 * adj.dot(x) + 0.1
    if np.linalg.norm(xn - x) < 1e-6:
        print(f"  收敛于第 {i} 次迭代")
        break
    x = xn
    if not np.isfinite(x).all():
        print(f"  第 {i} 次迭代出现非有限值，x 的最大绝对值={np.nanmax(np.abs(x[np.isfinite(x)])):.3e}")
        break
print(f"  迭代 1000 次后：全为有限值={np.isfinite(x).all()}，"
      f"max|x|={np.abs(x[np.isfinite(x)]).max() if np.isfinite(x).any() else float('nan'):.3e}")
print("  → 说明：作者 centrality_utils.py:58-67 固定 α=0.1 且无发散保护；")
print("     α·ρ(A) > 1 时 x 逐次放大直至 double 上溢为 inf，safe_normalize 后得到 NaN。")
