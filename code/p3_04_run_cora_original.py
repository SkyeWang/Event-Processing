# -*- coding: utf-8 -*-
"""
论文3 复现 · 第4步：在公开数据集 Cora 上直接运行附录原始流水线

复现对象：cora_citeseer/cora_subgraphs.py（作者未随附录提供 TechData.json，
          该脚本走的是公开引文网络分支，因此可以完整跑通）
流程：Planetoid 加载 Cora → GraphSAGE(2 层, 64→32, 200 epoch) 节点嵌入
      → 8 种中心性 → 每种取前 10% 作种子 → 2-hop 扩展 → 按重叠率>0.3 合并（节点数限制 [5,30]）
      → 保存 cora_subgraphs.json / summary.json → 调作者自带的 extract_triples() 抽四类三元组

唯一改动：把 Planetoid(root='/dataset') 重定向到工作区目录（避免往 C:\\dataset 写数据），
          其余代码原样执行。改动以 monkeypatch 方式注入，不改动附录文件。
"""
import os
import sys
import json
import runpy
import time
from collections import Counter

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
APPX = os.path.join(ROOT, "_appx", "p3", "代码验收")
CORADIR = os.path.join(APPX, "cora_citeseer")
DATA = os.path.join(ROOT, "_data", "pyg")
OUT = os.path.join(ROOT, "result_ly", "results", "p3")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# 让 Planetoid 直接用已下载好的 raw 文件（离线可用；文件已放在 _data/planetoid）
os.makedirs(os.path.join(DATA, "Cora", "raw"), exist_ok=True)
for f in os.listdir(os.path.join(ROOT, "_data", "planetoid")):
    if f.startswith("ind.cora."):
        dst = os.path.join(DATA, "Cora", "raw", f)
        if not os.path.exists(dst):
            src = os.path.join(ROOT, "_data", "planetoid", f)
            open(dst, "wb").write(open(src, "rb").read())


def main():
    log = []
    lines = []

    class Tee:
        """把原脚本的 print 一并收集到 log，便于存档（原脚本会打印 2708 行节点 ID）。"""
        def __init__(self, real):
            self.real = real
        def write(self, s):
            self.real.write(s)
            if s.strip():
                lines.append(s.rstrip())
        def flush(self):
            self.real.flush()

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    sys.stdout = Tee(sys.stdout)

    T0 = time.time()
    P("=" * 90)
    P("论文3 复现 · 在公开数据集 Cora 上直接运行附录原始流水线（cora_subgraphs.py）")
    P("=" * 90)

    # ---- 唯一改动：重定向 Planetoid 的 root ----
    import torch_geometric.datasets as D
    _Orig = D.Planetoid

    class PlanetoidLocal(_Orig):
        def __init__(self, root=DATA, name="Cora", *a, **kw):
            super().__init__(root=DATA, name=name, *a, **kw)

    D.Planetoid = PlanetoidLocal

    os.chdir(CORADIR)
    sys.path.insert(0, CORADIR)

    P(f"\n[0] 工作目录 {CORADIR}")
    P(f"    数据集缓存 {DATA}（预先放入 Planetoid 原始文件，避免联网）")
    P("    唯一改动：Planetoid(root=...) 指向工作区，其余代码原样执行\n")

    t0 = time.time()
    ns = runpy.run_path(os.path.join(CORADIR, "cora_subgraphs.py"), run_name="__main__")
    pipe_sec = time.time() - t0

    graph = ns["graph"]
    merged = ns["merged"]
    seeds = ns["seeds"]
    centralities = ns["centralities"]

    P(f"\n[1] 原始流水线执行完成，耗时 {pipe_sec:.1f}s")
    P(f"    图规模：节点 {graph.vcount()}，边 {graph.ecount()}（Cora 公开数据集）")
    P(f"    种子节点数（8 种中心性各取前 10% 去重后）：{len(seeds)}")
    P(f"    2-hop 扩展得到的候选子图：{len(ns['subgraphs'])}")
    P(f"    合并后子图数（节点数限制 [5,30]，重叠率>0.3 合并）：{len(merged)}")
    sizes = [len(s) for s in merged]
    P(f"    子图规模：均值 {sum(sizes)/len(sizes):.1f}，最小 {min(sizes)}，最大 {max(sizes)}")

    # ---- 中心性统计（作者 centrality_utils.py 的输出） ----
    P("\n[2] 8 种中心性统计（作者实现，igraph 版）")
    P(f"    {'指标':<14}{'均值':>10}{'标准差':>10}{'最大值':>10}")
    cent_stat = {}
    for k, v in centralities.items():
        import numpy as np
        v = np.asarray(v, dtype=float)
        cent_stat[k] = dict(mean=float(v.mean()), std=float(v.std()), max=float(v.max()))
        P(f"    {k:<14}{v.mean():>10.5f}{v.std():>10.5f}{v.max():>10.5f}")

    # ---- 调用作者自带的 extract_triples（该函数在 __main__ 里没有被调用） ----
    P("\n[3] 调用作者自带的 extract_triples() 抽取四类三元组 ...")
    outdir = os.path.join(CORADIR, "cora_subgraphs")
    os.makedirs(outdir, exist_ok=True)
    tri_path = os.path.join(outdir, "cora_triples.json")
    t0 = time.time()
    ns["extract_triples"](graph, merged, filename=tri_path, threshold=0.85)
    tri = json.load(open(tri_path, encoding="utf-8"))
    P(f"    三元组总数 {len(tri)}，耗时 {time.time()-t0:.1f}s")
    rel = Counter(t["relation"] for t in tri)
    P(f"    {'关系类型':<16}{'条数':>10}")
    for k, v in rel.most_common():
        P(f"    {k:<16}{v:>10}")
    P(f"    覆盖社区数（子图数）：{len(set(t['community'] for t in tri))}")
    P("\n    示例（来自 cora_triples.json 前 3 条）：")
    for t in tri[:3]:
        P("      " + json.dumps(t, ensure_ascii=False))

    # ---- 读回作者保存的子图/汇总文件 ----
    P("\n[4] 作者保存的产物")
    for fn in ("cora_subgraphs.json", "summary.json"):
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            obj = json.load(open(p, encoding="utf-8"))
            if isinstance(obj, dict):
                P(f"    {fn} 顶层键数={len(obj)}；键示例={list(obj)[:5]}")
                k0 = list(obj)[0]
                v0 = obj[k0]
                if isinstance(v0, dict):
                    P(f"      例 {k0}: 字段={list(v0)[:8]}")
            else:
                P(f"    {fn} 条目数={len(obj)}")

    # ---- 汇总落盘 ----
    summary = dict(
        dataset="Cora (Planetoid, 公开)",
        graph_nodes=graph.vcount(), graph_edges=graph.ecount(),
        seeds=len(seeds), candidate_subgraphs=len(ns["subgraphs"]), merged_subgraphs=len(merged),
        subgraph_size=dict(mean=sum(sizes) / len(sizes), min=min(sizes), max=max(sizes)),
        centralities=cent_stat,
        triples_total=len(tri), triples_by_relation=dict(rel),
        pipeline_seconds=round(pipe_sec, 1),
        total_seconds=round(time.time() - T0, 1),
        env="torch 2.14.1+cpu / igraph / torch_geometric / numpy，CPU",
        note="唯一改动：Planetoid root 重定向到工作区；extract_triples 由本驱动显式调用",
    )
    with open(os.path.join(OUT, "cora_pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "cora_pipeline.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    with open(os.path.join(OUT, "cora_pipeline_full.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    P(f"\n[OK] {os.path.join(OUT, 'cora_pipeline_summary.json')}")
    P(f"[OK] {os.path.join(OUT, 'cora_pipeline.log')}")
    P(f"    三元组文件：{tri_path}")
    P(f"\n总耗时 {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
