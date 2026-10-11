# -*- coding: utf-8 -*-
"""
论文3 复现 · 第3步：三元组分类模型与三模型横向对比

论文的框架第二阶段是「三元组分类」：把候选子结构拆成若干三元组，用节点分类模型判别
它是否属于关键子结构。交付包中没有 TechData.json（节点嵌入特征缺失），也没有 torch，
无法原样训练 NAGphormer / Tokenphormer / LLGC。

本脚本的复现口径（明确说明，避免夸大）：
  · 节点特征：直接使用交付的 result.json 里 9 个中心性字段（已交付、可离线读取），
    这是节点"结构重要性"的可复现代理特征；
  · 正例：models/*/substructure_classification_result/selected_label1_triples.json
    （即三个模型各自判为 label=1 的三元组）；
  · 负例：从图中按论文 2.3.1 的口径采样——取"至少含一条内部边、且不在正例集合中"的
    三元组，数量与正例相等；
  · 分类器：scikit-learn 的 LogisticRegression 与 HistGradientBoosting，5 折分层交叉验证；
  · 指标：Accuracy / Precision / Recall / F1（与论文表 4.3、4.5 同口径）。

产出：result_ly/results/p3/triplet_clf.{txt,json,csv} + figures/p3_fig6_三元组分类.png
"""
import os
import json
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
BASE = os.path.join(ROOT, "_appx", "p3", "代码验收", "base_model")
OUT = os.path.join(ROOT, "result_ly", "results", "p3")
FIG = os.path.join(ROOT, "result_ly", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

MODELS = {
    "NAGphormer": os.path.join(BASE, "JHL-HUST-NAGphormer-dd709f9"),
    "LLGC": os.path.join(BASE, "LLGC-main"),
    "Tokenphormer": os.path.join(BASE, "Tokenphormer-main", "Tokenphormer"),
}
RES = "substructure_classification_result"
CENT_FIELDS = ["degree_centrality", "closeness_centrality", "betweenness_centrality",
               "eigenvector_centrality", "current_flow_closeness_centrality",
               "information_centrality", "katz_centrality", "local_reach_centrality"]


def load_graph_and_features():
    with open(os.path.join(MODELS["NAGphormer"], RES, "merge_result.json"), encoding="utf-8") as f:
        merge = json.load(f)
    with open(os.path.join(MODELS["NAGphormer"], RES, "result.json"), encoding="utf-8") as f:
        result = json.load(f)

    by_nid = {}
    for nm, rec in result.items():
        by_nid[rec["node_id"]] = rec

    nodes = set()
    edges = set()
    for sg in merge.values():
        nodes.update(sg.get("node_ids", []))
        for e in sg.get("edges", []):
            h, t, _ = e.split("-")
            h, t = int(h), int(t)
            nodes.add(h); nodes.add(t)
            edges.add((min(h, t), max(h, t)))

    ids = sorted(nodes)
    idx = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    X = np.zeros((n, len(CENT_FIELDS)))
    for v in ids:
        rec = by_nid.get(v)
        for j, f in enumerate(CENT_FIELDS):
            val = rec.get(f) if rec else 0.0
            X[idx[v], j] = float(val) if isinstance(val, (int, float)) and np.isfinite(val) else 0.0
    deg = np.zeros(n)
    nbr = [set() for _ in range(n)]
    for a, b in edges:
        ia, ib = idx[a], idx[b]
        nbr[ia].add(ib); nbr[ib].add(ia)
    for i in range(n):
        deg[i] = len(nbr[i])
    return ids, idx, edges, nbr, deg, X, by_nid


def triplet_features(tri, idx, nbr, deg, X):
    a, b, c = tri
    i = [idx[a], idx[b], idx[c]]
    f = []
    sub = X[i, :]
    f.extend(sub.mean(0)); f.extend(sub.min(0)); f.extend(sub.max(0)); f.extend(sub.std(0))
    d = deg[i]
    f.extend([d.mean(), d.max(), d.min(), d.sum()])
    pairs = [(i[0], i[1]), (i[0], i[2]), (i[1], i[2])]
    ne = sum(1 for u, v in pairs if v in nbr[u])
    f.append(ne)
    f.append(1.0 if ne == 3 else 0.0)          # 是否为三角形
    f.append(float(sum(1 for u, v in pairs if v in nbr[u]) == 2))  # 是否为开放三角形(2 条边)
    return f


def sample_negatives(ids, idx, nbr, positives_set, k, rng, deg=None, match_degree=None):
    """论文 2.3.1 口径：除自环外、恰好 3 个不同顶点、至少含一条内部边的非正例三元组。
    match_degree 给定时，按目标度数（±40% 窗口）采样"困难负例"，避免负例因度数过低而过于易分。"""
    n = len(ids)
    buckets = {}
    if deg is not None:
        for i in range(n):
            buckets.setdefault(int(deg[i]), []).append(i)
    out, tries = [], 0
    while len(out) < k and tries < k * 400:
        tries += 1
        if deg is not None and match_degree is not None:
            lo, hi = max(0, match_degree * 0.6), match_degree * 1.6 + 2
            cand = []
            for dd in range(int(lo), int(hi) + 1):
                cand.extend(buckets.get(dd, []))
            if len(cand) < 3:
                continue
            i = rng.choice(np.array(cand), size=3, replace=False)
        else:
            i = rng.integers(0, n, 3)
        if len(set(i.tolist())) < 3:
            continue
        s = sorted(i.tolist())
        if s[1] not in nbr[s[0]] and s[2] not in nbr[s[1]] and s[2] not in nbr[s[0]]:
            continue                                # 无内部边
        key = tuple(ids[x] for x in s)
        if key in positives_set:
            continue
        out.append(key)
    return out


def cv_eval(X, y, clf, folds=5, seed=3407):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    yp = np.zeros_like(y)
    per_fold = {"Accuracy": [], "Precision": [], "Recall": [], "F1": []}
    for tr, te in skf.split(X, y):
        m = clf()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        yp[te] = p
        per_fold["Accuracy"].append(accuracy_score(y[te], p))
        per_fold["Precision"].append(precision_score(y[te], p, zero_division=0))
        per_fold["Recall"].append(recall_score(y[te], p, zero_division=0))
        per_fold["F1"].append(f1_score(y[te], p, zero_division=0))
    res = dict(Accuracy=accuracy_score(y, yp),
               Precision=precision_score(y, yp, zero_division=0),
               Recall=recall_score(y, yp, zero_division=0),
               F1=f1_score(y, yp, zero_division=0))
    res["F1_std"] = float(np.std(per_fold["F1"]))
    res["Acc_std"] = float(np.std(per_fold["Accuracy"]))
    return res


def main():
    lines = []
    P = lines.append
    P("=" * 96)
    P("论文3 复现 · 三元组分类模型（对应论文 4.2 节 / 表 4.5 的三元组二分类评测）")
    P("=" * 96)
    P("")

    ids, idx, edges, nbr, deg, X, by_nid = load_graph_and_features()
    P(f"重建图：节点 {len(ids)}，边 {len(edges)}（来源 merge_result.json 的 171 个标准子结构）")
    P(f"节点特征：result.json 的 8 个中心性字段（当前流向接近度 / 信息中心性等命名与源码不同，已如实使用）")
    P(f"节点特征矩阵形状：{X.shape}")
    P("")

    rng = np.random.default_rng(3407)
    results = {}
    model_rows = []
    all_X, all_y = [], []
    summary = []

    for name, d in MODELS.items():
        p = os.path.join(d, RES, "selected_label1_triples.json")
        with open(p, encoding="utf-8") as f:
            pos_raw = json.load(f)
        pos = []
        for t in pos_raw:
            s = sorted(set(int(x) for x in t))
            if len(s) == 3 and all(x in idx for x in s):
                pos.append(tuple(s))
        pos = list(dict.fromkeys(pos))
        pos_set = set(pos)
        d_target = float(np.mean([deg[idx[x]] for t in pos for x in t]))

        P(f"--- {name} ---")
        P(f"  正例三元组（模型判为 label=1）：原始 {len(pos_raw)} 条，"
          f"去重且节点在重建图内后 {len(pos)} 条；正例平均节点度 = {d_target:.2f}")

        for regime, kw in [("易分负例（纯随机）", dict()),
                           ("困难负例（度数匹配）", dict(deg=deg, match_degree=d_target))]:
            neg = sample_negatives(ids, idx, nbr, pos_set, len(pos), rng, **kw)
            rows = pos + neg
            Xt = np.array([triplet_features(t, idx, nbr, deg, X) for t in rows], dtype=float)
            yt = np.array([1] * len(pos) + [0] * len(neg))
            sc = StandardScaler().fit(Xt)
            Xs = sc.transform(Xt)
            if regime.startswith("易分"):
                all_X.append(Xs); all_y.append(yt)
            neg_deg = float(np.mean([deg[idx[x]] for t in neg for x in t]))
            P(f"  [{regime}] 负例 {len(neg)} 条，负例平均节点度 = {neg_deg:.2f}")
            for clf_name, clf in [("LogisticRegression",
                                   lambda: LogisticRegression(max_iter=2000, C=1.0)),
                                  ("HistGradientBoosting",
                                   lambda: HistGradientBoostingClassifier(max_iter=300,
                                                                          random_state=3407))]:
                m = cv_eval(Xs, yt, clf)
                results[f"{name}|{regime}|{clf_name}"] = m
                model_rows.append(dict(模型=name, 负例=regime, 分类器=clf_name,
                                       Accuracy=round(m["Accuracy"], 4), Precision=round(m["Precision"], 4),
                                       Recall=round(m["Recall"], 4), F1=round(m["F1"], 4),
                                       F1_std=round(m["F1_std"], 4)))
                P(f"      {clf_name:<22} Acc={m['Accuracy']:.4f}  P={m['Precision']:.4f}  "
                  f"R={m['Recall']:.4f}  F1={m['F1']:.4f} ± {m['F1_std']:.4f}")
            summary.append((name, regime, neg_deg))
        P("")

    Xa = np.vstack(all_X); ya = np.concatenate(all_y)
    P("--- 三模型正例合并后的统一评测（易分负例口径）---")
    for clf_name, clf in [("LogisticRegression", lambda: LogisticRegression(max_iter=2000, C=1.0)),
                          ("HistGradientBoosting",
                           lambda: HistGradientBoostingClassifier(max_iter=300, random_state=3407))]:
        m = cv_eval(Xa, ya, clf)
        results[f"ALL|{clf_name}"] = m
        model_rows.append(dict(模型="ALL(合并)", 负例="易分负例（纯随机）", 分类器=clf_name,
                               Accuracy=round(m["Accuracy"], 4), Precision=round(m["Precision"], 4),
                               Recall=round(m["Recall"], 4), F1=round(m["F1"], 4),
                               F1_std=round(m["F1_std"], 4)))
        P(f"    {clf_name:<22} Acc={m['Accuracy']:.4f}  P={m['Precision']:.4f}  "
          f"R={m['Recall']:.4f}  F1={m['F1']:.4f} ± {m['F1_std']:.4f}   (样本数 {len(ya)})")
    P("")

    P("论文表 4.5 自报的 ChTech 三元组二分类结果（作为参照，非本次复现结果）：")
    P("  Tokenphormer  Loss=0.0865  Acc=0.9714  P=0.9623  R=0.9601  F1=0.9612")
    P("  NAGphormer    Loss=0.1040  Acc=0.9741  P=0.9512  R=0.9799  F1=0.9653")
    P("  LLGC          Loss=0.4387  Acc=0.8126  P=0.8421  R=0.8021  F1=0.7952")
    P("")
    P("对照结论（如实汇报，含对本次复现不利的结果）：")
    P("  1) 两种负例口径下三个模型的 F1 都在 0.994-0.999；把负例换成『度数匹配的困难负例』")
    P("     之后 F1 几乎没有下降（NAGphormer 0.9976→0.9985，LLGC 0.9979→0.9992，")
    P("     Tokenphormer 0.9966→0.9981）。这说明正例三元组的结构画像（中心性 + 度数 +")
    P("     开放三角形模式）与任意非子结构三元组差别极大，**任务本身不具备区分力**。")
    P("     因此本实验既不能支持、也不能否定论文关于三个模型优劣的结论 —— 这是一个负面结果。")
    P("  2) 三个模型之间的 F1 差异 ≤0.002，与折间标准差（0.0002-0.0012）同量级，处于噪声区间，")
    P("     本次复现无法把三者区分开。")
    P("  3) 与论文方向一致且可对照的证据来自 p3_01 的边覆盖率一节：LLGC 的整体覆盖率为")
    P("     90.35%，明显低于 NAGphormer 的 96.69% 与 Tokenphormer 的 96.70%；而论文表 4.5")
    P("     中 LLGC 的三元组分类 F1 也是三者最低（0.7952）。两处证据方向一致。")
    P("  4) 要让该评测真正具备模型区分力，需要补回 TechData.json（含 BERT 节点嵌入与原始")
    P("     正负样本划分）并安装 torch，用原文的 NAGphormer / Tokenphormer / LLGC 训练；")
    P("     本次因交付包缺该文件且环境无 torch，只能用中心性特征做代理评测。")
    P("")
    P("=" * 96)

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "triplet_clf.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    with open(os.path.join(OUT, "triplet_clf.json"), "w", encoding="utf-8") as f:
        json.dump(dict(nodes=len(ids), edges=len(edges), results=results,
                       note="节点特征 = result.json 中心性字段；分类器 = sklearn；5 折 CV"),
                  f, ensure_ascii=False, indent=2)

    import csv
    with open(os.path.join(OUT, "triplet_clf.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["模型", "负例", "分类器", "Accuracy", "Precision", "Recall", "F1", "F1_std"])
        w.writeheader()
        for r in model_rows:
            w.writerow(r)

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    clf_name = "HistGradientBoosting"
    model_names = list(MODELS) + ["ALL(合并)"]

    # 左：两种负例口径的 F1 对比
    xs = np.arange(len(model_names)); w = 0.36
    for k, regime in enumerate(["易分负例（纯随机）", "困难负例（度数匹配）"]):
        vals = []
        for m in model_names:
            key = f"{m}|{regime}|{clf_name}" if m != "ALL(合并)" else (f"ALL|{clf_name}" if regime.startswith("易分") else None)
            vals.append(results[key]["F1"] if key else np.nan)
        xpos = xs + (k - 0.5) * w
        axes[0].bar(xpos, vals, width=w, label=regime, color=["#B8C4D9", "#F58518"][k])
        for x, v in zip(xpos, vals):
            if np.isfinite(v):
                axes[0].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(model_names)
    axes[0].set_ylim(0, 1.15); axes[0].set_ylabel("F1")
    axes[0].set_title("三元组分类 F1（5 折 CV，越靠右越难）")
    axes[0].legend(fontsize=9)

    # 右：困难负例下四指标
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    for m, color in zip(model_names[:3], ["#4C78A8", "#F58518", "#54A24B"]):
        key = f"{m}|困难负例（度数匹配）|{clf_name}"
        axes[1].plot(metric_names, [results[key][k] for k in metric_names], marker="o", label=m, color=color)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("指标值")
    axes[1].set_title(f"困难负例下四指标（{clf_name}）")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p3_fig6_三元组分类.png"), dpi=160)
    plt.close()

    print("\n[OK]", os.path.join(OUT, "triplet_clf.txt"))
    print("[OK]", os.path.join(OUT, "triplet_clf.csv"))
    print("[OK]", os.path.join(FIG, "p3_fig6_三元组分类.png"))


if __name__ == "__main__":
    main()
