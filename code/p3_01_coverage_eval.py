# -*- coding: utf-8 -*-
"""
论文3 复现 · 第1步：边覆盖率评测（忠实复刻 substructure_restore.py 的统计逻辑）

附录 base_model/JHL-HUST-NAGphormer-dd709f9/substructure_restore.py 第 105-159 行
的统计过程只依赖两个文件：
    substructure_classification_result/restored_datas.json   （模型预测正例还原出的边）
    substructure_classification_result/merge_result.json     （人工审核后的标准子结构）
这两个文件在交付包内都存在，因此该评测可以**原样离线复算**，无需 torch。

（说明：脚本前半段还需要 dataset/TechData.json 来生成 restored_datas.json，
  该文件在交付包中缺失，所以复现的是"评测"环节而非"还原"环节。）

产出：result_ly/results/p3/coverage_eval.{txt,csv} + figures/p3_fig*_覆盖率*.png
"""
import os
import json
import csv
import numpy as np
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


def load_restored_edges(path):
    """对应源码第 84-103 行：把 restored_datas 里的 "h-t-type" 展开成双向边集合。"""
    with open(path, encoding="utf-8") as f:
        restored = json.load(f)
    edges = set()
    bad = 0
    for sub in restored:
        for e in sub:
            try:
                h, t, typ = e.split("-")
                h, t = int(h), int(t)
                edges.add((h, t, typ))
                edges.add((t, h, typ))
            except ValueError:
                bad += 1
    return edges, len(restored), bad


def load_merge(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cover_one(model_dir):
    """对应源码第 113-143 行。"""
    re_set, n_restored, bad = load_restored_edges(os.path.join(model_dir, RES, "restored_datas.json"))
    merge = load_merge(os.path.join(model_dir, RES, "merge_result.json"))

    per_sub, total_edges, covered = {}, 0, 0
    uncovered = []
    for name, sg in merge.items():
        es = sg.get("edges", [])
        c = 0
        for e in es:
            total_edges += 1
            try:
                h, t, typ = e.split("-")
                h, t = int(h), int(t)
                if (h, t, typ) in re_set:
                    c += 1
                    covered += 1
                else:
                    uncovered.append((e, name))
            except ValueError:
                pass
        per_sub[name] = (c / len(es)) if es else 0.0
    overall = covered / total_edges if total_edges else 0.0
    return dict(n_restored=n_restored, bad_edges=bad, n_subgraphs=len(merge),
                total_standard_edges=total_edges, covered_edges=covered,
                overall=overall, per_sub=per_sub, uncovered=uncovered,
                n_pred_edges=len(re_set) // 2)


def main():
    stats = {m: cover_one(d) for m, d in MODELS.items()}

    lines = []
    P = lines.append
    P("=" * 92)
    P("论文3 复现 · 边覆盖率评测（复刻 substructure_restore.py:105-159 的统计逻辑）")
    P("=" * 92)
    P("")
    P("指标定义（源码原义）：")
    P("  整体覆盖率 = 被模型预测命中的标准边数 / 标准边总数")
    P("  标准边 = merge_result.json 中每个 SubGraph 的 edges 字段（共 171 个子结构）")
    P("  命中判据 = (head, tail, edge_type) 出现在 restored_datas.json 展开的双向边集合中")
    P("")

    hdr = f"{'模型':<14}{'预测正例三元组':>14}{'还原边条数':>12}{'标准边总数':>12}{'命中边数':>10}{'整体覆盖率':>12}"
    P(hdr)
    P("-" * len(hdr))
    for m, s in stats.items():
        P(f"{m:<14}{s['n_restored']:>14}{s['n_pred_edges']:>12}{s['total_standard_edges']:>12}"
          f"{s['covered_edges']:>10}{s['overall']*100:>11.2f}%")
    P("")

    # 每子结构覆盖率分布
    P("各 SubGraph 覆盖率分布（171 个）：")
    hdr2 = f"{'模型':<14}{'均值':>9}{'中位数':>9}{'为0的子结构数':>14}{'=100%的子结构数':>16}{'最低':>8}{'最高':>8}"
    P(hdr2)
    P("-" * len(hdr2))
    dist = {}
    for m, s in stats.items():
        arr = np.array(list(s["per_sub"].values()))
        dist[m] = arr
        P(f"{m:<14}{arr.mean()*100:>8.2f}%{np.median(arr)*100:>8.2f}%"
          f"{(arr == 0).sum():>14}{(arr >= 0.999).sum():>16}{arr.min()*100:>7.1f}%{arr.max()*100:>7.1f}%")
    P("")

    # 子结构粒度命中统计
    P("子结构粒度命中情况（一个 SubGraph 的边被命中比例分档）：")
    bins = [0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0001]
    names = ["完全未命中(0%)", "0-25%", "25-50%", "50-75%", "75-100%", "完全命中(100%)"]
    hdr3 = f"{'模型':<14}" + "".join(f"{n:>16}" for n in names)
    P(hdr3)
    P("-" * len(hdr3))
    binned = {}
    for m, s in stats.items():
        arr = np.array(list(s["per_sub"].values()))
        cnt = []
        for i in range(len(bins) - 1):
            if i == 0:
                cnt.append(int((arr <= 0.001).sum()))
            elif i == len(bins) - 2:
                cnt.append(int((arr >= 0.999).sum()))
            else:
                cnt.append(int(((arr > bins[i]) & (arr <= bins[i + 1])).sum()))
        binned[m] = cnt
        P(f"{m:<14}" + "".join(f"{c:>16}" for c in cnt))
    P("")

    # 与论文自报值的对照
    P("论文自报的对照口径（正文第 47-48 页提到：通过对比中心度子结构，在边覆盖和漏预测等")
    P("指标下挖掘出了诸如区块链相关的『能源-农业』跨域协作子结构、华为相关的通信行业聚集")
    P("子结构等，验证了分类模型在关键子结构挖掘上的可行性；表 4.5 另给出 ChTech 上三元组")
    P("二分类的 Loss/Acc/P/R/F1）：")
    P("  Tokenphormer  Acc=0.9714 P=0.9623 R=0.9601 F1=0.9612")
    P("  NAGphormer    Acc=0.9741 P=0.9512 R=0.9799 F1=0.9653")
    P("  LLGC          Acc=0.8126 P=0.8421 R=0.8021 F1=0.7952")
    P("  → 上表是三元组二分类指标；本节的边覆盖率是『还原后结构层面』的指标，两者口径不同。")
    P("")

    # 名称一致性检查
    P("附带检查：三个模型目录下 merge_result.json 是否一致")
    sizes = {}
    for m, d in MODELS.items():
        p = os.path.join(d, RES, "merge_result.json")
        sizes[m] = os.path.getsize(p)
    P(f"  merge_result.json 字节数：{sizes}")
    P(f"  全部相同：{len(set(sizes.values())) == 1}")
    P("")
    P("=" * 92)

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "coverage_eval.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    # ---- CSV ----
    with open(os.path.join(OUT, "coverage_eval_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型", "预测正例三元组数", "还原边条数", "标准边总数", "命中边数", "整体覆盖率",
                    "子结构均值", "子结构中位数", "完全未命中数", "完全命中数"])
        for m, s in stats.items():
            arr = dist[m]
            w.writerow([m, s["n_restored"], s["n_pred_edges"], s["total_standard_edges"],
                        s["covered_edges"], f"{s['overall']*100:.4f}%",
                        f"{arr.mean()*100:.4f}%", f"{np.median(arr)*100:.4f}%",
                        int((arr == 0).sum()), int((arr >= 0.999).sum())])
    # 每子结构明细
    with open(os.path.join(OUT, "coverage_per_subgraph.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["SubGraph"] + list(MODELS))
        keys = list(stats["NAGphormer"]["per_sub"])
        for k in keys:
            w.writerow([k] + [f"{stats[m]['per_sub'][k]*100:.2f}%" for m in MODELS])

    # ---- 图1：整体覆盖率对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ms = list(MODELS)
    ov = [stats[m]["overall"] * 100 for m in ms]
    bars = axes[0].bar(ms, ov, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylabel("整体边覆盖率 (%)")
    axes[0].set_title("三个模型的标准子结构边覆盖率")
    axes[0].set_ylim(0, max(ov) * 1.35 if max(ov) > 0 else 1)
    for b, v in zip(bars, ov):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=11)

    # 预测正例规模
    sz = [stats[m]["n_restored"] for m in ms]
    bars = axes[1].bar(ms, sz, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_ylabel("预测为正例的三元组数量")
    axes[1].set_title("模型判为正例的三元组规模")
    for b, v in zip(bars, sz):
        axes[1].text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p3_fig1_覆盖率对比.png"), dpi=160)
    plt.close()

    # ---- 图2：子结构覆盖率分布 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, m in zip(axes, ms):
        ax.hist(dist[m] * 100, bins=21, range=(0, 100), color="#4C78A8", edgecolor="white")
        ax.set_title(f"{m}  （均值 {dist[m].mean()*100:.1f}%）")
        ax.set_xlabel("单个 SubGraph 的边覆盖率 (%)")
        ax.set_ylabel("SubGraph 个数")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p3_fig2_子结构覆盖率分布.png"), dpi=160)
    plt.close()

    # ---- 图3：分档堆叠 ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bottom = np.zeros(len(ms))
    colors = ["#D64545", "#F58518", "#F2C14E", "#9CCB86", "#54A24B", "#2E6F40"]
    for i, n in enumerate(names):
        vals = np.array([binned[m][i] for m in ms])
        ax.bar(ms, vals, bottom=bottom, label=n, color=colors[i])
        for x, (b, v) in enumerate(zip(bottom, vals)):
            if v > 8:
                ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=9)
        bottom += vals
    ax.set_ylabel("SubGraph 个数（共 171）")
    ax.set_title("子结构粒度的边命中分档")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p3_fig3_命中分档.png"), dpi=160)
    plt.close()

    print("\n[OK] 结果文件：")
    for p in ["coverage_eval.txt", "coverage_eval_summary.csv", "coverage_per_subgraph.csv"]:
        print("  ", os.path.join(OUT, p))
    for p in ["p3_fig1_覆盖率对比.png", "p3_fig2_子结构覆盖率分布.png", "p3_fig3_命中分档.png"]:
        print("  ", os.path.join(FIG, p))


if __name__ == "__main__":
    main()
