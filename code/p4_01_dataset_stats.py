# -*- coding: utf-8 -*-
"""
论文4 复现 · 第1步：数据集统计与可视化
复刻附录 ner/data.py 中 build_corpus_technology() 的解析规则：
    每行 "字/词 标签"，空行分隔样本；末项为标签，其余拼回为词。
产出：result_ly/results/p4/dataset_stats.{md,csv} + 若干图
"""
import os
import sys
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def md_table(df):
    """不依赖 tabulate 的 Markdown 表格输出。"""
    df = pd.DataFrame(df)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


ROOT = r"C:\Users\LiYi\Desktop\大创文件"
NER = os.path.join(ROOT, "_appx", "p4", "ner")
OUT = os.path.join(ROOT, "result_ly", "results", "p4")
FIG = os.path.join(ROOT, "result_ly", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SPLITS = [
    ("科技·字符级", "TechnologyNer", "train.char.bmes"),
    ("科技·字符级", "TechnologyNer", "dev.char.bmes"),
    ("科技·字符级", "TechnologyNer", "test.char.bmes"),
    ("科技·词语级", "TechnologyNer_word", "train.char.bmes"),
    ("科技·词语级", "TechnologyNer_word", "dev.char.bmes"),
    ("科技·词语级", "TechnologyNer_word", "test.char.bmes"),
]


def parse_bmes(path):
    """返回 (sentences, tags)，sentences 为 list[list[str]]，tags 同形。"""
    sents, tags = [], []
    cur_w, cur_t = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.replace("\r", "").replace("\n", "")
            if line.strip() == "":
                if cur_w:
                    sents.append(cur_w)
                    tags.append(cur_t)
                    cur_w, cur_t = [], []
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            tag = parts[-1]
            word = "".join(parts[:-1])
            cur_w.append(word)
            cur_t.append(tag)
    if cur_w:
        sents.append(cur_w)
        tags.append(cur_t)
    return sents, tags


def entity_spans(tags):
    """BIOES -> [(type, start, end)]，与附录 BIOES_eval.py 的 get_entities 等价。"""
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            typ = t[2:]
            j = i + 1
            while j < n and tags[j] == "I-" + typ:
                j += 1
            if j < n and tags[j] == "E-" + typ:
                spans.append((typ, i, j))
                i = j + 1
                continue
            spans.append((typ, i, j - 1))
            i = j
        elif t.startswith("S-"):
            spans.append((t[2:], i, i))
            i += 1
        elif t == "B":
            j = i + 1
            while j < n and tags[j] == "I":
                j += 1
            if j < n and tags[j] == "E":
                spans.append(("UNTYPED", i, j))
                i = j + 1
                continue
            spans.append(("UNTYPED", i, j - 1))
            i = j
        elif t == "S":
            spans.append(("UNTYPED", i, i))
            i += 1
        else:
            i += 1
    return spans


def main():
    rows = []
    all_tags = {}
    all_sents = {}
    for name, d, fn in SPLITS:
        p = os.path.join(NER, d, fn)
        sents, tags = parse_bmes(p)
        tc = Counter(t for seq in tags for t in seq)
        vocab = set(w for seq in sents for w in seq)
        L = [len(s) for s in sents]
        ents = [e for seq in tags for e in entity_spans(seq)]
        etypes = Counter(t for t, _, _ in ents)
        elen = [b - a + 1 for _, a, b in ents]
        rows.append(dict(
            数据集=name, 划分=fn.replace(".char.bmes", ""),
            文件=os.path.join(d, fn),
            样本条数=len(sents), 总token数=sum(L), 词表大小=len(vocab),
            标签种类数=len(tc), 平均句长=round(float(np.mean(L)), 2),
            最大句长=int(np.max(L)),
            实体提及数=len(ents), 实体类型数=len(etypes),
            平均实体长度=round(float(np.mean(elen)), 2) if elen else 0.0,
        ))
        all_tags[(name, fn)] = tc
        all_sents[(name, fn)] = (sents, tags, L)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "dataset_overview.csv"), index=False, encoding="utf-8-sig")

    # ---- Markdown 报告 ----
    md = ["# 论文4 复现 · 数据集统计（离线实测）", "",
          "> 由 `code/p4_repro/p4_01_dataset_stats.py` 生成；解析规则复刻 `ner/data.py:108-128`。", "",
          "## 1. 总览", "", md_table(df), ""]

    md += ["## 2. 标签分布（字符级）", ""]
    ch = {k: v for k, v in all_tags.items() if k[0] == "科技·字符级"}
    labels_sorted = sorted(set().union(*[set(v) for v in ch.values()]),
                           key=lambda x: -sum(v.get(x, 0) for v in ch.values()))
    t2 = pd.DataFrame({k[1].replace(".char.bmes", ""): [v.get(l, 0) for l in labels_sorted] for k, v in ch.items()},
                      index=labels_sorted)
    t2.insert(0, "标签", t2.index)
    md += [md_table(t2), ""]

    md += ["## 3. 标签分布（词语级）", ""]
    wd = {k: v for k, v in all_tags.items() if k[0] == "科技·词语级"}
    labels_sorted2 = sorted(set().union(*[set(v) for v in wd.values()]),
                            key=lambda x: -sum(v.get(x, 0) for v in wd.values()))
    t3 = pd.DataFrame({k[1].replace(".char.bmes", ""): [v.get(l, 0) for l in labels_sorted2] for k, v in wd.items()},
                      index=labels_sorted2)
    t3.insert(0, "标签", t3.index)
    md += [md_table(t3), ""]

    # 类型汇总
    md += ["## 4. 实体类型（带类型后缀标签）汇总", ""]
    tmap = defaultdict(lambda: Counter())
    for (name, fn), tc in all_tags.items():
        for tag, c in tc.items():
            if "-" in tag and tag.split("-")[0] in ("B", "I", "E", "S"):
                tmap[tag.split("-")[1]][name + "/" + fn.replace(".char.bmes", "")] += c
    trows = []
    for typ, cn in sorted(tmap.items(), key=lambda kv: -sum(kv[1].values())):
        trows.append(dict(类型代码=typ, 出现总次数=sum(cn.values())))
    md += [md_table(pd.DataFrame(trows)), ""]
    md += ["注：类型代码含义在附录代码中未给出映射表，正文中出现的解释为 "
           "`js(技术) / cp(产品) / gs(公司) / ly(领域)`（论文 2.2.1 节，第 13 页）。", ""]

    # 无类型 vs 有类型
    md += ["## 5. 无类型标签 vs 带类型后缀标签（token 占比）", ""]
    trows = []
    for (name, fn), tc in all_tags.items():
        total = sum(tc.values())
        typed = sum(c for t, c in tc.items() if "-" in t)
        trows.append(dict(数据集=name, 划分=fn.replace(".char.bmes", ""),
                          带类型token=typed, 无类型token=total - typed,
                          带类型占比=f"{100*typed/total:.2f}%"))
    md += [md_table(pd.DataFrame(trows)), ""]

    with open(os.path.join(OUT, "dataset_stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---- 图 1：句长分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, level in zip(axes, ["科技·字符级", "科技·词语级"]):
        for fn in ["train.char.bmes", "dev.char.bmes", "test.char.bmes"]:
            sents, tags, L = all_sents[(level, fn)]
            ax.hist(L, bins=60, alpha=0.55, label=fn.split(".")[0], density=True)
        ax.set_title(f"{level} 句子长度分布")
        ax.set_xlabel("句子长度（token 数）")
        ax.set_ylabel("密度")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p4_fig1_句长分布.png"), dpi=150)
    plt.close()

    # ---- 图 2：标签分布（字符级/词语级）----
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, (level, tdf) in zip(axes, [("科技·字符级", t2), ("科技·词语级", t3)]):
        d = tdf[tdf["标签"] != "O"].set_index("标签")[["train", "dev", "test"]]
        d = d.loc[d.sum(axis=1).sort_values(ascending=False).index][:25]
        d.plot(kind="barh", ax=ax, width=0.8)
        ax.invert_yaxis()
        ax.set_title(f"{level} 各标签 token 数（Top25，已剔除 O）")
        ax.set_xlabel("token 数")
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p4_fig2_标签分布.png"), dpi=150)
    plt.close()

    # ---- 图 3：实体长度分布 + 实体类型频次 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    elens, etypes_cnt = Counter(), Counter()
    for (name, fn), (sents, tags, L) in all_sents.items():
        if fn != "train.char.bmes":
            continue
        for seq in tags:
            for typ, a, b in entity_spans(seq):
                elens[b - a + 1] += 1
                etypes_cnt[typ] += 1
    ks = sorted(elens)
    axes[0].bar([str(k) for k in ks], [elens[k] for k in ks], color="#4C78A8")
    axes[0].set_title("实体长度分布（train）")
    axes[0].set_xlabel("实体包含的 token 数")
    axes[0].set_ylabel("实体个数")
    top = etypes_cnt.most_common(15)
    axes[1].barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#F58518")
    axes[1].set_title("实体类型频次（train，Top15）")
    axes[1].set_xlabel("实体个数")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "p4_fig3_实体长度与类型.png"), dpi=150)
    plt.close()

    print(df.to_string(index=False))
    print("\n[OK] 输出：")
    print(" ", os.path.join(OUT, "dataset_overview.csv"))
    print(" ", os.path.join(OUT, "dataset_stats.md"))
    print(" ", os.path.join(FIG, "p4_fig1_句长分布.png"))
    print(" ", os.path.join(FIG, "p4_fig2_标签分布.png"))
    print(" ", os.path.join(FIG, "p4_fig3_实体长度与类型.png"))


if __name__ == "__main__":
    main()
