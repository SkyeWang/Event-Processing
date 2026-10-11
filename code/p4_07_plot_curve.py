# -*- coding: utf-8 -*-
"""论文4 复现 · 把四组配置的逐轮 test 探针 F1 画成曲线（数据到齐后运行）。"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\LiYi\Desktop\大创文件"
RES = os.path.join(ROOT, "result_ly", "results", "p4")
FIG = os.path.join(ROOT, "result_ly", "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR = {"A": "#4C78A8", "B": "#F58518", "B2": "#E45756", "C": "#54A24B", "D": "#B279A2"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
got = False
for c in ["A","B","B2","C","D"]:
    p = os.path.join(RES, f"orig_{c}.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p, encoding="utf-8"))
    cur = r.get("curve") or []
    if not cur:
        continue
    got = True
    ep = [x["epoch"] for x in cur]
    f1 = [x.get("test_probe_f1", 0) for x in cur]
    vl = [x.get("val_loss", 0) for x in cur]
    axes[0].plot(ep, f1, marker="o", color=COLOR[c], label=f"{c} · {r['desc']}")
    axes[1].plot(ep, vl, marker="o", color=COLOR[c], label=c)

if got:
    axes[0].set_xlabel("训练轮数")
    axes[0].set_ylabel("实体级 F1（test 前 256 条）")
    axes[0].set_title("四组配置的 test 探针 F1 曲线")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("训练轮数")
    axes[1].set_ylabel("验证集损失")
    axes[1].set_title("验证集损失（dev，作者口径）")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG, "p4_fig4_训练曲线.png")
    plt.savefig(out, dpi=160)
    print("[OK]", out)
else:
    print("[skip] 还没有 orig_*.json，无法出图")
