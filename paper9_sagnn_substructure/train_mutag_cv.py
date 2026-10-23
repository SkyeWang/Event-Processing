"""在 MUTAG 上做分层 10 折交叉验证（SAGNN 论文协议）。

论文报告的 MUTAG 95.2±3.0 是 10 折结果；``train_mutag.py`` 的单次
150/18/20 划分只适合快速检查，不能直接对比。本脚本把 188 张图按类别
分层切成 10 折，每折用 9 折训练、1 折测试，并从训练折中再切 10% 做
验证；最后报告 10 折 test 的均值和标准差。

示例：
    python train_mutag_cv.py --folds 10 --epochs 100
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from core.config import cfg
from core.transform import SubgraphsTransform
from train_mutag import build_model, evaluate, set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "MUTAG")
RESULT_DIR = os.path.join(ROOT, "results")


def stratified_folds(dataset, k, seed):
    """按类别分层切 k 折，避免小数据集出现某折全是同一类。

    返回 ``folds[i]``：第 i 折的测试样本索引。
    """
    by_class = {}
    for i, data in enumerate(dataset):
        by_class.setdefault(int(data.y.view(-1)[0]), []).append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for label, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        # 轮转分配：每个类别的样本依次放入 0..k-1 折
        for j, idx in enumerate(indices):
            folds[j % k].append(idx)
    return folds


def train_fold(model, train_set, val_set, test_set, args, device):
    """训练并评估一折。

    ``select='best_val'``：按验证集最优时的 test 作为该折结果；
    ``select='final'``：取最后一个 epoch 的 test。
    """
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val = 0.0
    best_test = 0.0
    for epoch in range(1, args.epochs + 1):
        # ---- 训练 ----
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y.view(-1)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ---- 按验证集最优记录 test；final 模式则等训练结束再测 ----
        if args.select == "best_val" and val_set:
            _, val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                _, best_test = evaluate(model, test_loader, device)
    if args.select == "final":
        _, best_test = evaluate(model, test_loader, device)
    return best_val, best_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--gnn", type=str, default="GINEConv")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--select", choices=["best_val", "final"], default="best_val")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    print(f"device={device}")

    dataset = TUDataset(root=DATA_DIR, name="MUTAG")
    print(f"MUTAG graphs={len(dataset)}")

    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.subgraph.hops = args.hops
    cfg.model.gnn_type = args.gnn
    transform = SubgraphsTransform(cfg)

    t0 = time.time()
    all_data = [transform(dataset[i]) for i in range(len(dataset))]
    print(f"preprocessing {len(all_data)} graphs took {time.time() - t0:.1f}s")

    folds = stratified_folds(dataset, args.folds, args.seed)
    results = []
    for fold in range(args.folds):
        test_idx = folds[fold]
        train_idx = [i for j, f in enumerate(folds) if j != fold for i in f]
        if args.select == "best_val":
            # carve a small validation set out of the training folds
            rng = random.Random(args.seed + fold)
            rng.shuffle(train_idx)
            n_val = max(10, int(len(train_idx) * 0.1))
            val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
        else:
            val_idx = []

        model = build_model(args, dataset.num_node_features, dataset.num_classes).to(device)
        val_acc, test_acc = train_fold(
            model,
            [all_data[i] for i in train_idx],
            [all_data[i] for i in val_idx],
            [all_data[i] for i in test_idx],
            args,
            device,
        )
        results.append((fold, val_acc, test_acc))
        print(f"fold {fold + 1:02d}: val={val_acc:.4f} test={test_acc:.4f}")

    tests = np.array([r[2] for r in results])
    print(f"10-fold test acc: {tests.mean():.4f} ± {tests.std():.4f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "mutag_sagnn_10fold.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "best_val_acc", "test_acc"])
        writer.writerows(results)
        writer.writerow(["mean", "", float(tests.mean())])
        writer.writerow(["std", "", float(tests.std())])
    print("results ->", path)


if __name__ == "__main__":
    main()
