"""用原 DGL 管线在 MUTAG 上做 10 折交叉验证。

与原 PyG 版 ``train_mutag_cv.py`` 的区别：本脚本直接 import 原作者的
``core.transform`` / ``core.model`` / ``core.PE_init``，保留 DGL 的图容器、
``node_subgraph`` 和 ``to_networkx``，用于验证“换掉 DGL 是否影响结果”。

运行环境：torch 2.3 + DGL 2.2.1 + PyG 2.6.1（见 dgl_repro/README.md）。
``DATA_DIR`` 默认指向 ``../data/MUTAG``，换机器时按实际数据目录修改。

示例：
    python run_mutag_dgl.py --folds 10 --epochs 100
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
from core.model import SAGNN
from core.transform import SubgraphsTransform

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data", "MUTAG")
RESULT_DIR = os.path.join(ROOT, "results_dgl")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, nfeat, nout):
    """构造原 SAGNN 模型；MUTAG 的 nfeat=7（one-hot），nout=2。"""
    model = SAGNN(
        nfeat,
        None,
        nhid=args.hidden,
        nout=nout,
        nlayer_outer=args.layers,
        nlayer_inner=args.mini_layers,
        gnn_types=[args.gnn],
        embedding_types="rand_walk",
        hop_dim=16,
        fullgraph_pos_enc_dim=20,
        egograph_pos_enc_dim=16,
        cutgraph_pos_enc_dim=16,
        pos_enc_dim=24,
        pooling="add",
        global_embedding=True,
        dropout=0.0,
    )
    model.reset_parameters()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定 loader 上算分类准确率；不建计算图，节省显存。"""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1)
        correct += int((model(batch).argmax(-1) == y).sum())
        total += y.numel()
    return correct / max(total, 1)


def train_fold(model, train_set, val_set, test_set, args, device):
    """训练一折：按验证集最优保存 test 准确率。"""
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    best_val = best_test = 0.0
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_test = evaluate(model, test_loader, device)
    return best_val, best_test


def stratified_folds(dataset, k, seed):
    """按类别分层切 k 折，返回每折的测试索引列表。"""
    by_class = {}
    for i, data in enumerate(dataset):
        by_class.setdefault(int(data.y.view(-1)[0]), []).append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for _, idxs in sorted(by_class.items()):
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            folds[j % k].append(idx)
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mini-layers", type=int, default=0)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--gnn", type=str, default="GINEConv")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discrete", action="store_true", help="use atom-type ids with DiscreteEncoder")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    dataset = TUDataset(root=DATA_DIR, name="MUTAG")
    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.subgraph.hops = args.hops
    transform = SubgraphsTransform(cfg)

    t0 = time.time()
    all_data = [transform(dataset[i]) for i in range(len(dataset))]
    if args.discrete:
        for d in all_data:
            d.x = d.x.argmax(dim=1, keepdim=True).long()
    print(f"preprocessing {len(all_data)} graphs: {time.time() - t0:.1f}s")

    folds = stratified_folds(dataset, args.folds, args.seed)
    rows = []
    for fold in range(args.folds):
        test_idx = folds[fold]
        train_idx = [i for j, f in enumerate(folds) if j != fold for i in f]
        rng = random.Random(args.seed + fold)
        rng.shuffle(train_idx)
        n_val = max(10, int(len(train_idx) * 0.1))
        val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
        model = build_model(
            args, None if args.discrete else dataset.num_node_features, dataset.num_classes
        ).to(device)
        val_acc, test_acc = train_fold(
            model,
            [all_data[i] for i in train_idx],
            [all_data[i] for i in val_idx],
            [all_data[i] for i in test_idx],
            args,
            device,
        )
        rows.append((fold, val_acc, test_acc))
        print(f"fold {fold + 1:02d}: val={val_acc:.4f} test={test_acc:.4f}")

    tests = np.array([r[2] for r in rows])
    print(f"10-fold test acc: {tests.mean():.4f} ± {tests.std():.4f}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "mutag_10fold_dgl.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fold", "best_val_acc", "test_acc"])
        writer.writerows(rows)
        writer.writerow(["mean", "", float(tests.mean())])
        writer.writerow(["std", "", float(tests.std())])
    print("results ->", path)


if __name__ == "__main__":
    main()
