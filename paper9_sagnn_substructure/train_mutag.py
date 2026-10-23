"""在 MUTAG 上训练 SAGNN（去 DGL 的 PyG 预处理版本）。

流程：
1. 用 PyG TUDataset 读取 MUTAG（188 张图，2 类，7 维节点特征）；
2. 按 8:1:1 随机划分 train/val/test；
3. 用 core.transform.SubgraphsTransform 提取 Ego/Cut 子图并计算 RWPE；
4. 用 SAGNN 训练，逐 epoch 记录 train/val/test 指标；
5. 结果写入 results/mutag_sagnn_seed0.csv。

注意：这里只是单次划分的快速验证；论文的 MUTAG 95.2 是 10 折结果，
对应脚本是 train_mutag_cv.py，不要拿本脚本的单次 test 直接对比论文。

示例：
    python train_mutag.py --epochs 100 --batch-size 32
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
DATA_DIR = os.path.join(ROOT, "data", "MUTAG")
RESULT_DIR = os.path.join(ROOT, "results")


def set_seed(seed):
    # 固定 Python / NumPy / PyTorch 的随机种子，保证单次实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, seed=0, ratios=(0.8, 0.1, 0.1)):
    # 简单随机划分：MUTAG 只有 188 张图，单次划分的 test 只有 20 张，
    # 波动很大，因此正式对比应使用 train_mutag_cv.py 的 10 折。
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def build_model(args, nfeat_node, nout):
    # MUTAG 的 x 是 7 维 one-hot，用 MLP 编码；ZINC 的原子类型是 long，
    # 对应脚本 train_zinc.py 会传 nfeat_node=None，走 DiscreteEncoder。
    model = SAGNN(
        nfeat_node,
        None,
        nhid=args.hidden,
        nout=nout,
        nlayer_outer=args.layers,
        nlayer_inner=0,
        gnn_types=[args.gnn],
        embedding_types="rand_walk",
        hop_dim=16,
        fullgraph_pos_enc_dim=20,
        egograph_pos_enc_dim=16,
        cutgraph_pos_enc_dim=16,
        pos_enc_dim=24,
        pooling="add",
        global_embedding=True,
        dropout=args.dropout,
    )
    model.reset_parameters()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定 loader 上算分类 loss 和准确率（不反传）。"""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y = batch.y.view(-1)
        loss_sum += float(F.cross_entropy(out, y)) * y.numel()
        correct += int((out.argmax(dim=-1) == y).sum())
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    """MUTAG 单次划分训练入口：读数据、预处理、训练并记录逐 epoch 指标。"""
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--limit", type=int, default=0, help="use only N graphs (debug)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    print(f"device={device}")

    dataset = TUDataset(root=DATA_DIR, name="MUTAG")
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"MUTAG graphs={len(dataset)} classes={dataset.num_classes} node_features={dataset.num_node_features}")

    train_idx, val_idx, test_idx = split_indices(len(dataset), seed=args.seed)
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.subgraph.hops = args.hops
    cfg.model.gnn_type = args.gnn
    # 三个 split 共用同一个 transform：Ego/Cut 子图、hop indicator、RWPE
    # 都在这里一次性算好，训练时只做前向和反向。
    transform = SubgraphsTransform(cfg)

    t0 = time.time()
    train_set = [transform(dataset[i]) for i in train_idx]
    val_set = [transform(dataset[i]) for i in val_idx]
    test_set = [transform(dataset[i]) for i in test_idx]
    print(f"preprocessing took {time.time() - t0:.1f}s")

    # 训练集打乱，验证/测试不打乱；batch 内由 SubgraphsData.__inc__ 处理子图索引偏移
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # MUTAG 是图分类任务：输出 2 类，损失用交叉熵
    model = build_model(args, dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f"mutag_sagnn_seed{args.seed}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "time_s"])

        best_val = 0.0
        best_test = 0.0
        for epoch in range(1, args.epochs + 1):
            # ---- 训练一个 epoch ----
            model.train()
            total = correct = 0
            loss_sum = 0.0
            t_epoch = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch)
                y = batch.y.view(-1)
                loss = F.cross_entropy(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss) * y.numel()
                correct += int((out.argmax(dim=-1) == y).sum())
                total += y.numel()
            scheduler.step()

            # ---- 验证/测试；按验证集最优保存权重 ----
            train_loss = loss_sum / max(total, 1)
            train_acc = correct / max(total, 1)
            val_loss, val_acc = evaluate(model, val_loader, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
                torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"mutag_best_seed{args.seed}.pt"))

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, time.time() - t_epoch])
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

    print(f"best val_acc={best_val:.4f} test_acc_at_best_val={best_test:.4f}")
    print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()
