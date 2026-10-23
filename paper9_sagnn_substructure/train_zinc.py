"""在 ZINC 子集上训练 SAGNN（图回归，评价指标 MAE）。

数据来源：官方 PyG 的 ZINC 地址指向 Dropbox，本机不可达；download_zinc.py
从 HuggingFace 镜像 graphs-datasets/ZINC 流式取 JSONL，字段与 PyG 版本一致
（node_feat、edge_index、edge_attr、y、num_nodes）。

训练设置与官方 train/zinc.py 对齐：
- 6 层、hidden=256、GINEConv、hops=3、batch=128；
- 原子类型/键类型用 long 编码，走 DiscreteEncoder；
- L1 损失（不是 MSE），450 epochs，StepLR 每 50 轮衰减。

示例：
    python download_zinc.py
    python train_zinc.py --epochs 450 --train-limit 10000 --val-limit 1000 --test-limit 1000
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from core.config import cfg
from core.model import SAGNN
from core.transform import SubgraphsTransform

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "ZINC_hf")
RESULT_DIR = os.path.join(ROOT, "results")


def load_split(path, limit=0):
    """读取 JSONL 并构造 PyG Data 列表。

    x 和 edge_attr 保持 long：原子类型/键类型是类别值，交给 DiscreteEncoder。
    """
    graphs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit and len(graphs) >= limit:
                break
            row = json.loads(line)
            x = torch.tensor(row["node_feat"], dtype=torch.long).view(-1, 1)
            edge_index = torch.tensor(row["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(row["edge_attr"], dtype=torch.long).view(-1)
            y = torch.tensor(row["y"], dtype=torch.float).view(-1)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


def build_model(args):
    """构造 SAGNN 回归模型：nout=1，原子/键类型走 DiscreteEncoder。"""
    model = SAGNN(
        None,
        None,
        nhid=args.hidden,
        nout=1,
        nlayer_outer=args.layers,
        nlayer_inner=1,
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
    """在给定 loader 上算 MAE。"""
    model.eval()
    mae_sum = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        y = batch.y.view(-1)
        mae_sum += float((pred - y).abs().sum())
        total += y.numel()
    return mae_sum / max(total, 1)


def main():
    """ZINC 训练入口：读数据、预处理、逐 epoch 训练并记录 MAE。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--gnn", type=str, default="GINEConv")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0, help="limit all splits (debug)")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(0)
    print(f"device={device}")

    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.subgraph.hops = args.hops
    cfg.model.gnn_type = args.gnn
    transform = SubgraphsTransform(cfg)

    t0 = time.time()
    train_raw = load_split(os.path.join(DATA_DIR, "train.jsonl"), args.train_limit or args.limit)
    val_raw = load_split(os.path.join(DATA_DIR, "val.jsonl"), args.val_limit or args.limit)
    test_raw = load_split(os.path.join(DATA_DIR, "test.jsonl"), args.test_limit or args.limit)
    print(f"raw: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")

    # 预计算子图与位置编码：12k 张图约 4 分钟，训练阶段只做前向/反向
    train_set = [transform(g) for g in train_raw]
    val_set = [transform(g) for g in val_raw]
    test_set = [transform(g) for g in test_raw]
    print(f"preprocessing took {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, "zinc_sagnn.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_mae", "val_mae", "test_mae", "time_s"])
        best_val = float("inf")
        best_test = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            mae_sum = 0.0
            total = 0
            t_epoch = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                y = batch.y.view(-1)
                # 官方 train/zinc.py 用 L1 损失；这是 ZINC 能到 0.08 量级的关键
                loss = (pred - y).abs().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss) * y.numel()
                mae_sum += float((pred - y).abs().sum())
                total += y.numel()
            scheduler.step()
            train_loss = loss_sum / max(total, 1)
            train_mae = mae_sum / max(total, 1)
            val_mae = evaluate(model, val_loader, device)
            test_mae = evaluate(model, test_loader, device)
            if val_mae < best_val:
                best_val, best_test = val_mae, test_mae
                torch.save(model.state_dict(), os.path.join(RESULT_DIR, "zinc_best.pt"))
            writer.writerow([epoch, train_loss, train_mae, val_mae, test_mae, time.time() - t_epoch])
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:03d} train_mae={train_mae:.4f} val_mae={val_mae:.4f} test_mae={test_mae:.4f}")

    print(f"best val MAE={best_val:.4f}, test MAE at best val={best_test:.4f}")
    print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()

