"""用原 DGL 管线在 ZINC 子集上做图回归。

与原 PyG 版 ``train_zinc.py`` 的区别：保留 DGL 的图容器和 ``core.transform``，
用于对照“DGL vs PyG”对 ZINC MAE 的影响。注意本脚本默认用 MSE（与原作者
``main.py`` 一致）；官方 ``train/zinc.py`` 用 L1 + 450 epochs，结果见
``train_zinc.py``。

DATA_DIR 默认指向 ``../ZINC_hf``，换机器时按实际数据目录修改。

示例：
    python run_zinc_dgl.py --epochs 100 --hidden 256 --layers 6 --hops 3
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
DATA_DIR = os.path.join(ROOT, "..", "ZINC_hf")
RESULT_DIR = os.path.join(ROOT, "results_dgl")


def load_split(path, limit=0):
    """读取 JSONL，构造 PyG Data；x/edge_attr 保持 long，供 DiscreteEncoder 使用。"""
    graphs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit and len(graphs) >= limit:
                break
            row = json.loads(line)
            graphs.append(
                Data(
                    x=torch.tensor(row["node_feat"], dtype=torch.long).view(-1, 1),
                    edge_index=torch.tensor(row["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(row["edge_attr"], dtype=torch.long).view(-1),
                    y=torch.tensor(row["y"], dtype=torch.float).view(-1),
                )
            )
    return graphs


def build_model(args):
    """构造 SAGNN 回归模型：nout=1，原子/键类型走 DiscreteEncoder。"""
    model = SAGNN(
        None,
        None,
        nhid=args.hidden,
        nout=1,
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
    """在给定 loader 上算 MAE（平均绝对误差）。"""
    model.eval()
    total = 0
    mae = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        y = batch.y.view(-1)
        mae += float((pred - y).abs().sum())
        total += y.numel()
    return mae / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--mini-layers", type=int, default=1)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--gnn", type=str, default="GINEConv")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.subgraph.hops = args.hops
    transform = SubgraphsTransform(cfg)

    t0 = time.time()
    train_raw = load_split(os.path.join(DATA_DIR, "train.jsonl"), args.limit)
    val_raw = load_split(os.path.join(DATA_DIR, "val.jsonl"), args.limit)
    test_raw = load_split(os.path.join(DATA_DIR, "test.jsonl"), args.limit)
    print(f"raw train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")
    train_set = [transform(g) for g in train_raw]
    val_set = [transform(g) for g in val_raw]
    test_set = [transform(g) for g in test_raw]
    print(f"preprocessing {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "zinc_dgl.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_mae", "val_mae", "test_mae", "time_s"])
        best_val = best_test = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0
            loss_sum = 0.0
            mae_sum = 0.0
            t_epoch = time.time()
            for batch in train_loader:
                batch = batch.to(device)
                pred = model(batch).view(-1)
                y = batch.y.view(-1)
                loss = F.mse_loss(pred, y)
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
            writer.writerow([epoch, train_loss, train_mae, val_mae, test_mae, time.time() - t_epoch])
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:03d} train_mae={train_mae:.4f} val_mae={val_mae:.4f} test_mae={test_mae:.4f}")
    print(f"best val MAE={best_val:.4f}, test MAE at best val={best_test:.4f}")
    print("results ->", path)


if __name__ == "__main__":
    main()
