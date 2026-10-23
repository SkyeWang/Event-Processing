# 第九篇：融合多粒度子结构的高维图表征方法复现

以下命令默认从本目录 `paper9_sagnn_substructure` 执行。

## 代码与报告对应

| 报告部分 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 1.2.1 1-WL 与子结构 | `core/transform.py`、`core/PE_init.py` | 图 → Ego/Cut 子图 + 随机游走位置编码 |
| 1.2.2 双通道注入 | `core/model.py` | 子图编码 + 节点特征 → 图级预测 |
| 1.2.3 算法流程 | `train_mutag.py`、`train_zinc.py` | 图数据 → 训练/评估 |
| 1.3.1 代码结构 | `core/simple_graph.py` | 去 DGL 的图容器 |
| 1.3.2 MUTAG | `train_mutag_cv.py` | MUTAG → 10 折准确率 |
| 1.3.2 ZINC | `download_zinc.py`、`train_zinc.py` | HF 镜像 → ZINC JSONL → MAE |
| 1.3.3 原 DGL 对照 | `dgl_repro/` | 原 DGL 管线 → MUTAG/ZINC |
| 1.3.4 工程适配 | `dgl_repro/PE_init_dgl22.py`、`core/simple_graph.py` | DGL 2.2 接口适配 |

## 运行环境

- PyG 版：Python 3.9，torch 2.6.0+cu124，PyG 2.6.1，SciPy，NetworkX。
- 原 DGL 版：Python 3.10，torch 2.3.1+cu121，DGL 2.2.1，torch_scatter / torch_sparse / torch_cluster。

依赖见 `../requirements-sagnn.txt`。DGL 2.2.1 在 Windows 上需要把 `graphbolt_pytorch_2.3.0.dll`、`dgl_sparse_pytorch_2.3.0.dll` 复制成 `2.3.1`，详见 [`dgl_repro/README.md`](dgl_repro/README.md)。

## 数据与资产

MUTAG 由 PyG 自动下载；ZINC 使用 HuggingFace 镜像 `graphs-datasets/ZINC` 的前 10000/1000/1000 条：

```powershell
python .\download_zinc.py
```

模型权重、MUTAG 处理文件和 checkpoint 不随仓库提交，按 `assets_manifest.json` 从原工作目录复制并校验：

```powershell
python .\prepare_assets.py --component mutag zinc checkpoint
python .\prepare_assets.py --component mutag zinc checkpoint --copy --verify
```

## 运行

MUTAG 单次划分（快速检查）：

```powershell
python .\train_mutag.py --epochs 100 --batch-size 32
```

MUTAG 10 折（与论文协议对应）：

```powershell
python .\train_mutag_cv.py --folds 10 --epochs 100 --hidden 128 --layers 4 --hops 2 --select best_val
```

ZINC（官方 L1 + 450 epochs）：

```powershell
python .\train_zinc.py --epochs 450 --train-limit 10000 --val-limit 1000 --test-limit 1000 --hidden 256 --layers 6 --batch-size 128
```

原 DGL 管线：

```powershell
python .\dgl_repro\run_mutag_dgl.py --folds 10 --epochs 100 --hidden 128 --layers 4 --hops 2
python .\dgl_repro\run_zinc_dgl.py --epochs 100 --hidden 256 --layers 6 --hops 3 --batch-size 128
```

## 本次结果

| 实验 | 结果 | 论文 |
| --- | --- | --- |
| MUTAG 10 折（原 DGL） | 86.29% ± 9.76% | 95.2% ± 3.0% |
| MUTAG 10 折（PyG） | 89.92% ± 9.69% | 95.2% ± 3.0% |
| ZINC（L1 + 450 epochs） | test MAE 0.0843 | 0.072 ± 0.001 |

结果文件：`../attachments/mutag_10fold_dgl.csv`、`../attachments/zinc_sagnn.csv`。与论文的差异、未覆盖的 EXP/CEXP 及后续对齐方案见 [`VALIDATION.md`](VALIDATION.md)。
