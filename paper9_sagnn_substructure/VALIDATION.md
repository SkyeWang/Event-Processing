# 验证说明

本目录验证的是 SAGNN 的训练与评估链路，不包含论文未提供的 EXP/CEXP 合成数据。

## 已运行

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| MUTAG 单次划分 | `python train_mutag.py --epochs 100` | best val 94.44%，test 90.00% |
| MUTAG 10 折（PyG） | `python train_mutag_cv.py --folds 10 ...` | 89.92% ± 9.69% |
| MUTAG 10 折（原 DGL） | `python dgl_repro/run_mutag_dgl.py --folds 10 ...` | 86.29% ± 9.76% |
| ZINC（L1 + 450 epochs） | `python train_zinc.py --epochs 450 ...` | test MAE 0.0843，best val 0.1020 |
| ZINC（原 DGL，MSE 100） | `python dgl_repro/run_zinc_dgl.py --epochs 100 ...` | test MAE 0.1413 |

## 未覆盖

- EXP/CEXP：原 `create_dataset_EXP()` 实际加载 ZINC，附录不含 EXP 数据。
- 官方 ZINC 10k 子集索引：本次用 HF 镜像前 10k，结果与论文 0.072 接近但非逐位对齐。
- 多 seed / 10 次运行平均：当前 MUTAG 10 折为单次分层划分，ZINC 为单次训练。

## 复跑提示

- MUTAG 数据由 PyG 自动下载；ZINC 先运行 `download_zinc.py`。
- 原 DGL 管线需要 torch 2.3 + DGL 2.2.1，并按 `dgl_repro/README.md` 处理 DLL 与 PE 接口。
- `assets_manifest.json` 只列不随仓库提交的数据/权重；`prepare_assets.py` 默认只检查，加 `--copy` 才复制。
