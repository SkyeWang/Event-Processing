# 原 DGL 管线复现（Windows + torch 2.3）

`sagnn/` 根目录是去 DGL 的 PyG 版本；本目录保留**原作者的 DGL 实现**，用于验证
“换掉 DGL 是否影响结果”。结论：MUTAG 上 DGL 86.29%、PyG 89.92%，差异在划分/训练随机性
范围内；ZINC 的主要差距来自损失函数和训练轮数，不是 DGL。

## 环境

- conda 环境：`unifolm-wma`（Python 3.10，torch 2.3.1+cu121，CUDA 可用）
- DGL 2.2.1（Windows 最后一个 wheel，官方只支持 torch ≤2.3）
- PyG 2.6.1（纯 Python 侧载）
- torch_scatter 2.1.2 / torch_sparse 0.6.18 / torch_cluster 1.6.3
  （`+pt23cu121`、cp310、win_amd64 wheel）

DGL 2.2.1 的 wheel 按 torch 2.3.0 编译，torch 2.3.1 下需要复制 DLL：

```text
dgl/graphbolt/graphbolt_pytorch_2.3.0.dll  -> graphbolt_pytorch_2.3.1.dll
dgl/dgl_sparse/dgl_sparse_pytorch_2.3.0.dll -> dgl_sparse_pytorch_2.3.1.dll
```

然后设置：

```powershell
$env:PYTHONPATH="<pylibs_dgl310>;<原 SAGNN 代码目录>"
$env:DGLDEFAULTDIR="<workspace>\data\dgl310"
$env:DGLBACKEND="pytorch"
```

## DGL 2.2 API 适配

原 `core/PE_init.py` 使用 `adjacency_matrix(scipy_fmt="csr")`、
`adjacency_matrix_scipy()`、`dgl.backend.asnumpy()`，DGL 2.2 已移除/改名。
`PE_init_dgl22.py` 是等价补丁：`adjacency_matrix().to_dense()` 转 SciPy，
`in_degrees().cpu().numpy()` 取度。把它覆盖到原代码的 `core/PE_init.py` 即可。

## 运行

把原附录 `sagnn_createdataset` 放到 `<原 SAGNN 代码目录>`，覆盖 `core/PE_init.py`，然后：

```bash
python run_mutag_dgl.py --folds 10 --epochs 100 --hidden 128 --layers 4 --hops 2
python run_zinc_dgl.py --epochs 100 --hidden 256 --layers 6 --hops 3 --batch-size 128
```

`run_zinc_dgl.py` 默认用 MSE（与原作者 `main.py` 一致）；官方 `train/zinc.py` 用 L1 +
450 epochs，仓库根目录的 `train_zinc.py` 已按官方设置改为 L1/450，并得到 test MAE 0.0843。

## 结果

| 实验 | 结果 | 文件 |
|---|---|---|
| MUTAG 10 折（原 DGL） | 86.29% ± 9.76% | `results/mutag_10fold_dgl.csv` |
| ZINC（原 DGL，MSE 100 epochs） | best val 0.1660 / test 0.1413 | `results/zinc_dgl.csv` |
| ZINC（官方 L1，450 epochs，PyG） | best val 0.1020 / test 0.0843 | `../results/zinc_sagnn.csv` |

论文：MUTAG 95.2±3.0，ZINC 0.072±0.001。ZINC 已接近（1.17 倍），MUTAG 仍差 6–9 个点。
