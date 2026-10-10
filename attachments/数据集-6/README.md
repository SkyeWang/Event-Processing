# 数据集交付说明

由 `收集结果.md` 经 `build_dataset_from_md.py` 自动生成（切分种子 42，可复现）。

- 三元组总条数：163
- CSV 文件数：16（8 领域 × 中/英）
- 编码均为 UTF-8 无 BOM；表头 `entity1,relation,entity2,source_text,source_type`

## 各文件条数

| 文件 | 条数 |
|---|---|
| ch/6G_综合_with_verbs.csv | 14 |
| ch/BCI_综合_with_verbs.csv | 22 |
| ch/具身智能_综合_with_verbs.csv | 13 |
| ch/新能源_综合_with_verbs.csv | 12 |
| ch/氢能核聚变_综合_with_verbs.csv | 11 |
| ch/生物制造_综合_with_verbs.csv | 10 |
| ch/量子科技_综合_with_verbs.csv | 9 |
| ch/集成电路_综合_with_verbs.csv | 11 |
| en/6G_mixed_with_verbs.csv | 8 |
| en/BCI_mixed_with_verbs.csv | 12 |
| en/biomanufacturing_mixed_with_verbs.csv | 7 |
| en/embodied_ai_mixed_with_verbs.csv | 6 |
| en/hydrogen_fusion_mixed_with_verbs.csv | 7 |
| en/IC_mixed_with_verbs.csv | 7 |
| en/new_energy_mixed_with_verbs.csv | 7 |
| en/quantum_mixed_with_verbs.csv | 7 |

## 同步生成的代码用数据文件

- `data/triplets/{ch,en}/all_triplets.jsonl`
- `data/splits/ch/{train,val,test}_triplets.jsonl`（80/10/10）
- `data/splits/en_types/{train,val,test}_triplets.jsonl`（80/10/10）
- `data/{en_}entity_frequencies.json`、`data/{en_}relation_frequencies.json`

## 规模提示

当前为种子数据（每领域几十条），远低于任务书最低验收线（中文 ≥5000 条/领域、英文 ≥2500 条/领域）。扩大规模后重跑 `build_dataset_from_md.py` 即可整体再生成。
