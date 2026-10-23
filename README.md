# Event-Processing

组内论文复现仓库。每个成员在独立分支提交自己负责的两篇论文；当前默认分支为 `lzc`（第七、八篇），`fy` 分支对应第九、十篇。

| 分支 | 负责 | 内容 |
| --- | --- | --- |
| `cjn` | 陈嘉宁 | 第一、二篇 |
| `ly` | 李羿 | 第三、四篇 |
| `lq` | 刘启 | 第五、六篇 |
| `lzc` | 李泽诚 | 第七、八篇 |
| `fy` | 方瑜 | 第九、十篇 |

`fy` 分支新增：

- `paper9_sagnn_substructure/`：SAGNN 的 DGL 原管线、PyG 重写版、MUTAG/ZINC 训练脚本。
- `paper10_domain_kg_extraction/`：领域知识抽取的爬虫、标注、LoRA + EWC + 回放训练与 Flask 系统。
- `report9-10_fy.md`：第九、十篇的合并报告。
- `attachments/`：MUTAG/ZINC 与知识抽取的结果文件。
- `source_manifest.json`：源码与附录原文件的 SHA-256 对应关系。

各论文目录下的 `README.md` 给出代码与报告章节的对应表、运行命令和结果；`assets_manifest.json` 列出不随仓库提交的数据与权重，`prepare_assets.py` 负责复制与校验。
