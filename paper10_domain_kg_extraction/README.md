# 第十篇：面向领域文本的元素子结构持续学习与抽取复现

以下命令默认从本目录 `paper10_domain_kg_extraction` 执行。

## 代码与报告对应

| 报告部分 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 2.2.1 数据构建 | `crawlers/zh/`、`crawlers/en/` | 关键词 → 新闻/论文/视频 CSV |
| 2.2.1 清洗与标注 | `crawlers/zh/百度新闻/clean_data.py`、`annotation/v1/annotate_to_mysql.py` | CSV → 句子 → MySQL labeled |
| 2.2.1 二次标注 | `annotation/v2/annotate_to_mysql_v2.py` | 已有三元组 CSV → annotations/triples 表 |
| 2.2.2 持续学习 | `training/train_smoke.py` | JSONL → LoRA + EWC + 回放训练 |
| 2.3 系统 | `flask/app.py`、`flask/test_client.py` | 文本 → `/extract` 三元组 JSON |

## 运行环境

Python 3.10 + torch 2.6.0+cu126 + transformers 4.40 + peft 0.10 + datasets 3.6 + accelerate 1.7；Flask 3.1。依赖见 `../requirements-kg.txt`。

## 数据与资产

仓库附带 24 条合成样本（`data/train_data`），只用于跑通流程；论文的 44 万条 JSONL、7B LoRA 权重和 600 条评估集按 `assets_manifest.json` 从原工作目录准备：

```powershell
python .\prepare_assets.py --component qwen lora
python .\prepare_assets.py --component qwen lora --copy --verify
```

Qwen2.5-0.5B-Instruct 约 942MB，LoRA 适配器约 8.8MB，均不随仓库提交。

## 运行

生成 24 条样本：

```powershell
python .\data\make_sample_data.py
```

小模型训练（3 个批次、每批 3 步）：

```powershell
python .\training\train_smoke.py --model_path models\Qwen2.5-0.5B-Instruct --data_dir data\train_data --output_dir output\smoke --max_steps 3
```

Flask 接口测试（不启动服务）：

```powershell
python .\flask\test_client.py
```

启动服务：

```powershell
set KG_MODEL_PATH=models\Qwen2.5-0.5B-Instruct
set KG_LORA_PATH=output\smoke\lora_ch_batch_2
set KG_MAX_NEW_TOKENS=256
python .\flask\app.py
```

## 本次结果

| 批次 | 训练样本 | train loss | val loss |
| --- | --- | --- | --- |
| ch/batch_1 | 6 | 2.6775 | 3.6387 |
| ch/batch_2 | 8（含 2 条回放） | 2.2889 | 3.1756 |
| en/batch_1 | 9（含 3 条回放） | 2.3740 | 2.6699 |

Flask `/extract` 中英文均返回 200；中文示例输出“OpenAI—发布—GPT-4.5 模型”“GPT-4.5 模型—具备—多模态推理能力”，英文示例输出“NVIDIA—released—Blackwell B200 GPU”。完整结果见 `../attachments/kg_summary.json`、`../attachments/flask_response.json`。

论文的实体 F1 0.82 / 关系 F1 0.77 对应作者 7B 权重与 600 条评估集，本次以流程复现和 0.5B 演示为主；范围说明见 [`VALIDATION.md`](VALIDATION.md)。
