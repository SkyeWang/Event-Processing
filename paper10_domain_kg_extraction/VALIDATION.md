# 验证说明

本目录验证的是知识抽取的训练与 Flask 系统链路。论文的 7B 指标需要作者本地数据、权重和 600 条评估集，本次以 0.5B + 小样本演示为主。

## 已运行

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 样本生成 | `python data/make_sample_data.py` | 3 个批次，每批 6 train / 2 val |
| 小模型训练 | `python training/train_smoke.py --max_steps 3 ...` | train loss 2.6775 / 2.2889 / 2.3740 |
| Flask 接口 | `python flask/test_client.py` | 中文 200，英文 200 |

- 第二批起 EWC 生效，336 个 LoRA 张量参与 Fisher。
- 中文输出示例：`OpenAI—发布—GPT-4.5 模型`、`GPT-4.5 模型—具备—多模态推理能力`。
- 英文输出示例：`NVIDIA—released—Blackwell B200 GPU`。

## 未覆盖

- 论文的 44 万条训练 JSONL、7B LoRA 权重、600 条人工评估集。
- 实体 F1 0.82 / 关系 F1 0.77：缺少金标和 7B 模型，未计算。
- 完整爬虫与 44 万条 API 标注：需要有效 DashScope key 与 MySQL，本次只验证脚本结构和小样本。

## 安全说明

附录代码中的 DashScope key 与 MySQL 密码已改为环境变量：

```powershell
$env:DASHSCOPE_API_KEY = "你的 key"
$env:DEEPSEEK_API_KEY = "你的 key"
$env:MYSQL_PASSWORD = "你的密码"
```

不要将真实 key 写回源码或提交到仓库。
