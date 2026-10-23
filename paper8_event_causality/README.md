# 第八篇：中美科技竞争事件间因果关系抽取

以下命令均从本目录 `paper8_event_causality` 执行，使用 Python 3.11。非 PyTorch 依赖见 `ECE_MAIN/requirements.txt`，PyTorch 应沿用本机与 GPU 匹配的版本。

## 代码与报告对应

| 报告部分 | 代码 | 内容 |
| --- | --- | --- |
| 2.2 BERT | `ECE_MAIN/src/model.py`、`layers.py` | BERT、事件类型条件层归一化、原因/结果双网格、标签解码与事件配对 |
| 2.3.2 训练评估 | `ECE_MAIN/run_pipeline.py`、`src/train.py`、`src/data_loader.py`、`src/metric.py`、`src/utils.py`、`src/compat_adamw.py`、`src/log.py` | 批处理、掩码损失、AdamW、warmup、验证集选最佳权重、测试评估 |
| 数据准备 | `ECE_MAIN/make_subset.py`、`check_environment.py` | 按作者原划分取子集，校验规模与 CUDA |
| 2.3.2 英文推理 | `ECE_MAIN/infer_technology_bert.py` | 原 39 类标签前缀、150 字符截断、字符编码、论元还原 |
| 2.3.1 正则 | `ECE_MAIN/infer_tech_regex.py` | 表 2.4 中的连接词、方向与字符位置输出 |
| 2.3.3 去重 | `ECE_MAIN/deduplicate_technology.py` | BERT→正则顺序、逐对词级 TF-IDF、阈值 0.40、保留首次项、合并审计 |
| 辅助查看 | `ECE_MAIN/postprocess_graph.py`、`graph_template.html`、`monitor_training.py`、`monitor_training.html` | 事件图与边表、训练日志/GPU 监控 |
| 原附件参考 | `legacy/` | 原中文/英文规则与未完成的相似度草稿 |

模型基础代码来自本机保存的 `cuishiyao96/ECE` 提交 `59f21d15fcaebc48d798b99090890de14c5af5bb`，提交版补充了中文注释，并为首次运行自动创建日志目录。已有本机适配包括：兼容旧 AdamW、按 dev ECE F1 保存最佳权重、英文解码时用原文字符位置还原论元。标签定义、双网格计算、损失、评估及原始划分保持已有实验实现。

## 训练与模型资源

本目录不重复提交 BERT 权重、训练 pickle 及训练所得 checkpoint。原工作目录存在这些资源时，运行：

```powershell
# 默认只检查；加 --copy 才复制，复制后校验 SHA-256
python .\prepare_assets.py --component ece-data bert checkpoint --copy
python .\ECE_MAIN\check_environment.py --subset full7000 --require-cuda
```

`--source-root '原论文工作目录'` 可指定不同来源。`ece-data` 包括公开 ECE-CCKS 的 5,600/700/700 划分、已编码 pickle、完整 7,000 条子集和 700 条抽样子集；`bert` 为本机 `google-bert/bert-base-chinese` 快照 `8f23c25b06e129b6c986331a13d8d025a92cf0ea`；`checkpoint` 为已有 10/100 轮实验的最佳权重。逐文件大小和摘要见 `assets_manifest.json`。若只训练，可不准备 `checkpoint`；若分发到其他机器，必须另外提供或取得这些资源。

复跑报告采用的 10 轮设置，使用新任务名：

```powershell
python .\ECE_MAIN\run_pipeline.py --task-name report10_new --subset full7000 --epochs 10 --batch-size 8 --lr 0.00005
```

对照的 100 轮设置：

```powershell
python .\ECE_MAIN\run_pipeline.py --task-name report100_new --subset full7000 --epochs 100 --batch-size 8 --lr 0.00005
```

两次均沿用隐藏维 768、dropout 0、warmup 0.1、seed 42、阈值 0.5。10 轮实验的最佳 dev ECE F1 为 47.79%，测试 F1 为 44.53%；100 轮实验的最佳 dev F1 为 46.50%，测试 F1 为 44.91%。报告最终按**验证集**表现选用 10 轮权重。随机训练和硬件差异可能影响复跑数值。

`run_pipeline.py` 原本默认值为毕业论文表格设置（100 轮、batch 6、lr 5e-4），与报告选用设置不同，复跑时请保留上面的显式参数。本机此前发现 lr 5e-4 退化，因此不要省略学习率。训练过程按 dev 分数保存 `best.pickle`，结束后在 test 评估并导出图，输出位于 `ECE_MAIN/models/任务名/`、`log/`、`res/任务名/`。10 轮与 100 轮的 warmup 步数随总训练步数改变。

## 全量英文语料推理与正则

本篇复用第七篇随包附带的 15,633 条英文 JSONL，无需从项目外引用原目录。使用已有 10 轮最佳权重：

```powershell
python .\ECE_MAIN\infer_technology_bert.py `
  --input ..\paper7_event_attributes\LLM\data\original_sentences_full.jsonl `
  --checkpoint .\ECE_MAIN\models\ece_full_author_lr_run1\best.pickle `
  --output .\ECE_MAIN\res\report_new\bert_sentences.jsonl `
  --pairs-output .\ECE_MAIN\res\report_new\bert_pairs.jsonl `
  --summary .\ECE_MAIN\res\report_new\bert_summary.json --device cuda

python .\ECE_MAIN\infer_tech_regex.py `
  --input ..\paper7_event_attributes\LLM\data\original_sentences_full.jsonl `
  --output .\ECE_MAIN\res\report_new\regex_pairs.jsonl

python .\ECE_MAIN\deduplicate_technology.py `
  --bert .\ECE_MAIN\res\report_new\bert_pairs.jsonl `
  --regex .\ECE_MAIN\res\report_new\regex_pairs.jsonl `
  --output-dir .\ECE_MAIN\res\report_new\dedup --threshold 0.40
```

BERT 可加 `--limit 5 --device cpu` 做小规模运行。训练编码要求最多 150 个字符，不能为了英文效果改成英语 WordPiece；每句是否截断会写入输出。当前中文金融标签迁移到英文科技语料的效果有限；这些数据没有人工真值，不能计算英文 F1。

正则结果预期为 5,861 对，BERT 历史结果为 562 对，合计 6,423 对。TF-IDF 将每条事件对写为 `cause + 空格 + effect`，只对两条文本拟合，达到 `>=0.40` 时保留先出现者。论文没有给出序列化细节，该选择已在代码中注明。普通词袋忽略方向，去重也不会判断关系真伪；论文 `result in` 的反常方向按原表保留。

## 不加载模型复跑最终去重

`reference_inputs/` 保存本机报告原始中间预测，分别 562 对和 5,861 对，允许只用 Python 标准库复跑：

```powershell
python .\ECE_MAIN\deduplicate_technology.py `
  --bert .\reference_inputs\bert_pairs.jsonl `
  --regex .\reference_inputs\regex_pairs.jsonl `
  --output-dir .\ECE_MAIN\res\dedup_from_saved --threshold 0.40
```

应得到 4,577 条保留项及 1,846 条合并记录，`retained.jsonl` 与 `../attachments/tech_causal_candidates_dedup_4577.jsonl` 的记录一致；`duplicate_mapping.jsonl` 可追溯合并，`summary.json` 保存数量与设置。

## 辅助入口与原附件

需要小样本训练时，先用 `make_subset.py --name smoke10_new --train 6 --dev 2 --test 2 --seed 42` 生成独立子集，再用 `run_pipeline.py --task-name smoke10_new --subset smoke10_new --epochs 1 --batch-size 6 --lr 0.00005 --allow-cpu` 检查。每条命令都需带 `ECE_MAIN/` 路径。

`monitor_training.py --task-name report10_new --port 8765` 提供只读本机监控页；`postprocess_graph.py prepare/export --help` 列出独立制图选项。生成图的边方向为原因→结果，不代表人工审核通过。

`legacy/causality_extract_chn.py` 依赖旧 pyltp/LTP 模型；英文旧脚本依赖 NLTK 和 `111.txt`，且有导入即执行等历史问题。原 `similarity.py` 有缩进错误、不可达文件处理且计算的是字符词频余弦，因此原样保留为 `similarity_original.py.txt`，正式去重使用已完整实现的 `deduplicate_technology.py`，没有把草稿冒充可运行实验。具体已完成验证见 [`VALIDATION.md`](VALIDATION.md)。
