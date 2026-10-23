# 第七篇：基于规则的中美科技竞争事件属性挖掘

以下命令从本文件所在目录 `paper7_event_attributes` 执行。

## 代码与报告对应

| 报告部分 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 1.3.1 数据集 | `LLM/prepare_original_full_corpus.py` | `AMR/camr/camr/sentences.txt` → `LLM/data/original_sentences_full.jsonl` |
| 1.2.1 原始方法 | `LLM/llm_information_extraction.py` | 作者提示词、语言示例、属性 schema、JSON 解析；保留原 ChatGLM 演示入口 |
| 1.3.2 本机 LLM 实验 | `LLM/run_original_method_llama_cpp.py` | 每句先分类、再抽取；输出 ID、来源行、原文、分类、原始回复、解析结果 |
| 模型准备 | `LLM/download_qwen35_9b_q8_0.py`、`LLM/start_llama_server.ps1` | 固定版本 Qwen3.5-9B Q8_0 下载校验、本地 llama.cpp 服务 |
| 1.3.3 AMR | `AMR/reproduction/run_amr.sh` | Stanford/BLLIP → CAMR → 格式转换 → JAMR Aligner |
| 数据采集/标注背景 | `legacy/Spider-February-17/`、`legacy/Label-March-27/` | 原附件爬虫、实体/关系/三元组处理及标注词表 |

## 语料准备

原始与清理后的语料均已附带。需要重新构建时：

```powershell
python .\LLM\prepare_original_full_corpus.py
```

清理后保留全部 15,633 条，`attachment-00001` 等 ID 由原行号生成；忽略大小写唯一文本为 15,498 条。程序只统计重复，不去掉这 135 条重复记录。报告后续实验的输入为全部 15,633 条。此数据为英文、无人工真值；本包没有补造中文数据或标签。

## LLM：运行报告的方法

使用本机既有 `pytorch_gpu` 环境即可；新环境的非 PyTorch 依赖见 `LLM/requirements.txt`。另需安装可用的 `llama-server.exe` 并加入 PATH。当前工作区没有报告使用的 Qwen GGUF 文件，重新获取命令为：

```powershell
python .\LLM\download_qwen35_9b_q8_0.py
```

该脚本固定仓库 revision、Q8_0 文件名及 SHA-256，下载约 9.5 GB 文件，并预检查至少 12 GiB 空间。若模型保存在其他位置，可在下面启动命令增加 `-Model '模型完整路径'`。

第一个 PowerShell 窗口启动服务并保持运行：

```powershell
.\LLM\start_llama_server.ps1
```

默认监听 `127.0.0.1:8080`、上下文 4096、GPU 放置全部层、关闭思考模式。第二个窗口从同一目录运行：

```powershell
# 单句检查；再次运行时更换 output 文件名
python .\LLM\run_original_method_llama_cpp.py --limit 1 --output .\LLM\output\check1.jsonl

# 从头生成对应报告规模的前 151 条记录
python .\LLM\run_original_method_llama_cpp.py --first 151 --output .\LLM\output\first151.jsonl --resume

# 处理全部语料；中断后重复执行同一命令按 ID 续跑
python .\LLM\run_original_method_llama_cpp.py --output .\LLM\output\full.jsonl --resume
```

`--first N` 固定输入前 N 条；`--limit N` 只限制本次新增处理条数。每句两次调用，保留原作者 few-shot、分类检查和回复清理逻辑；模型输出不是合法 JSON 时保存原始字符串，不伪造结构化结果。抽样生成和模型环境会影响具体回答，重新运行不保证逐字等于已有 151 条附件。

报告中的两个英语标注示例用于说明输出格式；实际运行提示词以 `llm_information_extraction.py` 中的 `ie_examples` 为准，本次未擅自把报告示例写入提示词。ChatGLM 演示属于旧环境参考，当前运行入口为 llama.cpp 适配脚本。

## AMR 与历史代码

完整准备、构建、前 100 行运行和阶段说明见 [`AMR/README.md`](AMR/README.md)。AMR 使用独立 CPU 容器环境，不依赖上面的模型服务。

历史爬虫及标注代码的依赖、缺失外部配置与用途见 [`legacy/README.md`](legacy/README.md)。这些历史脚本不参与本次固定语料的 LLM/AMR 预测。

结果对应 `../attachments/llm_qwen35_9b_q8_151_records.jsonl` 和 `../attachments/amr_camr_jamr_first100_lines_114_graphs.txt`。本次验证范围见 [`VALIDATION.md`](VALIDATION.md)。
