# CAMR → JAMR 实验

## 源码

- `camr/camr/`：CAMR 原始 Python 2 解析、图状态、特征、预处理和训练/预测脚本，提交 `6431b0bc3cd588d8a7a4cc3f4138ff1c331e8a47`。
- `jamr/jamr/`：JAMR 的 Scala Aligner、图与对齐代码、构建文件和说明，提交 `ddfbe318dea39e746c5e6cf559d5a286cb0211f0`。
- `reproduction/`：本机实验的 Dockerfile、仓库配置、下载脚本及四阶段编排。`run_amr.sh` 中有中文阶段注释。
- `README_original.md`：原附件的 CAMR→JAMR 操作顺序。各项目原许可证随源码保留。

不包含嵌套 `.git`。`source_layout.json` 保存原 Unix 执行位及软链接；提交版 `run_amr.sh` 复制随包源码并恢复这些信息，不再依赖从原目录 Git clone。解析模型和算法没有更改。

## 准备与运行

以下命令从上级 `paper7_event_attributes` 目录执行。需 Docker Desktop 处于 Linux 容器模式。大文件资源仍在原工作目录时，先按清单复制并校验（约数百 MB）：

```powershell
python .\prepare_assets.py --component amr --copy
```

`--source-root '原论文工作目录'` 可指定不同来源；默认是 `submission` 上一级。清单包含原 CAMR 权重、原资源词典、CoreNLP JAR、Stanford 2015 ZIP、BLLIP 模型、WordNet、Miniconda2 安装包和 sbt 启动器。单独分发代码时，需另提供这些资源；每项的来源位置、大小、SHA-256 见 `../assets_manifest.json`，不能用名称相似的新版本直接代替旧环境资源。

已有 `paper1-amr:local` 镜像可直接运行；新建镜像时，先把可用的 Ubuntu 24.04 镜像标记为 Dockerfile 使用的本地基础镜像，再构建：

```powershell
docker pull ubuntu:24.04
docker tag ubuntu:24.04 amr-ubuntu-base:local
docker build -t paper1-amr:local .\AMR\reproduction
```

以独立工作卷复跑报告的前 100 行：

```powershell
$amrRoot = (Resolve-Path .\AMR).Path
$resultRoot = Join-Path $amrRoot 'reproduction\results'
New-Item -ItemType Directory -Force $resultRoot | Out-Null
docker run --rm --name submission-paper7-amr-first100 `
  --mount "type=bind,source=$amrRoot,target=/source,readonly" `
  --mount "type=bind,source=$resultRoot,target=/results" `
  --mount 'type=volume,source=submission-paper7-amr-work,target=/work' `
  paper1-amr:local bash /source/reproduction/run_amr.sh full --limit 100
```

首行检查可把末尾改为 `run_amr.sh smoke`；全量可改为 `run_amr.sh full`。默认源语料为 `camr/camr/sentences.txt` 的原始 15,633 行。使用新的工作卷名可隔离不同版本源码和中间缓存。

## 四阶段与结果

1. CoreNLP 分词、词性、命名实体与 BLLIP/Charniak 句法分析，生成 `.prp`、`.tok`、依赖文件。
2. CAMR 加载作者模型，生成 `.parsed` AMR 图。
3. CAMR 的 `--amrfmt amr` 预处理，生成 `.amr.tok`。
4. JAMR `Aligner --print-nodes-and-edges`，输出节点、边和词语对齐区间。

前 100 行的结果路径为 `reproduction/results/first100_output_file`，详细阶段日志在 `results/logs/`。一行可能被 Stanford 拆为多个句子，已有报告因此为 114 个图。程序要求 CAMR 图数、JAMR 图数、对齐数三者一致且非零；成功结束并打印 `Complete` 才代表完成。

CAMR/JAMR 为旧版 Python 2 / Scala 代码，不能用 Python 3 的编译检查代替运行验证。本次整理未重新执行 AMR 容器，见上级 `VALIDATION.md`。
