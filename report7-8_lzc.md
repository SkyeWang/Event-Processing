# 1. 第七篇：基于规则的中美科技竞争事件属性挖掘

## 1.1 简介

论文数据集通过中英文关键词爬取静态、动态网页的中美科技相关文章，按句处理，用程序标注实体集并回标相关属性，并通过大预言模型和 AMR 两种方法分别对中英文和英文语料进行数据挖掘。

## 1.2 中英文不同属性挖掘方法介绍

### 1.2.1 方法一：大语言模型

- **先用模型判断语言**：代码把一个中文例句和一个英文例句放进固定的对话历史，让模型回答待处理句属于“中文”还是“英文“。

- **再做属性挖掘**：通过对话历史中的 few shot 和输出规则，让 LLM 对每一句话输出一个结构化 json 文件。对于中文句子输出字段包括实体、关系、时间、国家；对于英文句子再加上形容词、副词。

### 1.2.2 方法二：AMR

- **预处理句子**：用 Stanford CoreNLP 做分词、词性和命名实体处理，再用 Charniak 解析句法依赖关系。
- **CAMR 生成 AMR 图**：加载作者提供的预训练模型，把句子解析成概念节点和关系边。
- **JAMR 做对齐**：先将 CAMR 的图转成 JAMR 所需的输入格式，再运行 Aligner，标出原句词语对应图中的哪些节点，并输出节点、边及对齐位置。

## 1.3 代码实践及运行效果

### 1.3.1 数据集构建

在项目文件 `AMR/camr/camr/sentences.txt` 中直接给出了英语原始句子共 15633 条，完成忽略大小写去重后共 15498 条，经过脚本转换成格式化 jsonl 文件后作为全量数据集。由于项目文件中没有给出中文句子，中文部分实验也复用这个全量数据集。

此外人工标注了两份英语 few shot 如下：

```json
[
  {
    "text": "AI will provide the computing power of the future, while quantum communication will be a new tool.",
    "language": "英文",
    "attributes": {
      "实体": ["AI", "computing power", "quantum communication", "tool"],
      "关系": ["provide"],
      "形容词": ["computing", "new"],
      "副词": ["原文中未提及"],
      "时间": ["future"],
      "国家": ["原文中未提及"]
    }
  },
  {
    "text": "US export controls restrict advanced computing chips and semiconductor manufacturing equipment shipped to China.",
    "language": "英文",
    "attributes": {
      "实体": ["export controls", "computing chips", "semiconductor manufacturing equipment"],
      "关系": ["restrict", "shipped"],
      "形容词": ["advanced", "computing", "semiconductor", "manufacturing"],
      "副词": ["原文中未提及"],
      "时间": ["原文中未提及"],
      "国家": ["US", "China"]
    }
  }
]
```

### 1.3.2 方法一：大语言模型

由于 ChatGLM 6b 模型已经过时，本实验在复现时通过 llama.cpp 运行 Qwen 3.5 9b 模型，8 bit 量化版本，并关闭思考模式。

已保存的 151 条完整预测记录见： [LLM 实验输出文件](attachments/llm_qwen35_9b_q8_151_records.jsonl)。

输出示例：

```json
[
  {
    "id": "attachment-00017",
    "source_line": 17,
    "text": "For instance, the US put Huawei on its Entity List, restricting any foreign semiconductor company from selling chips developed or produced using US technologies or software to Huawei without first obtaining a license, which is essentially a blockade.",
    "class_response": "英文",
    "parsed": {
      "实体": [
        "Huawei",
        "Entity List",
        "semiconductor company",
        "chips",
        "technologies",
        "software",
        "license",
        "blockade"
      ],
      "关系": [
        "put ... on",
        "restricting",
        "developed",
        "produced",
        "selling",
        "obtaining"
      ],
      "形容词": [
        "foreign",
        "first",
        "essentially"
      ],
      "副词": [
        "For instance",
        "without"
      ],
      "时间": ["原文中未提及"],
      "国家": ["US"]
    }
  },
  {
    "id": "attachment-00122",
    "source_line": 122,
    "text": "Huawei's Plan B also boosted the domestic semiconductor industry, which still lags behind foreign competitors in core technologies and relies heavily on imported chipsets and components.",
    "class_response": "英文",
    "parsed": {
      "实体": [
        "Huawei",
        "Plan B",
        "domestic semiconductor industry",
        "foreign competitors",
        "core technologies",
        "imported chipsets",
        "components"
      ],
      "关系": [
        "boosted",
        "lags behind",
        "relies on"
      ],
      "形容词": [
        "domestic",
        "core",
        "foreign",
        "imported"
      ],
      "副词": ["原文中未提及"],
      "时间": ["原文中未提及"],
      "国家": ["原文中未提及"]
    }
  }
]
```

### 1.3.3 方法二：AMR

本次对原始语料前 100 行运行 CAMR 解析与 JAMR 对齐，得到 114 个 AMR 图及 114 条对齐记录。以下从本轮原始输出中选取两条。

全部图和词语对齐记录见： [AMR 实验完整输出文件](attachments/amr_camr_jamr_first100_lines_114_graphs.txt)。

**示例一**：图中 `look-02` 的 `:ARG0` 指向 China，`:ARG1` 指向 `compete-01`；后者连接 US、下一轮竞争和 6G。

```text
# ::id 32
# ::snt China looks forward to competing with the US in the next round of competition: 6G.
# ::tok China looks forward to competing with the US in the next round of competition : 6G .
# ::alignments 0-1|0.0.0+0.0.0.0+0.0.0.0.0 7-8|0.0.2.0+0.0.2.0.0+0.0.2.0.0.0 1-2|0.0 4-5|0.0.2 15-16|0.0.2.2 11-12|0.0.2.1 10-11|0.0.2.1.0 2-3|0.0.1 ::annotator Aligner v.03 ::date 2026-09-22T08:05:59.321
# ::node	0	infer-01	
# ::node	0.0	look-02	1-2
# ::node	0.0.0	country	0-1
# ::node	0.0.0.0	name	0-1
# ::node	0.0.0.0.0	"China"	0-1
# ::node	0.0.1	forward	2-3
# ::node	0.0.2	compete-01	4-5
# ::node	0.0.2.0	country	7-8
# ::node	0.0.2.0.0	name	7-8
# ::node	0.0.2.0.0.0	"US"	7-8
# ::node	0.0.2.1	round	11-12
# ::node	0.0.2.1.0	next	10-11
# ::node	0.0.2.1.1	compete-01	
# ::node	0.0.2.2	6g	15-16
# ::root	0	infer-01
# ::edge	compete-01	ARG2	6g	0.0.2	0.0.2.2	
# ::edge	compete-01	ARG2	country	0.0.2	0.0.2.0	
# ::edge	compete-01	ARG2	round	0.0.2	0.0.2.1	
# ::edge	country	name	name	0.0.0	0.0.0.0	
# ::edge	country	name	name	0.0.2.0	0.0.2.0.0	
# ::edge	infer-01	ARG1	look-02	0	0.0	
# ::edge	look-02	ARG0	country	0.0	0.0.0	
# ::edge	look-02	ARG1	compete-01	0.0	0.0.2	
# ::edge	look-02	direction	forward	0.0	0.0.1	
# ::edge	name	op1	"China"	0.0.0.0	0.0.0.0.0	
# ::edge	name	op1	"US"	0.0.2.0.0	0.0.2.0.0.0	
# ::edge	round	mod	compete-01	0.0.2.1	0.0.2.1.1	
# ::edge	round	mod	next	0.0.2.1	0.0.2.1.0	
(xap0 / infer-01
	:ARG1 (x2 / look-02
		:ARG0 (x1 / country
			:name (n / name
				:op1 "China"))
		:direction (x3 / forward)
		:ARG1 (x5 / compete-01
			:ARG2 (x8 / country
				:name (n1 / name
					:op1 "US"))
			:ARG2 (x12 / round
				:mod (x11 / next)
				:mod (x14 / compete-01))
			:ARG2 (x16 / 6g))))
```

**示例二**：图中 `commend-01` 的 `:ARG0` 指向 China，`assist-01` 和 `cooperate-01` 通过 `and` 相连，相关节点指向 US 和 priority cases。

```text
# ::id 72
# ::snt China commended the US assistance and cooperation on priority cases.
# ::tok China commended the US assistance and cooperation on priority cases .
# ::alignments 0-1|0.0+0.0.0+0.0.0.0 3-4|0.1.0.0+0.1.0.0.0+0.1.0.0.0.0 1-2|0 5-6|0.1 9-10|0.1.1.0 8-9|0.1.1.0.0 6-7|0.1.1 4-5|0.1.0 ::annotator Aligner v.03 ::date 2026-09-22T08:05:59.932
# ::node	0	commend-01	1-2
# ::node	0.0	country	0-1
# ::node	0.0.0	name	0-1
# ::node	0.0.0.0	"China"	0-1
# ::node	0.1	and	5-6
# ::node	0.1.0	assist-01	4-5
# ::node	0.1.0.0	country	3-4
# ::node	0.1.0.0.0	name	3-4
# ::node	0.1.0.0.0.0	"US"	3-4
# ::node	0.1.1	cooperate-01	6-7
# ::node	0.1.1.0	case	9-10
# ::node	0.1.1.0.0	priority	8-9
# ::root	0	commend-01
# ::edge	and	op1	assist-01	0.1	0.1.0	
# ::edge	and	op2	cooperate-01	0.1	0.1.1	
# ::edge	assist-01	ARG0	country	0.1.0	0.1.0.0	
# ::edge	case	mod	priority	0.1.1.0	0.1.1.0.0	
# ::edge	commend-01	ARG0	country	0	0.0	
# ::edge	commend-01	ARG1	and	0	0.1	
# ::edge	cooperate-01	ARG0	country	0.1.1	0.1.0.0	
# ::edge	cooperate-01	ARG2	case	0.1.1	0.1.1.0	
# ::edge	country	name	name	0.0	0.0.0	
# ::edge	country	name	name	0.1.0.0	0.1.0.0.0	
# ::edge	name	op1	"China"	0.0.0	0.0.0.0	
# ::edge	name	op1	"US"	0.1.0.0.0	0.1.0.0.0.0	
(x2 / commend-01
	:ARG0 (x1 / country
		:name (n / name
			:op1 "China"))
	:ARG1 (x6 / and
		:op1 (x5 / assist-01
			:ARG0 (x4 / country
				:name (n1 / name
					:op1 "US")))
		:op2 (x7 / cooperate-01
			:ARG0 x4
			:ARG2 (x10 / case
				:mod (x9 / priority)))))
```

## 1.4 小结

通过本论文了解了属性挖掘任务的做法，主要学习了 few shot + 规则的大语言模型方法和使用 CAMR 构建 AMR 图，再用 JAMR 将图中节点与原文词语对齐的 AMR 算法流程，并在小规模数据集上复现了论文实验。

## 1.5 参考文献

[1] Wang Z, Wang X, Han X, et al. CLEVE: Contrastive Pre-training for Event Extraction. 2021.

[2] Wang C, Pradhan S, Pan X, et al. CAMR at SemEval-2016 Task 8: An Extended Transition-based AMR Parser. Proceedings of SemEval, 2016: 1173–1178.

[3] Wang C, Xue N, Pradhan S. Boosting Transition-based AMR Parsing with Refined Actions and Auxiliary Analyzers. Proceedings of ACL-IJCNLP, 2015: 857–862.

---

# 2. 第八篇：中美科技竞争事件间因果关系抽取

## 2.1 简介

论文基于通过关键词爬取的有关中美科技竞争的数据集，通过开源双网格数据集 ECE-CCKS 训练 BERT 模型，并结合了正则表达式的提取方法，对 BERT 推理和正则两路结果进行去重和人工审核，实现了科技竞争事件间因果关系的提取。

## 2.2 方法介绍

### 2.3.1 数据集构建

- 根据有关中美科技竞争事件的关键词通过爬虫从网络获取文本
- 通过正则中英文中表示因果的联系词、词组或表达，从文本中抽取出三个部分：“事件原因”、“事件结果”和“因果类型”
- 使用 Event Causality Extraction with Event Argument Correlations （CCKS2021）的开源双网格数据集 ECE-CCKS 作为BERT的训练集
- 使用前一篇论文的英文原始句子构成的结构化数据集进行因果关系提取

### 2.3.2 BERT

- 输入文本和候选事件类型到 BERT，获得文字向量和事件类型向量
- 分别对于原因和结果两张表通过 CLN （条件层归一化）融合文本特征和类型信息，再通过分类器（全连接层 + sigmoid）预测各网格位置的多标签概率
- 对照已标注的标准答案计算 loss，loss 为两张表损失之和
- 反向传播

### 2.3.3 结果处理

论文并没有给出错误抽取与无关事件处理的具体做法。

去重方面，先将事件文本转为 TF-IDF 向量，再计算两两之间的余弦相似度，以 0.4 为阈值进行筛选（ \(\operatorname{sim}(v_a,v_b)\geq 0.40\) 时合并，保留先出现的事件对）。

## 2.3 代码实践及运行效果

### 2.3.1 正则

实验根据论文表 2.4 的英文关键词及原因、结果方向构建正则脚本，对 15633 条原始英文文本抽取出 5861 对候选因果关系，涉及 4788 条文本。

### 2.3.2 BERT

实验使用 ECE-CCKS 全量数据集（训练/验证/测试为 5,600/700/700）训练 BERT base chinese 双网格模型，完成 10 轮训练。使用的超参数配置如下：

| 参数 | 数值 |
| --- | ---: |
| 训练轮数 | 10 |
| Batch size | 8 |
| 学习率 | 5e-5 |
| 隐藏维度 | 768 |
| Dropout | 0 |
| Warmup 比例 | 0.1 |
| 随机种子 | 42 |
| 推理阈值 | 0.5 |

第 10 轮的验证集 ECE F1 最高，为 47.79%；选取该轮权重在测试集上得到 ECE Precision 45.13%、Recall 43.94%、F1 44.53%。

此后也进行了 100 epoch 实验，但最高 F1 并没有超过 10 epoch 的第十轮，故最终选用 10 epoch 的第十轮权重进行推理。

用该权重对上述英文语料推理，得到 562 对候选关系，涉及 452 条文本。但实际效果不佳。

### 2.3.3 去重

将两路共 6423 对候选关系按论文 5.3 节的方法做逐对 TF-IDF 余弦相似度比较，达到 0.40 时保留先出现的一条，最终保留 4577 对，共 1,846 次合并。

### 2.3.4 实验结果示例

以下选取去重结果中较符合原句含义的示例：

| 原因 | 结果 |
| --- | --- |
| `China has improved its competitiveness` | `China is not afraid of US pressure` |
| `my old Huawei P9 became slow` | `I bought the new one` |
| `distribution channels have been built straight from there` | `Otigba is at the top of the technology equipment and consumables supply chain` |

完整实验结果见：[完整实验结果文件](attachments/tech_causal_candidates_dedup_4577.jsonl)


## 2.4 小结

通过本论文了解了双网格 + BERT 的因果关系抽取方法，并进行了代码实践来复现论文。

## 2.5 参考文献

[1] CCKS 2021. [面向金融领域的篇章级事件抽取和事件因果关系抽取](https://sigkg.cn/ccks2021/wp-content/uploads/2021/04/CCKS2021_%E9%9D%A2%E5%90%91%E9%87%91%E8%9E%8D%E9%A2%86%E5%9F%9F%E7%9A%84%E7%AF%87%E7%AB%A0%E7%BA%A7%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E5%92%8C%E4%BA%8B%E4%BB%B6%E5%9B%A0%E6%9E%9C%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96_final.pdf)[EB/OL]. 2021.

[2] Cui S, Sheng J, Cong X, et al. [Event Causality Extraction with Event Argument Correlations](https://aclanthology.org/2022.coling-1.201/)[C]//Proceedings of the 29th International Conference on Computational Linguistics. 2022: 2300–2312.

[3] 王朱君, 王石, 李雪晴, 等. 基于深度学习的事件因果关系抽取综述[J]. 计算机应用, 2021, 41(05): 1247–1255.

[4] 姜博, 左万利, 王英. 基于 BERT 的因果关系抽取[J]. 吉林大学学报(理学版), 2021, 59(06): 1439–1444.

[5] 李岳泽, 左祥麟, 左万利, 等. 基于 BERT-GCN 的因果关系抽取[J]. 吉林大学学报(理学版), 2023, 61(02): 325–330.
