# 知识图谱三元组抽取 Flask Web 应用

基于微调后的 Qwen2.5-7B-Instruct + LoRA 模型，提供交互式 Web 界面进行知识图谱三元组抽取。

## 部署步骤

### 1. 确认环境
```bash
conda activate qwen_finetune
```

### 2. 安装依赖
```bash
pip install flask
# 其他依赖（torch, transformers, peft等）训练环境应已具备
```

### 3. 配置模型路径

编辑 `app.py` 顶部的路径配置：
```python
MODEL_PATH = "/root/autodl-tmp/fjy/models/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/root/autodl-tmp/fjy/output/kg_extractor_lora_v2/lora_latest"
```

如果已有合并后的模型：
```python
USE_MERGED = True
MERGED_MODEL_PATH = "/root/autodl-tmp/fjy/output/kg_extractor_merged"
```

### 4. 启动服务
```bash
cd flask_kg
python app.py
```

服务启动后访问 http://localhost:5000

### 5. 后台运行（可选）
```bash
nohup python app.py > flask.log 2>&1 &
```

## 使用方式

1. 在文本框中输入/粘贴科技新闻文本
2. 选择语言（默认自动检测）
3. 点击"抽取三元组"按钮（或 Ctrl+Enter）
4. 查看结果：
   - **知识图谱可视化**：力导向图展示实体和关系
   - **表格视图**：结构化展示每个三元组
   - **JSON原始输出**：模型输出的完整JSON

## API 接口

### 单条抽取
```
POST /extract
Content-Type: application/json

{"text": "输入文本", "lang": "auto"}
```

### 批量抽取
```
POST /batch_extract
Content-Type: application/json

{"texts": ["文本1", "文本2"], "lang": "auto"}
```

## 文件结构
```
flask_kg/
├── app.py              # Flask主程序（模型加载+推理+路由）
├── templates/
│   └── index.html      # 前端页面（交互+可视化）
├── requirements.txt    # Python依赖
└── README.md           # 本文件
```
