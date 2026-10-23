"""
Flask 知识图谱三元组抽取 Web 应用
基于微调后的 Qwen2.5 + LoRA 模型

环境变量（复现时用，避免改代码）：
  KG_MODEL_PATH   基础模型目录，默认按 models/Qwen/... 查找
  KG_LORA_PATH    LoRA 适配器目录，默认按 output/.../lora_latest 查找
  KG_MERGED_PATH  合并后的完整模型目录（可选，优先级最高）
  KG_LOAD_4BIT    设为 1 才启用 4bit 量化；默认 bf16/fp32
  KG_MAX_NEW_TOKENS  生成上限，默认 1024，快速测试可设 256

路径说明：
  所有路径基于项目根目录（flask 的上一级目录）动态计算
  目录结构：
    workend/
    ├── output/kg_extractor_lora_v2/lora_latest/   (LoRA权重)
    ├── output/kg_extractor_merged/                (合并后的完整模型)
    ├── models/Qwen/Qwen2.5-7B-Instruct/           (基础模型)
    ├── ch/  en/                                   (原始数据)
    └── flask/                                     (本项目)
"""
import os
import json
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

# ============ 动态路径��置 ============
# 项目根目录 = flask_kg 的上级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基础模型路径（按优先级查找）
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct"),
    os.path.join(BASE_DIR, "models", "Qwen2.5-7B-Instruct"),
    "Qwen/Qwen2.5-7B-Instruct",  # 最后兜底：从HuggingFace在线加载
]

# LoRA权重路径（按优先级查找）
_LORA_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "kg_extractor_lora_v2", "lora_latest"),
    os.path.join(BASE_DIR, "output", "kg_extractor_lora"),
]

# 合���后的完整模型路径
_MERGED_CANDIDATES = [
    os.path.join(BASE_DIR, "output", "kg_extractor_merged"),
]


def _find_path(candidates, name):
    """从候选路径中找到第一个存在的"""
    for p in candidates:
        if os.path.isdir(p):
            return p
    # 最后一个候选如果是HF model id（不含os.sep），直接返回
    last = candidates[-1]
    if os.sep not in last and "/" in last:
        return last
    return None


MODEL_PATH = os.environ.get("KG_MODEL_PATH") or _find_path(_MODEL_CANDIDATES, "基础模型")
LORA_PATH = os.environ.get("KG_LORA_PATH") or _find_path(_LORA_CANDIDATES, "LoRA权重")
MERGED_MODEL_PATH = os.environ.get("KG_MERGED_PATH") or _find_path(_MERGED_CANDIDATES, "合并模型")

# 自动判断加载模式：优先用合并模型，否则用基础模型+LoRA
USE_MERGED = MERGED_MODEL_PATH is not None and os.path.isdir(MERGED_MODEL_PATH or "")

SYSTEM_PROMPT_ZH = (
    "从科技新闻中抽取知识图谱三元组。输出JSON格式，包含year、knowledge_type和triples数组。"
    "每个triple包含entity1、relation、entity2、relation_type、hierarchy_level。"
    "关系必须从预定义集合中选择：发布、提出、发明、创建、研发、开发、设计、制造、收购、投资、"
    "合作、应用、超越、导致、取代、融合、升级、基于、是、属于、包含、举办、担任、攻克、验证、探讨。"
    "只输出JSON，不要解释。"
)

SYSTEM_PROMPT_EN = (
    "Extract knowledge graph triples from tech news. Output JSON with year, knowledge_type, "
    "and triples array. Each triple contains entity1, relation, entity2, relation_type, "
    "hierarchy_level. Relations must be from: release, propose, invent, create, develop, "
    "design, manufacture, acquire, invest, cooperate, apply, surpass, cause, replace, "
    "integrate, upgrade, based_on, is, belong_to, contain, host, serve_as, overcome, "
    "verify, discuss, has_feature. Output JSON only, no explanation."
)

# ============ 全局模型变量 ============
model = None
tokenizer = None


def load_model():
    """加载模型（启动时调用一次）"""
    global model, tokenizer

    print(f"[配置] 项目根目录: {BASE_DIR}")
    print(f"[配置] 加载模式: {'合并模型' if USE_MERGED else '基础模型+LoRA'}")

    if USE_MERGED:
        print(f"[配置] 合并模型路径: {MERGED_MODEL_PATH}")
        print("正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("正在加载合并后的完整模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        if not MODEL_PATH:
            raise RuntimeError(
                f"未找到基础模型！请将 Qwen2.5-7B-Instruct 放到以下任一路径:\n"
                + "\n".join(f"  - {p}" for p in _MODEL_CANDIDATES[:-1])
                + "\n或确保可访问 HuggingFace 在线下载"
            )
        if not LORA_PATH:
            raise RuntimeError(
                f"未找到LoRA权重！请确认以下路径存在:\n"
                + "\n".join(f"  - {p}" for p in _LORA_CANDIDATES)
            )

        print(f"[配置] 基础模型路径: {MODEL_PATH}")
        print(f"[配置] LoRA权重路径: {LORA_PATH}")

        print("正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        use_4bit = os.environ.get("KG_LOAD_4BIT", "0") == "1"
        print(f"正在加载基础模型（{'4bit量化' if use_4bit else 'bf16/fp32'}）...")
        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        print(f"正在加载 LoRA 权重: {LORA_PATH}")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model.eval()

    print("模型加载完成！")


def extract_triples(text, lang="auto"):
    """
    对输入文本进行三元组抽取推理
    lang: "zh" | "en" | "auto"
    """
    # 自动检测语言
    if lang == "auto":
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        lang = "zh" if chinese_chars > len(text) * 0.1 else "en"

    system_prompt = SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("KG_MAX_NEW_TOKENS", "1024")),
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # 尝试解析JSON
    try:
        # 清理可能的markdown代码块标记
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        result = json.loads(clean)
    except json.JSONDecodeError:
        # 尝试寻找JSON部分
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            result = json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            result = {"raw_output": response, "parse_error": True}

    return result, lang


# ============ 路由 ============
@app.route("/")
def index():
    """渲染前端页面（templates/index.html）。"""
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    """单条文本抽取接口。

    请求 JSON：``{"text": "...", "lang": "zh|en|auto"}``；
    返回：``{"success": true, "language": "...", "result": {...}}``。
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "请提供text字段"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "输入文本不能为空"}), 400

    lang = data.get("lang", "auto")

    try:
        result, detected_lang = extract_triples(text, lang)
        return jsonify({
            "success": True,
            "input_text": text,
            "language": detected_lang,
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_extract", methods=["POST"])
def batch_extract():
    """批量抽取接口：请求 ``{"texts": ["...", ...]}``，逐条返回结果。"""
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "请提供texts数组"}), 400

    texts = data["texts"]
    lang = data.get("lang", "auto")
    results = []

    for text in texts:
        text = text.strip()
        if text:
            try:
                result, detected_lang = extract_triples(text, lang)
                results.append({
                    "input_text": text,
                    "language": detected_lang,
                    "result": result,
                })
            except Exception as e:
                results.append({"input_text": text, "error": str(e)})

    return jsonify({"success": True, "results": results})


# ============ 启动 ============
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
