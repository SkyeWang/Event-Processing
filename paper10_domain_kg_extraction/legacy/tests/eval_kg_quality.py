"""
eval_kg_quality.py - 知识图谱抽取质量评估

对测试集做推理，解析 JSON 输出，���算实体级和关系级的
Precision / Recall / F1，并按 relation_type 和 hierarchy_level 分层分析。

用法:
  # 使用验证集作为测试 (默认)
  python eval_kg_quality.py

  # 指定独立测试集目录
  python eval_kg_quality.py --test_dir /path/to/test_data

  # 调整生成参数
  python eval_kg_quality.py --max_new_tokens 1024 --temperature 0.1
"""

import os
import json
import argparse
import re
import torch
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ============ 默认路径 ============
MODEL_PATH = "/root/autodl-tmp/fjy/models/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/root/autodl-tmp/fjy/output/kg_extractor_lora_v2"
DATA_DIR = "/root/autodl-tmp/fjy/train_data"
LORA_PATH = f"{OUTPUT_DIR}/lora_latest"

BATCH_ORDER = [
    ("ch", "batch_1"), ("ch", "batch_2"), ("ch", "batch_3"),
    ("en", "batch_1"), ("en", "batch_2"), ("en", "batch_3"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=MODEL_PATH)
parser.add_argument("--lora_path", type=str, default=LORA_PATH)
parser.add_argument("--data_dir", type=str, default=DATA_DIR)
parser.add_argument("--test_dir", type=str, default=None,
                    help="独立测试集目录 (含 test.jsonl); 默认用各批次 val.jsonl")
parser.add_argument("--max_samples", type=int, default=100,
                    help="每个批次最多评估样本数")
parser.add_argument("--max_new_tokens", type=int, default=768)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=1024)
args = parser.parse_args()


def load_model_and_tokenizer():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载基座模型 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    print(f"加载 LoRA 权重: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def extract_json_from_text(text):
    """从模型输出中提取 JSON，处理各种边界情况"""
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 { ... } 块
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 尝试修复常见的 JSON 错误 (尾部逗号)
    if match:
        fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


def get_triples_set(data):
    """从解析后的 JSON 中提取三元组集合"""
    if data is None or "triples" not in data:
        return set(), set(), []

    entities = set()
    relations = set()
    triples = []

    for t in data["triples"]:
        e1 = t.get("entity1", "").strip()
        rel = t.get("relation", "").strip()
        e2 = t.get("entity2", "").strip()
        rtype = t.get("relation_type", "").strip()
        hlevel = str(t.get("hierarchy_level", "")).strip()

        if e1:
            entities.add(e1)
        if e2:
            entities.add(e2)
        if e1 and rel and e2:
            relations.add((e1, rel, e2))
            triples.append({
                "entity1": e1, "relation": rel, "entity2": e2,
                "relation_type": rtype, "hierarchy_level": hlevel,
            })

    return entities, relations, triples


def compute_prf(pred_set, gold_set):
    """计算 Precision, Recall, F1"""
    if len(pred_set) == 0 and len(gold_set) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0
    if len(gold_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_inference(model, tokenizer, messages):
    """对单条消息做推理，返回 assistant 生成的文本"""
    # 构建 prompt (不含 assistant 回复)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=args.max_seq_length
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    model, tokenizer = load_model_and_tokenizer()

    # 收集测试数据
    test_samples = []

    if args.test_dir:
        # 使用指定的测试集
        for lang_dir in ["ch", "en"]:
            test_file = os.path.join(args.test_dir, lang_dir, "test.jsonl")
            if not os.path.exists(test_file):
                test_file = os.path.join(args.test_dir, "test.jsonl")
            if os.path.exists(test_file):
                ds = load_dataset("json", data_files=test_file, split="train")
                for item in ds:
                    test_samples.append({"lang": lang_dir, "messages": item["messages"]})
    else:
        # 默认: 从每个批次的验证集中抽样
        for batch_idx, (lang, batch_name) in enumerate(BATCH_ORDER):
            val_file = f"{args.data_dir}/{lang}/{batch_name}/val.jsonl"
            if not os.path.exists(val_file):
                continue
            ds = load_dataset("json", data_files=val_file, split="train")
            indices = np.random.RandomState(42).choice(
                len(ds), min(args.max_samples, len(ds)), replace=False
            )
            for i in indices:
                test_samples.append({
                    "lang": lang,
                    "batch": f"{lang}/{batch_name}",
                    "messages": ds[int(i)]["messages"],
                })

    print(f"\n总共 {len(test_samples)} 条测试样本")

    # ====== 推理 + 评估 ======
    all_entity_p, all_entity_r, all_entity_f1 = [], [], []
    all_rel_p, all_rel_r, all_rel_f1 = [], [], []
    json_parse_ok = 0
    json_parse_fail = 0
    per_rtype_metrics = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    per_hlevel_metrics = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    per_lang_metrics = defaultdict(lambda: {
        "entity_f1": [], "rel_f1": [], "json_ok": 0, "json_fail": 0
    })

    detailed_results = []

    for sample in tqdm(test_samples, desc="Inference + Eval", ncols=80):
        messages = sample["messages"]
        lang = sample.get("lang", "unknown")

        # 获取 ground truth
        gold_text = ""
        for m in messages:
            if m["role"] == "assistant":
                gold_text = m["content"]
                break

        gold_data = extract_json_from_text(gold_text)
        gold_entities, gold_relations, gold_triples = get_triples_set(gold_data)

        # 推理
        pred_text = run_inference(model, tokenizer, messages)
        pred_data = extract_json_from_text(pred_text)

        if pred_data is not None:
            json_parse_ok += 1
            per_lang_metrics[lang]["json_ok"] += 1
        else:
            json_parse_fail += 1
            per_lang_metrics[lang]["json_fail"] += 1

        pred_entities, pred_relations, pred_triples = get_triples_set(pred_data)

        # 实体级指标
        ep, er, ef1 = compute_prf(pred_entities, gold_entities)
        all_entity_p.append(ep)
        all_entity_r.append(er)
        all_entity_f1.append(ef1)
        per_lang_metrics[lang]["entity_f1"].append(ef1)

        # 关系级指标 (三元组精确匹配)
        rp, rr, rf1 = compute_prf(pred_relations, gold_relations)
        all_rel_p.append(rp)
        all_rel_r.append(rr)
        all_rel_f1.append(rf1)
        per_lang_metrics[lang]["rel_f1"].append(rf1)

        # 按 relation_type 统计
        for t in gold_triples:
            rtype = t["relation_type"]
            per_rtype_metrics[rtype]["gold"] += 1
            if (t["entity1"], t["relation"], t["entity2"]) in pred_relations:
                per_rtype_metrics[rtype]["tp"] += 1
        for t in pred_triples:
            rtype = t["relation_type"]
            per_rtype_metrics[rtype]["pred"] += 1

        # 按 hierarchy_level 统计
        for t in gold_triples:
            hl = t["hierarchy_level"]
            per_hlevel_metrics[hl]["gold"] += 1
            if (t["entity1"], t["relation"], t["entity2"]) in pred_relations:
                per_hlevel_metrics[hl]["tp"] += 1
        for t in pred_triples:
            hl = t["hierarchy_level"]
            per_hlevel_metrics[hl]["pred"] += 1

        # 保存详细结果
        user_text = ""
        for m in messages:
            if m["role"] == "user":
                user_text = m["content"][:200]
                break

        detailed_results.append({
            "lang": lang,
            "input": user_text,
            "gold_triples_count": len(gold_relations),
            "pred_triples_count": len(pred_relations),
            "entity_f1": round(ef1, 4),
            "relation_f1": round(rf1, 4),
            "json_valid": pred_data is not None,
        })

    # ====== 输出报告 ======
    print(f"\n{'='*80}")
    print("KG 抽取质量评估报告")
    print(f"{'='*80}")

    total = json_parse_ok + json_parse_fail
    print(f"\nJSON 解析成功率: {json_parse_ok}/{total} ({json_parse_ok/total*100:.1f}%)")

    print(f"\n--- 整体指标 ---")
    print(f"实体级  Precision: {np.mean(all_entity_p):.4f}  Recall: {np.mean(all_entity_r):.4f}  F1: {np.mean(all_entity_f1):.4f}")
    print(f"关系级  Precision: {np.mean(all_rel_p):.4f}  Recall: {np.mean(all_rel_r):.4f}  F1: {np.mean(all_rel_f1):.4f}")

    print(f"\n--- 按语言分 ---")
    for lang in sorted(per_lang_metrics.keys()):
        m = per_lang_metrics[lang]
        total_l = m["json_ok"] + m["json_fail"]
        ent_f1 = np.mean(m["entity_f1"]) if m["entity_f1"] else 0
        rel_f1 = np.mean(m["rel_f1"]) if m["rel_f1"] else 0
        print(f"  {lang}: JSON OK {m['json_ok']}/{total_l}, Entity F1: {ent_f1:.4f}, Relation F1: {rel_f1:.4f}")

    print(f"\n--- 按 relation_type 分 ---")
    print(f"{'Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>8} {'Pred':>8}")
    print("-" * 68)
    for rtype in sorted(per_rtype_metrics.keys()):
        m = per_rtype_metrics[rtype]
        p = m["tp"] / m["pred"] if m["pred"] > 0 else 0
        r = m["tp"] / m["gold"] if m["gold"] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{rtype:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {m['gold']:>8} {m['pred']:>8}")

    print(f"\n--- 按 hierarchy_level 分 ---")
    print(f"{'Level':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>8} {'Pred':>8}")
    print("-" * 58)
    for hl in sorted(per_hlevel_metrics.keys()):
        m = per_hlevel_metrics[hl]
        p = m["tp"] / m["pred"] if m["pred"] > 0 else 0
        r = m["tp"] / m["gold"] if m["gold"] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"{hl:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {m['gold']:>8} {m['pred']:>8}")

    # 保存结果
    output_file = f"{OUTPUT_DIR}/kg_quality_results.json"
    summary = {
        "total_samples": total,
        "json_parse_rate": round(json_parse_ok / total, 4),
        "entity_precision": round(float(np.mean(all_entity_p)), 4),
        "entity_recall": round(float(np.mean(all_entity_r)), 4),
        "entity_f1": round(float(np.mean(all_entity_f1)), 4),
        "relation_precision": round(float(np.mean(all_rel_p)), 4),
        "relation_recall": round(float(np.mean(all_rel_r)), 4),
        "relation_f1": round(float(np.mean(all_rel_f1)), 4),
        "per_lang": {
            lang: {
                "entity_f1": round(float(np.mean(m["entity_f1"])), 4) if m["entity_f1"] else 0,
                "relation_f1": round(float(np.mean(m["rel_f1"])), 4) if m["rel_f1"] else 0,
                "json_ok": m["json_ok"],
                "json_fail": m["json_fail"],
            }
            for lang, m in per_lang_metrics.items()
        },
        "detailed_results": detailed_results,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
