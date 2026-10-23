"""
batch_inference.py - 批量推理 + 人工审查样本生成

从各批次验证集中抽样，做推理生成，保存输入-标准答案-模型输出对比，
输出 JSONL 文件供人工抽检。

用法:
  python batch_inference.py
  python batch_inference.py --samples_per_batch 30
  python batch_inference.py --output_file results.jsonl
"""

import os
import json
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime

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
parser.add_argument("--samples_per_batch", type=int, default=20,
                    help="每个批次抽样数")
parser.add_argument("--max_new_tokens", type=int, default=768)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--max_seq_length", type=int, default=1024)
parser.add_argument("--output_file", type=str, default=None,
                    help="输出文件路径 (默认: OUTPUT_DIR/inference_review_YYYYMMDD.jsonl)")
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


def run_inference(model, tokenizer, messages):
    """对单条消息做推理"""
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


def try_parse_json(text):
    """尝试解析 JSON"""
    import re
    text = text.strip()
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), True
        except json.JSONDecodeError:
            pass
    return None, False


def main():
    model, tokenizer = load_model_and_tokenizer()

    # 确定输出路径
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_DIR}/inference_review_{timestamp}.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_samples = 0
    total_json_ok = 0
    results_by_batch = {}

    print(f"\n{'='*80}")
    print(f"批量推理 - 每批次抽样 {args.samples_per_batch} 条")
    print(f"输出文件: {output_file}")
    print(f"{'='*80}\n")

    with open(output_file, "w", encoding="utf-8") as fout:
        for batch_idx, (lang, batch_name) in enumerate(BATCH_ORDER):
            val_file = f"{args.data_dir}/{lang}/{batch_name}/val.jsonl"
            if not os.path.exists(val_file):
                print(f"[SKIP] 批次 {batch_idx} ({lang}/{batch_name}): 验证集不存在")
                continue

            ds = load_dataset("json", data_files=val_file, split="train")
            n_sample = min(args.samples_per_batch, len(ds))
            indices = np.random.RandomState(42 + batch_idx).choice(
                len(ds), n_sample, replace=False
            )

            batch_key = f"{lang}/{batch_name}"
            batch_json_ok = 0
            print(f"--- 批次 {batch_idx}: {batch_key} (抽样 {n_sample} 条) ---")

            for i, idx in enumerate(tqdm(indices, desc=f"B{batch_idx}", ncols=60)):
                item = ds[int(idx)]
                messages = item["messages"]

                # 提取各角色内容
                system_text = ""
                user_text = ""
                gold_text = ""
                for m in messages:
                    if m["role"] == "system":
                        system_text = m["content"]
                    elif m["role"] == "user":
                        user_text = m["content"]
                    elif m["role"] == "assistant":
                        gold_text = m["content"]

                # 推理
                pred_text = run_inference(model, tokenizer, messages)

                # 解析
                gold_parsed, gold_ok = try_parse_json(gold_text)
                pred_parsed, pred_ok = try_parse_json(pred_text)

                if pred_ok:
                    batch_json_ok += 1
                    total_json_ok += 1
                total_samples += 1

                # 统计三元组数
                gold_n_triples = len(gold_parsed.get("triples", [])) if gold_parsed else 0
                pred_n_triples = len(pred_parsed.get("triples", [])) if pred_parsed else 0

                record = {
                    "batch": batch_key,
                    "batch_idx": batch_idx,
                    "sample_idx": int(idx),
                    "lang": lang,
                    "input": user_text,
                    "gold_output": gold_text,
                    "pred_output": pred_text,
                    "gold_json_valid": gold_ok,
                    "pred_json_valid": pred_ok,
                    "gold_triples_count": gold_n_triples,
                    "pred_triples_count": pred_n_triples,
                    # 预留人工审查字段
                    "review": {
                        "json_format_ok": None,       # bool: JSON 格式是否正确
                        "entities_correct": None,      # bool: 实体抽取是否准确
                        "relations_correct": None,     # bool: 关系抽取是否准确
                        "hierarchy_correct": None,     # bool: 层级关系是否合理
                        "missing_triples": None,       # int: 遗漏的三元组数
                        "extra_triples": None,         # int: 多余的三元组数
                        "notes": "",                   # str: 审查备注
                    }
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            results_by_batch[batch_key] = {
                "samples": n_sample,
                "json_ok": batch_json_ok,
                "json_rate": round(batch_json_ok / n_sample * 100, 1),
            }
            print(f"  JSON 解析: {batch_json_ok}/{n_sample} ({batch_json_ok/n_sample*100:.1f}%)")

    # ====== 汇总 ======
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")
    print(f"总样本数: {total_samples}")
    print(f"JSON 解析成功率: {total_json_ok}/{total_samples} ({total_json_ok/total_samples*100:.1f}%)")
    print(f"\n按批次统计:")
    for batch_key, stats in results_by_batch.items():
        print(f"  {batch_key}: {stats['json_ok']}/{stats['samples']} ({stats['json_rate']}%)")

    print(f"\n输出文件: {output_file}")
    print(f"\n人工审查说明:")
    print(f"  每条记录包含 'review' 字段，审查时填写以下内容:")
    print(f"  - json_format_ok: JSON 格式是否完整正确")
    print(f"  - entities_correct: 实体是否正确识别")
    print(f"  - relations_correct: 关系类型是否准确")
    print(f"  - hierarchy_correct: hierarchy_level 是否合理")
    print(f"  - missing_triples: 遗漏了多少应该抽取的三元组")
    print(f"  - extra_triples: 多余了多少不应存在的三元组")
    print(f"  - notes: 其他备注")

    # 保存汇总统计
    summary_file = output_file.replace(".jsonl", "_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "total_json_ok": total_json_ok,
        "json_parse_rate": round(total_json_ok / total_samples, 4),
        "samples_per_batch": args.samples_per_batch,
        "per_batch": results_by_batch,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 0.9,
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总统计: {summary_file}")


if __name__ == "__main__":
    main()
