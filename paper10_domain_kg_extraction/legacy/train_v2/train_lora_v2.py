"""
train_lora_v2.py - 分批次持续学习训练 (EWC + 经验回放)

训练顺序: ch/batch_1 -> ch/batch_2 -> ch/batch_3 -> en/batch_1 -> en/batch_2 -> en/batch_3
每个批次训练后:
  1. 计算 Fisher 信息矩阵用于 EWC 正则化
  2. 从当前批次抽样加入经验回放缓冲区
  3. 下一批次训练时混入回放样本，防止灾难性遗忘

用法:
  python train_lora_v2.py                    # 从头开始训练所有批次
  python train_lora_v2.py --start_batch 3    # 从第3个批次开始（跳过已训练的）
  python train_lora_v2.py --ewc_lambda 500   # 调整EWC正则化强度
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import copy
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# ============ 配置 ============
MODEL_PATH = "/root/autodl-tmp/fjy/models/Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/root/autodl-tmp/fjy/train_data"
OUTPUT_DIR = "/root/autodl-tmp/fjy/output/kg_extractor_lora_v2"
REPLAY_BUFFER_PATH = f"{OUTPUT_DIR}/replay_buffer.jsonl"
FISHER_PATH = f"{OUTPUT_DIR}/fisher_info.pt"

BATCH_ORDER = [
    ("ch", "batch_1"), ("ch", "batch_2"), ("ch", "batch_3"),
    ("en", "batch_1"), ("en", "batch_2"), ("en", "batch_3"),
]

# ============ 参数解析 ============
parser = argparse.ArgumentParser()
parser.add_argument("--start_batch", type=int, default=0, help="从第几个批次开始(0-indexed)")
parser.add_argument("--ewc_lambda", type=float, default=400.0, help="EWC正则化强度")
parser.add_argument("--replay_ratio", type=float, default=0.1, help="回放样本占比")
parser.add_argument("--replay_buffer_size", type=int, default=2000, help="回放缓冲区最大样本数")
parser.add_argument("--fisher_samples", type=int, default=200, help="计算Fisher信息用的样本数")
parser.add_argument("--epochs_per_batch", type=int, default=2, help="每批次训练轮数")
parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
parser.add_argument("--batch_size", type=int, default=1, help="per_device_train_batch_size")
parser.add_argument("--grad_accum", type=int, default=16, help="梯度累积步数")
parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列长度")
parser.add_argument("--packing", action="store_true", help="启用packing提升吞吐")
args = parser.parse_args()


# ============ EWC回调 ============
class EWCCallback(TrainerCallback):
    """在每个训练步的loss上加EWC正则项"""

    def __init__(self, model, fisher_dict, prev_params, ewc_lambda):
        self.fisher_dict = fisher_dict
        self.prev_params = prev_params
        self.ewc_lambda = ewc_lambda
        self.model = model

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def compute_ewc_loss(self):
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and param.requires_grad:
                fisher = self.fisher_dict[name]
                prev = self.prev_params[name]
                ewc_loss += (fisher * (param - prev).pow(2)).sum()
        return self.ewc_lambda * 0.5 * ewc_loss


class EWCSFTTrainer(SFTTrainer):
    """重写compute_loss以加入EWC正则项"""

    def __init__(self, *a, ewc_callback=None, **kw):
        super().__init__(*a, **kw)
        self.ewc_cb = ewc_callback

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.ewc_cb and self.ewc_cb.fisher_dict:
            ewc_loss = self.ewc_cb.compute_ewc_loss()
            loss = loss + ewc_loss
        return (loss, outputs) if return_outputs else loss


# ============ 辅助函数 ============
def compute_fisher(model, tokenizer, dataset, num_samples=200):
    """用训练数据子集计算对角Fisher信息矩阵"""
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    for idx in indices:
        text = dataset[idx]["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=args.max_seq_length).to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)

    for name in fisher_dict:
        fisher_dict[name] /= len(indices)

    model.train()
    return fisher_dict


def save_replay_buffer(buffer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_replay_buffer(path):
    if not os.path.exists(path):
        return []
    buffer = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                buffer.append(json.loads(line))
    return buffer


def update_replay_buffer(buffer, dataset, max_size, sample_count):
    """蓄水池抽样: 从当前数据集抽取sample_count条加入buffer"""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    new_samples = [{"text": dataset[i]["text"]} for i in indices[:sample_count]]
    buffer.extend(new_samples)
    if len(buffer) > max_size:
        random.shuffle(buffer)
        buffer = buffer[:max_size]
    return buffer


def preprocess(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def preprocess_logits_for_metrics(logits, labels):
    """显存友好：在 logits 还在 GPU 上时立刻取 argmax，避免把全词表 logits 传给 compute_metrics"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """token-level 准确率：next-token 预测对齐后忽略 -100"""
    preds, labels = eval_pred
    # preds 已经是 argmax 后的 token id (来自 preprocess_logits_for_metrics)
    # 因果 LM: 用位置 t 的预测对齐位置 t+1 的 label
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0, "seq_accuracy": 0.0}

    correct = (preds == labels) & mask
    token_acc = correct.sum() / mask.sum()

    # 序列级准确率：该样本所有非 -100 token 全部预测正确才算对
    per_seq_correct = correct.sum(axis=1)
    per_seq_total = mask.sum(axis=1)
    seq_correct = (per_seq_correct == per_seq_total) & (per_seq_total > 0)
    seq_acc = seq_correct.sum() / max(int((per_seq_total > 0).sum()), 1)

    return {
        "token_accuracy": float(token_acc),
        "seq_accuracy": float(seq_acc),
    }


# ============ 主流程 ============
def main():
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型（4bit量化）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    # 如果从中间批次恢复，加载已有LoRA权重
    lora_checkpoint = f"{OUTPUT_DIR}/lora_latest"
    if args.start_batch > 0 and os.path.exists(lora_checkpoint):
        print(f"从检查点恢复LoRA权重: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # 加载已有的Fisher和回放缓冲
    fisher_dict = {}
    prev_params = {}
    if args.start_batch > 0 and os.path.exists(FISHER_PATH):
        print("加载Fisher信息矩阵...")
        saved = torch.load(FISHER_PATH, map_location="cpu")
        fisher_dict = {k: v.to(model.device) for k, v in saved["fisher"].items()
                       if k in dict(model.named_parameters())}
        prev_params = {k: v.to(model.device) for k, v in saved["params"].items()
                       if k in dict(model.named_parameters())}

    replay_buffer = load_replay_buffer(REPLAY_BUFFER_PATH)
    print(f"回放缓冲区: {len(replay_buffer)} 条")

    # ============ 逐批次训练 ============
    for batch_idx, (lang, batch_name) in enumerate(BATCH_ORDER):
        if batch_idx < args.start_batch:
            continue

        batch_dir = f"{DATA_DIR}/{lang}/{batch_name}"
        train_file = f"{batch_dir}/train.jsonl"
        val_file = f"{batch_dir}/val.jsonl"

        if not os.path.exists(train_file):
            print(f"\n跳过批次 {batch_idx} ({lang}/{batch_name}): 数据不存在")
            continue

        print(f"\n{'='*60}")
        print(f"批次 {batch_idx}/{len(BATCH_ORDER)-1}: {lang}/{batch_name}")
        print(f"{'='*60}")

        dataset = load_dataset("json", data_files={
            "train": train_file, "validation": val_file
        })
        print(f"训练: {len(dataset['train'])} 条, 验证: {len(dataset['validation'])} 条")

        train_ds = dataset["train"].map(
            lambda x: preprocess(x, tokenizer),
            remove_columns=dataset["train"].column_names
        )
        eval_ds = dataset["validation"].map(
            lambda x: preprocess(x, tokenizer),
            remove_columns=dataset["validation"].column_names
        )

        # 混入经验回放样本
        if replay_buffer:
            replay_count = min(
                int(len(train_ds) * args.replay_ratio),
                len(replay_buffer)
            )
            if replay_count > 0:
                sampled = random.sample(replay_buffer, replay_count)
                from datasets import Dataset
                replay_ds = Dataset.from_list(sampled)
                train_ds = concatenate_datasets([train_ds, replay_ds])
                print(f"混入回放样本: {replay_count} 条, 总训练: {len(train_ds)} 条")

        # 训练参数 - 根据之前训练经验优化
        batch_output = f"{OUTPUT_DIR}/batch_{batch_idx}_{lang}_{batch_name}"
        total_steps = (len(train_ds) // (args.batch_size * args.grad_accum)) * args.epochs_per_batch
        save_steps = max(200, total_steps // 6)
        eval_steps = save_steps

        training_args = TrainingArguments(
            output_dir=batch_output,
            num_train_epochs=args.epochs_per_batch,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=eval_steps,
            per_device_eval_batch_size=args.batch_size,
            bf16=True,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        training_args.max_seq_length = args.max_seq_length
        training_args.packing = args.packing
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}

        # 创建EWC训练器
        ewc_cb = EWCCallback(model, fisher_dict, prev_params, args.ewc_lambda)

        trainer = EWCSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            ewc_callback=ewc_cb if fisher_dict else None,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        print(f"开始训练批次 {batch_idx} (steps: ~{total_steps}, save_every: {save_steps})...")
        trainer.train()

        # 保存当前LoRA权重
        print("保存LoRA权重...")
        trainer.save_model(batch_output)
        model.save_pretrained(lora_checkpoint)
        tokenizer.save_pretrained(lora_checkpoint)

        # 计算Fisher信息矩阵
        print(f"计算Fisher信息矩阵 (samples={args.fisher_samples})...")
        new_fisher = compute_fisher(model, tokenizer, train_ds, args.fisher_samples)

        # 合并Fisher: 新旧取平均（如有旧的）
        if fisher_dict:
            for name in new_fisher:
                if name in fisher_dict:
                    fisher_dict[name] = 0.5 * fisher_dict[name] + 0.5 * new_fisher[name]
                else:
                    fisher_dict[name] = new_fisher[name]
        else:
            fisher_dict = new_fisher

        prev_params = {name: param.data.clone()
                       for name, param in model.named_parameters()
                       if param.requires_grad}

        torch.save({"fisher": fisher_dict, "params": prev_params}, FISHER_PATH)

        # 更新回放缓冲区
        sample_count = min(int(len(train_ds) * 0.05), 500)
        replay_buffer = update_replay_buffer(
            replay_buffer, train_ds, args.replay_buffer_size, sample_count
        )
        save_replay_buffer(replay_buffer, REPLAY_BUFFER_PATH)
        print(f"回放缓冲区更新: {len(replay_buffer)} 条")

        print(f"批次 {batch_idx} 训练完成!")

    # 最终保存
    print(f"\n{'='*60}")
    print("所有批次训练完成!")
    print(f"最终LoRA权重: {lora_checkpoint}")
    print(f"各批次检查点: {OUTPUT_DIR}/batch_*/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
