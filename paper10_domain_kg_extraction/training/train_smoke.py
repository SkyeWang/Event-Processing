"""LoRA + EWC + 经验回放的小模型训练脚本（流程复现版）。

原 train_lora_v2.py 面向 AutoDL 上的 4bit Qwen2.5-7B，使用绝对 Linux 路径，
且 trl 版本与当前环境不一致。本脚本保留原配方（LoRA、对角 Fisher EWC、
回放缓冲区、六批次顺序），但改用 0.5B 模型，便于在单卡上验证流程。

流程：
1. 读取 data/train_data/{ch,en}/batch_*/train.jsonl 与 val.jsonl；
2. 用 chat template 拼成文本，tokenize 后交给 Trainer；
3. 每批训练结束后计算 Fisher、更新回放缓冲区、保存 LoRA 适配器；
4. 下一批训练时按 replay_ratio 混入历史样本。

示例：
    python train_smoke.py --model_path models/Qwen2.5-0.5B-Instruct \
        --data_dir data/train_data --output_dir output/smoke --max_steps 3
"""

import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BATCH_ORDER = [
    ("ch", "batch_1"), ("ch", "batch_2"), ("ch", "batch_3"),
    ("en", "batch_1"), ("en", "batch_2"), ("en", "batch_3"),
]


def read_jsonl(path):
    """读取 JSONL，返回 dict 列表；文件不存在时返回空列表（用于跳过缺失批次）。"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render(rows, tokenizer):
    """用 chat template 把 messages 拼成训练文本。"""
    return [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows
    ]


def encode(rows, tokenizer, max_len):
    """tokenize 后把 input_ids 复制为 labels（全序列语言模型训练）。"""
    texts = render(rows, tokenizer)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=False)
    enc["labels"] = [ids[:] for ids in enc["input_ids"]]
    return Dataset.from_dict(enc)


class EWC:
    # EWC：保存旧任务参数 prev 和对角 Fisher，训练新批次时加
    # lambda/2 * sum F_i (theta_i - theta*_i)^2 的正则项。
    def __init__(self, model, lam):
        self.model = model
        self.lam = lam
        self.fisher = {}
        self.prev = {}

    def penalty(self):
        """计算 EWC 正则项；Fisher 为空（第一批）时返回 0。"""
        if not self.fisher:
            return torch.zeros((), device=next(self.model.parameters()).device)
        total = torch.zeros((), device=next(self.model.parameters()).device)
        trainable = dict(self.model.named_parameters())
        for name, f in self.fisher.items():
            p = trainable[name]
            total = total + (f * (p - self.prev[name]).pow(2)).sum()
        return self.lam * 0.5 * total

    def snapshot(self):
        """保存当前可训练参数，作为下一批 EWC 的 theta*（旧任务最优参数）。"""
        self.prev = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }


class EWCTrainer(Trainer):
    """在 transformers.Trainer 的 loss 上叠加 EWC 正则项。"""

    def __init__(self, *args, ewc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """前向 + 语言模型 loss + EWC penalty。"""
        outputs = model(**inputs)
        loss = outputs.loss
        if self.ewc is not None:
            loss = loss + self.ewc.penalty()
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def evaluate(model, tokenizer, rows, max_len):
    """在验证集上算平均语言模型 loss（不反传）。"""
    if not rows:
        return None
    model.eval()
    texts = render(rows, tokenizer)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    loss = model(**enc, labels=enc["input_ids"]).loss
    model.train()
    return float(loss)


def compute_fisher(model, tokenizer, rows, max_len, num_samples):
    """用当前批次的一小部分样本估计对角 Fisher 信息。

    对每个可训练参数累加梯度平方，再除以样本数；样本越少噪声越大。
    本脚本默认只取 6 条，用于验证 EWC 链路。
    """
    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    if not rows:
        return fisher
    model.eval()
    device = next(model.parameters()).device
    sample = rows[:num_samples]
    for row in sample:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).to(device)
        model.zero_grad(set_to_none=True)
        loss = model(**enc, labels=enc["input_ids"]).loss
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach().pow(2)
    for name in fisher:
        fisher[name] /= max(len(sample), 1)
    model.zero_grad(set_to_none=True)
    model.train()
    return fisher


def main():
    """逐批训练入口：加载模型/LoRA，循环批次，保存适配器和 summary。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--ewc_lambda", type=float, default=400.0)
    parser.add_argument("--fisher_samples", type=int, default=6)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model.config.use_cache = False
    if torch.cuda.is_available():
        model = model.to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ewc = EWC(model, args.ewc_lambda)
    replay = []
    summary = {"batches": [], "model": args.model_path}

    # 按论文顺序训练：ch/batch_1 -> ch/batch_2 -> ch/batch_3 -> en/batch_1 ...
    # 缺失的批次（本仓库只有 3 个）会自动跳过。
    for idx, (lang, batch) in enumerate(BATCH_ORDER):
        train_rows = read_jsonl(os.path.join(args.data_dir, lang, batch, "train.jsonl"))
        val_rows = read_jsonl(os.path.join(args.data_dir, lang, batch, "val.jsonl"))
        if not train_rows:
            print(f"skip {lang}/{batch}: no data")
            continue

        print(f"\n=== batch {idx}: {lang}/{batch} ({len(train_rows)} train, {len(val_rows)} val) ===")
        if replay:
            take = max(1, int(len(train_rows) * args.replay_ratio))
            train_rows = train_rows + random.sample(replay, min(take, len(replay)))
            print(f"mixed {min(take, len(replay))} replay samples")

        train_ds = encode(train_rows, tokenizer, args.max_seq_length)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"batch_{idx}_{lang}_{batch}"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="no",
            report_to=[],
            bf16=torch.cuda.is_available(),
            optim="adamw_torch",
            max_grad_norm=0.3,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            seed=args.seed,
        )

        # 第一批 Fisher 为空，不启用 EWC；从第二批开始传入 ewc 对象，
        # EWCTrainer.compute_loss 会在语言模型 loss 上叠加正则项。
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
            ewc=ewc if ewc.fisher else None,
        )
        result = trainer.train()
        batch_loss = float(result.metrics.get("train_loss", 0.0))
        val_loss = evaluate(model, tokenizer, val_rows, args.max_seq_length)
        print(f"batch {idx} loss={batch_loss:.4f} val_loss={val_loss}")

        ckpt = os.path.join(args.output_dir, f"lora_{lang}_{batch}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

        new_fisher = compute_fisher(model, tokenizer, train_rows, args.max_seq_length, args.fisher_samples)
        if ewc.fisher:
            for name in new_fisher:
                if name in ewc.fisher:
                    ewc.fisher[name] = 0.5 * ewc.fisher[name] + 0.5 * new_fisher[name]
                else:
                    ewc.fisher[name] = new_fisher[name]
        else:
            ewc.fisher = new_fisher
        ewc.snapshot()

        replay.extend(train_rows[:2])
        summary["batches"].append({
            "index": idx, "lang": lang, "batch": batch,
            "train_samples": len(train_rows), "val_samples": len(val_rows),
            "train_loss": batch_loss, "val_loss": val_loss,
            "fisher_params": len(ewc.fisher), "lora_dir": ckpt,
        })

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\nsummary ->", os.path.join(args.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
