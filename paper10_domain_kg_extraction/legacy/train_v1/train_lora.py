import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============ 配置（根据实际情况修改） ============
# 注意: 下载完模型后先 ls /root/autodl-tmp/fjy/models/Qwen/ 看实际文件夹名
# ModelScope下载的可能是 Qwen2___5-7B-Instruct 或 Qwen2.5-7B-Instruct
MODEL_PATH = "/root/autodl-tmp/fjy/models/Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = "/root/autodl-tmp/fjy/train_data"
OUTPUT_DIR = "/root/autodl-tmp/fjy/output/kg_extractor_lora"

# ============ 加载tokenizer ============
print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============ 4bit量化加载模型 ============
print("加载模型（4bit量化）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

# ============ LoRA配置 ============
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 加载数据 ============
print("加载训练数据...")
dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/val.jsonl",
})
print(f"训练集: {len(dataset['train'])} 条, 验证集: {len(dataset['validation'])} 条")

def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# ============ 训练参数 ============
# per_device_train_batch_size: 显存不够就改成1
# gradient_accumulation_steps: 相应改成16，保持等效batch_size=16
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=1,
    bf16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# ============ 开始训练 ============
training_args.max_seq_length = 1024
training_args.packing = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}

# 用formatting_func预处理数据
print("预处理数据...")
def preprocess(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["validation"].map(preprocess, remove_columns=dataset["validation"].column_names)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("=" * 50)
print("开始训练...")
print("=" * 50)
trainer.train()

print("保存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n训练完成！模型已保存到: {OUTPUT_DIR}")
