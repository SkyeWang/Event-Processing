import json
import glob
import numpy as np

# 批次配置
BATCH_ORDER = [
    ("ch", "batch_1"), ("ch", "batch_2"), ("ch", "batch_3"),
    ("en", "batch_1"), ("en", "batch_2"), ("en", "batch_3"),
]

OUTPUT_DIR = "/root/autodl-tmp/fjy/output/kg_extractor_lora_v2"

# 存储所有批次的数据
batch_data = []

# 逐个批次读取数据
for batch_idx, (lang, batch_name) in enumerate(BATCH_ORDER):
    batch_dir = f"{OUTPUT_DIR}/batch_{batch_idx}_{lang}_{batch_name}"
    checkpoint_files = glob.glob(f"{batch_dir}/checkpoint-*/trainer_state.json")
    
    if not checkpoint_files:
        print(f"跳过批次 {batch_idx} ({lang}/{batch_name}): 未找到检查点")
        continue
    
    # 选择最新的检查点（数字最大的）
    checkpoint_files.sort(key=lambda x: int(x.split('-')[-1].split('/')[0]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"分析批次 {batch_idx} ({lang}/{batch_name}): {latest_checkpoint}")
    
    with open(latest_checkpoint, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state["log_history"]
    
    # 提取数据
    train_losses = []
    eval_losses = []
    learning_rates = []
    grad_norms = []
    
    for entry in log_history:
        step = entry.get("step", 0)
        
        if "loss" in entry:
            train_losses.append({"step": step, "loss": entry["loss"]})
        
        if "eval_loss" in entry:
            eval_losses.append({"step": step, "loss": entry["eval_loss"]})
        
        if "learning_rate" in entry:
            learning_rates.append({"step": step, "lr": entry["learning_rate"]})
        
        if "grad_norm" in entry:
            grad_norms.append({"step": step, "grad_norm": entry["grad_norm"]})
    
    batch_data.append({
        "batch_idx": batch_idx,
        "lang": lang,
        "batch_name": batch_name,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
        "final_step": trainer_state.get("global_step", 0),
        "final_epoch": trainer_state.get("epoch", 0),
    })

# 生成分析报告
print("\n" + "="*80)
print("多批次持续学习训练分析报告")
print("="*80)

# 1. 各批次损失对比
print("\n" + "="*80)
print("一、各批次损失对比")
print("="*80)

print(f"{'批次':<15} {'语言':<6} {'初始训练损失':<15} {'最终训练损失':<15} {'最终验证损失':<15} {'损失下降':<10}")
print("-"*80)

for data in batch_data:
    lang = data["lang"]
    batch_name = data["batch_name"]
    train_losses = data["train_losses"]
    eval_losses = data["eval_losses"]
    
    initial_loss = train_losses[0]["loss"] if train_losses else None
    final_train_loss = train_losses[-1]["loss"] if train_losses else None
    final_eval_loss = eval_losses[-1]["loss"] if eval_losses else None
    
    if initial_loss and final_train_loss:
        decrease_rate = ((initial_loss - final_train_loss) / initial_loss * 100)
    else:
        decrease_rate = None
    
    print(f"{f'{lang}/{batch_name}':<15} {lang:<6} "
          f"{initial_loss:<15.4f} {final_train_loss:<15.4f} "
          f"{final_eval_loss:<15.4f} {f'{decrease_rate:.1f}%':<10}")

# 2. 各批次梯度范数统计
print("\n" + "="*80)
print("二、各批次梯度范数统计（训练稳定性）")
print("="*80)

print(f"{'批次':<15} {'语言':<6} {'平均梯度范数':<15} {'最大梯度范数':<15} {'最小梯度范数':<15} {'标准差':<10}")
print("-"*80)

for data in batch_data:
    grad_norms = data["grad_norms"]
    if grad_norms:
        gn_array = np.array([g["grad_norm"] for g in grad_norms])
        mean_gn = np.mean(gn_array)
        max_gn = np.max(gn_array)
        min_gn = np.min(gn_array)
        std_gn = np.std(gn_array)
    else:
        mean_gn = max_gn = min_gn = std_gn = None
    
    print(f"{f'{data["lang"]}/{data["batch_name"]}':<15} {data['lang']:<6} "
          f"{mean_gn:<15.4f} {max_gn:<15.4f} {min_gn:<15.4f} {std_gn:<10.4f}")

# 3. 各批次学习率分析
print("\n" + "="*80)
print("三、各批次学习率分析（余弦退火调度器）")
print("="*80)

print(f"{'批次':<15} {'语言':<6} {'初始学习率':<15} {'峰值学习率':<15} {'最终学习率':<15}")
print("-"*80)

for data in batch_data:
    learning_rates = data["learning_rates"]
    if learning_rates:
        initial_lr = learning_rates[0]["lr"]
        peak_lr = max(l["lr"] for l in learning_rates)
        final_lr = learning_rates[-1]["lr"]
    else:
        initial_lr = peak_lr = final_lr = None
    
    print(f"{f'{data["lang"]}/{data["batch_name"]}':<15} {data['lang']:<6} "
          f"{f'{initial_lr:.2e}':<15} {f'{peak_lr:.2e}':<15} {f'{final_lr:.2e}':<15}")

# 4. 持续学习效果分析
print("\n" + "="*80)
print("四、持续学习效果分析")
print("="*80)

# 按语言分组分析
ch_batches = [d for d in batch_data if d["lang"] == "ch"]
en_batches = [d for d in batch_data if d["lang"] == "en"]

print("\n【中文批次分析】")
if ch_batches:
    print(f"批次数量: {len(ch_batches)}")
    avg_train_loss = np.mean([d["train_losses"][-1]["loss"] for d in ch_batches if d["train_losses"]])
    avg_eval_loss = np.mean([d["eval_losses"][-1]["loss"] for d in ch_batches if d["eval_losses"]])
    print(f"平均训练损失: {avg_train_loss:.4f}")
    print(f"平均验证损失: {avg_eval_loss:.4f}")
    
    # 检查是否有遗忘现象
    first_ch_eval = ch_batches[0]["eval_losses"][-1]["loss"] if ch_batches[0]["eval_losses"] else None
    last_ch_eval = ch_batches[-1]["eval_losses"][-1]["loss"] if ch_batches[-1]["eval_losses"] else None
    if first_ch_eval and last_ch_eval:
        print(f"验证损失变化: {first_ch_eval:.4f} → {last_ch_eval:.4f}")
        if last_ch_eval < first_ch_eval:
            print("✓ 模型在中文任务上持续改进")
        else:
            print("! 注意：可能存在轻微遗忘")

print("\n【英文批次分析】")
if en_batches:
    print(f"批次数量: {len(en_batches)}")
    avg_train_loss = np.mean([d["train_losses"][-1]["loss"] for d in en_batches if d["train_losses"]])
    avg_eval_loss = np.mean([d["eval_losses"][-1]["loss"] for d in en_batches if d["eval_losses"]])
    print(f"平均训练损失: {avg_train_loss:.4f}")
    print(f"平均验证损失: {avg_eval_loss:.4f}")
    
    first_en_eval = en_batches[0]["eval_losses"][-1]["loss"] if en_batches[0]["eval_losses"] else None
    last_en_eval = en_batches[-1]["eval_losses"][-1]["loss"] if en_batches[-1]["eval_losses"] else None
    if first_en_eval and last_en_eval:
        print(f"验证损失变化: {first_en_eval:.4f} → {last_en_eval:.4f}")

# 5. 总结
print("\n" + "="*80)
print("五、总结")
print("="*80)

total_batches = len(batch_data)
completed_batches = sum(1 for d in batch_data if d["train_losses"])

print(f"已完成批次: {completed_batches}/{total_batches}")

# 检查训练稳定性
stable_batches = 0
for data in batch_data:
    grad_norms = data["grad_norms"]
    if grad_norms:
        gn_array = np.array([g["grad_norm"] for g in grad_norms])
        if np.max(gn_array) < 10.0 and np.min(gn_array) > 0.1:
            stable_batches += 1

print(f"训练稳定性: {stable_batches}/{total_batches} 批次稳定")

# 评估EWC效果
if len(batch_data) >= 2:
    first_eval = batch_data[0]["eval_losses"][-1]["loss"] if batch_data[0]["eval_losses"] else None
    last_eval = batch_data[-1]["eval_losses"][-1]["loss"] if batch_data[-1]["eval_losses"] else None
    if first_eval and last_eval:
        if last_eval < first_eval * 1.5:  # 验证损失没有大幅上升
            print("✓ EWC有效：验证损失保持稳定，未出现明显遗忘")
        else:
            print("! 注意：可能存在遗忘现象，建议调整EWC参数")

print("\n训练分析完成！")
