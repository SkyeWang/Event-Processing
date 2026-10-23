import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# ============ 配置 ============
BASE_DIR = "/root/autodl-tmp/fjy/ch"
FOLDERS = ["1", "2", "3"]
OUTPUT_DIR = "/root/autodl-tmp/fjy/train_data"

SYSTEM_PROMPT = "从科技新闻中抽取知识图谱三元组。输出JSON格式，包含year、knowledge_type和triples数组。每个triple包含entity1、relation、entity2、relation_type、hierarchy_level。关系必须从预定义集合中选择：发布、提出、发明、创建、研发、开发、设计、制造、收购、投资、合作、应用、超越、导致、取代、融合、升级、基于、是、属于、包含、举办、担任、攻克、验证、探讨。只输出JSON，不要解释。"

# ============ 1. 读取所有CSV ============
all_data = []
for folder in FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"目录不存在，跳过: {folder_path}")
        continue
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(".csv"):
            filepath = os.path.join(folder_path, f)
            try:
                df = pd.read_csv(filepath, encoding="utf-8-sig")
            except:
                try:
                    df = pd.read_csv(filepath, encoding="gb18030")
                except Exception as e:
                    print(f"跳过 {filepath}: {e}")
                    continue
            all_data.append(df)
            print(f"加载: {filepath} ({len(df)} 条)")

if not all_data:
    print("没有读到任何数据，请检查路径和文件")
    exit(1)

df_all = pd.concat(all_data, ignore_index=True)
print(f"\n总行数: {len(df_all)}")
print(f"列名: {list(df_all.columns)}")

# ============ 2. 列名兼容处理 ============
col_map = {}
for c in df_all.columns:
    cl = c.lower().strip()
    if cl in ["content", "text", "sentence", "news"]:
        col_map[c] = "content"
    elif cl in ["entity1", "entity_1", "head_entity", "head"]:
        col_map[c] = "entity1"
    elif cl in ["entity2", "entity_2", "tail_entity", "tail"]:
        col_map[c] = "entity2"
    elif cl in ["relation", "rel", "predicate"]:
        col_map[c] = "relation"
    elif cl in ["relation_type", "rel_type"]:
        col_map[c] = "relation_type"
    elif cl in ["hierarchy_level", "level"]:
        col_map[c] = "hierarchy_level"
    elif cl in ["year"]:
        col_map[c] = "year"
    elif cl in ["knowledge_type"]:
        col_map[c] = "knowledge_type"

df_all = df_all.rename(columns=col_map)

# 检查必须列
for c in ["content", "entity1", "relation", "entity2"]:
    if c not in df_all.columns:
        print(f"缺少必须列: {c}，当前列: {list(df_all.columns)}")
        exit(1)

# ============ 3. 过滤无效数据 ============
df_valid = df_all.dropna(subset=["content", "entity1", "relation", "entity2"])
df_valid = df_valid[df_valid["content"].str.len() >= 10]
df_valid = df_valid[df_valid["entity1"].str.strip() != ""]
df_valid = df_valid[df_valid["entity2"].str.strip() != ""]
df_valid = df_valid[df_valid["relation"].str.strip() != ""]
# 过滤实体过长的（质量差）
df_valid = df_valid[df_valid["entity1"].str.len() <= 50]
df_valid = df_valid[df_valid["entity2"].str.len() <= 50]
print(f"有效行数: {len(df_valid)}")

# ============ 4. 按content聚合三元组 ============
grouped = df_valid.groupby("content", sort=False)

samples = []
for content, group in grouped:
    content = str(content).strip()
    if len(content) < 10:
        continue

    year = ""
    if "year" in group.columns:
        years = group["year"].dropna().astype(str).unique()
        years = [y for y in years if y and y != "nan"]
        year = years[0] if years else ""

    kt = "dynamic"
    if "knowledge_type" in group.columns:
        kts = group["knowledge_type"].dropna().astype(str).unique()
        kts = [k for k in kts if k and k != "nan"]
        kt = kts[0] if kts else "dynamic"

    triples = []
    seen = set()
    for _, row in group.iterrows():
        e1 = str(row["entity1"]).strip()
        rel = str(row["relation"]).strip()
        e2 = str(row["entity2"]).strip()

        key = (e1, rel, e2)
        if key in seen:
            continue
        seen.add(key)

        rt = str(row.get("relation_type", "associative"))
        hl = str(row.get("hierarchy_level", "1"))

        triples.append({
            "entity1": e1,
            "relation": rel,
            "entity2": e2,
            "relation_type": rt if rt != "nan" else "associative",
            "hierarchy_level": hl if hl != "nan" else "1"
        })

    if not triples:
        continue

    output_json = json.dumps({
        "year": year,
        "knowledge_type": kt,
        "triples": triples
    }, ensure_ascii=False)

    samples.append({
        "input": content[:1000],
        "output": output_json
    })

print(f"\n训练样本数（按文本去重聚合后）: {len(samples)}")

if len(samples) < 10:
    print("样本太少，请检查数据")
    exit(1)

# ============ 5. 划分训练/验证集 ============
val_ratio = 0.05 if len(samples) > 200 else 0.1
train_samples, val_samples = train_test_split(samples, test_size=val_ratio, random_state=42)
print(f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}")

# ============ 6. 转换为Qwen对话格式并保存 ============
def to_messages(sample):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]}
        ]
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(to_messages(s), ensure_ascii=False) + "\n")

with open(os.path.join(OUTPUT_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
    for s in val_samples:
        f.write(json.dumps(to_messages(s), ensure_ascii=False) + "\n")

print(f"\n保存完成:")
print(f"  {OUTPUT_DIR}/train.jsonl  ({len(train_samples)} 条)")
print(f"  {OUTPUT_DIR}/val.jsonl    ({len(val_samples)} 条)")

# 打印一条样例看看
print(f"\n===== 样例 =====")
print(f"输入: {train_samples[0]['input'][:200]}...")
print(f"输出: {train_samples[0]['output'][:300]}...")
