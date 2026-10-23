import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# ============ 配置 ============
BASE_DIR = "/root/autodl-tmp/fjy"
LANG_DIRS = {"ch": f"{BASE_DIR}/ch", "en": f"{BASE_DIR}/en"}
FOLDERS = ["1", "2", "3"]
OUTPUT_DIR = f"{BASE_DIR}/train_data"

SYSTEM_PROMPT_CH = "从科技新闻中抽取知识图谱三元组。输出JSON格式，包含year、knowledge_type和triples数组。每个triple包含entity1、relation、entity2、relation_type、hierarchy_level。关系必须从预定义集合中选择：发布、提出、发明、创建、研发、开发、设计、制造、收购、投资、合作、应用、超越、导致、取代、融合、升级、基于、是、属于、包含、举办、担任、攻克、验证、探讨。只输出JSON，不要解释。"

SYSTEM_PROMPT_EN = "Extract knowledge graph triples from tech news. Output JSON with year, knowledge_type, and triples array. Each triple contains entity1, relation, entity2, relation_type, hierarchy_level. Relations must be from: release, propose, invent, create, develop, design, manufacture, acquire, invest, cooperate, apply, surpass, cause, replace, integrate, upgrade, based_on, is, belong_to, contain, host, serve_as, overcome, verify, discuss, has_feature. Output JSON only, no explanation."


def load_csvs(lang_dir, folders):
    all_data = []
    for folder in folders:
        folder_path = os.path.join(lang_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".csv"):
                continue
            filepath = os.path.join(folder_path, f)
            try:
                df = pd.read_csv(filepath, encoding="utf-8-sig")
            except Exception:
                try:
                    df = pd.read_csv(filepath, encoding="gb18030")
                except Exception as e:
                    print(f"跳过 {filepath}: {e}")
                    continue
            all_data.append(df)
            print(f"加载: {filepath} ({len(df)} 条)")
    return all_data


def normalize_columns(df):
    col_map = {}
    for c in df.columns:
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
        elif cl == "year":
            col_map[c] = "year"
        elif cl == "knowledge_type":
            col_map[c] = "knowledge_type"
    return df.rename(columns=col_map)


def filter_data(df):
    for c in ["content", "entity1", "relation", "entity2"]:
        if c not in df.columns:
            print(f"缺少必须列: {c}，当前列: {list(df.columns)}")
            return pd.DataFrame()
    df = df.dropna(subset=["content", "entity1", "relation", "entity2"])
    df = df[df["content"].str.len() >= 10]
    df = df[df["entity1"].str.strip() != ""]
    df = df[df["entity2"].str.strip() != ""]
    df = df[df["relation"].str.strip() != ""]
    df = df[df["entity1"].str.len() <= 50]
    df = df[df["entity2"].str.len() <= 50]
    return df


def aggregate_samples(df):
    grouped = df.groupby("content", sort=False)
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
                "entity1": e1, "relation": rel, "entity2": e2,
                "relation_type": rt if rt != "nan" else "associative",
                "hierarchy_level": hl if hl != "nan" else "1"
            })
        if not triples:
            continue
        output_json = json.dumps({
            "year": year, "knowledge_type": kt, "triples": triples
        }, ensure_ascii=False)
        samples.append({"input": content[:1000], "output": output_json})
    return samples


def save_jsonl(samples, system_prompt, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            msg = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": s["input"]},
                    {"role": "assistant", "content": s["output"]}
                ]
            }
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


# ============ 按语言+批次分别处理 ============
for lang, lang_dir in LANG_DIRS.items():
    system_prompt = SYSTEM_PROMPT_CH if lang == "ch" else SYSTEM_PROMPT_EN
    print(f"\n{'='*50}")
    print(f"处理语言: {lang}")
    print(f"{'='*50}")

    for folder in FOLDERS:
        folder_path = os.path.join(lang_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"目录不存在，跳过: {folder_path}")
            continue

        csvs = load_csvs(lang_dir, [folder])
        if not csvs:
            print(f"  批次 {folder}: 无数据")
            continue

        df = pd.concat(csvs, ignore_index=True)
        df = normalize_columns(df)
        df = filter_data(df)
        if df.empty:
            continue

        samples = aggregate_samples(df)
        print(f"  批次 {folder}: {len(samples)} 条样本")

        if len(samples) < 5:
            print(f"  样本太少，跳过")
            continue

        val_ratio = 0.05 if len(samples) > 200 else 0.1
        train_s, val_s = train_test_split(samples, test_size=val_ratio, random_state=42)

        batch_dir = f"{OUTPUT_DIR}/{lang}/batch_{folder}"
        save_jsonl(train_s, system_prompt, f"{batch_dir}/train.jsonl")
        save_jsonl(val_s, system_prompt, f"{batch_dir}/val.jsonl")
        print(f"  保存: {batch_dir}/train.jsonl ({len(train_s)}), val.jsonl ({len(val_s)})")

print("\n数据准备完成!")
print(f"输出目录结构:")
print(f"  {OUTPUT_DIR}/ch/batch_1/  ch/batch_2/  ch/batch_3/")
print(f"  {OUTPUT_DIR}/en/batch_1/  en/batch_2/  en/batch_3/")
