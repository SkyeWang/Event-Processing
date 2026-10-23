"""
English version - Knowledge Graph Annotation v2.0
- Aliyun Qwen API + MySQL + triple splitting + KG enhancement rules
- Input: enlish/ directory, CSV columns: id, text, domain, entity1, relation, entity2
- Same DB schema as annotate_to_mysql_v2.py, but stored in database `kg_annotation_en`
- IDs are BIGINT (tweet IDs can exceed INT range)
"""
import os
import sys
import time
import re
import json
import pymysql
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pandas as pd

if sys.platform == 'win32':
    os.system('chcp 65001 >nul')

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": "kg_annotation_en",
    "charset": "utf8mb4"
}

# Aliyun DashScope (Qwen)
client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============================================================
# Atomic relations (English) - rule 2.1: no free-form relations
# ============================================================
ATOMIC_RELATIONS = {
    # basic
    "is": "is", "is a": "is", "are": "is",
    "belongs to": "belongs_to", "part of": "belongs_to",
    "contains": "contains", "includes": "contains", "has": "contains",
    "equals": "equals", "equal to": "equals",
    # actions
    "releases": "release", "released": "release", "launches": "release",
    "launched": "release", "announces": "release", "announced": "release",
    "unveils": "release", "unveiled": "release", "publishes": "release",
    "proposes": "propose", "proposed": "propose",
    "invents": "invent", "invented": "invent",
    "creates": "create", "created": "create", "builds": "create",
    "built": "create", "establishes": "create", "established": "create",
    "develops": "develop", "developed": "develop", "designs": "develop",
    "designed": "develop", "manufactures": "develop", "produced": "develop",
    "acquires": "acquire", "acquired": "acquire", "buys": "acquire",
    "invests": "invest", "invested": "invest", "funds": "invest", "funded": "invest",
    "partners with": "cooperate", "collaborates with": "cooperate",
    "cooperates with": "cooperate", "signs": "cooperate",
    "applies": "apply", "applied": "apply", "used for": "apply",
    "used in": "apply", "uses": "apply", "enables": "apply",
    "surpasses": "surpass", "outperforms": "surpass", "exceeds": "surpass",
    "beats": "surpass", "leads": "surpass",
    "causes": "cause", "caused": "cause", "leads to": "cause",
    "results in": "cause", "triggers": "cause",
    "based on": "based_on", "depends on": "based_on", "relies on": "based_on",
    "replaces": "replace", "replaced": "replace", "supersedes": "replace",
    "merges with": "merge", "combines with": "merge", "integrates with": "merge",
    "integrates": "merge",
    "upgrades": "upgrade", "updated": "upgrade", "iterates": "upgrade",
    # temporal
    "precedes": "before", "before": "before", "earlier than": "before",
    "after": "after", "later than": "after", "follows": "after",
    # related-to
    "related to": "related_to", "associated with": "related_to",
    "connected to": "related_to", "linked to": "related_to",
    "located in": "located_in", "based in": "located_in",
    "feature": "has_feature", "feature*": "has_feature",
    "features": "has_feature",
}

ENTITY_NORMALIZATION = {
    "LLM": "Large Language Model",
    "LLMs": "Large Language Model",
    "Large language model": "Large Language Model",
    "large language model": "Large Language Model",
    "Large Language Models": "Large Language Model",
    "GPT-4": "GPT-4", "GPT4": "GPT-4",
    "ChatGPT": "ChatGPT",
    "transformer": "Transformer",
    "Transformer model": "Transformer",
    "BERT": "BERT",
    "CNN": "CNN", "convolutional neural network": "CNN",
    "RNN": "RNN", "recurrent neural network": "RNN",
    "LSTM": "LSTM",
    "AI": "AI", "artificial intelligence": "AI", "Artificial Intelligence": "AI",
    "ML": "Machine Learning", "machine learning": "Machine Learning",
    "DL": "Deep Learning", "deep learning": "Deep Learning",
    "generative AI": "Generative AI", "GenAI": "Generative AI",
    "AIGC": "AIGC",
}

SYSTEM_PROMPT = """You are extracting knowledge-graph triples from English tech news. Follow these rules strictly.

[Tag format]
- Head entity: wrapped in <js>...</js>
- Relation: wrapped in <R>...</R>
- Tail entity: wrapped in <cp>...</cp>

[Entity types (nestable, follow priority rule 1.2)]
- <gs>Company</gs> <cp>Product</cp> <js>Technology</js> <ly>Field</ly>
- <yw>Business</yw> <jg>Institution</jg> <rw>Person</rw>
- <dd>Location</dd> <xm>Project</xm>

[Atomic relation set (rule 2.1 - pick one, do NOT invent)]
release, propose, invent, create, develop, acquire, invest, cooperate,
apply, surpass, cause, replace, merge, upgrade, based_on,
is, belongs_to, contains, equals, before, after, related_to,
located_in, has_feature

[Entity rules]
1.1 Maximum match: entity should cover the full semantic unit.
1.2 Type priority: concrete instance (Product/Person) > abstract concept (Technology/Field) > generic (Institution/Location).
1.3 Normalization: LLM/large language model -> Large Language Model; AI/artificial intelligence -> AI.

[Relation rules]
2.1 Use only atomic relations from the set above.
2.2 Direction matters: head -> tail is not reversible.

[Temporal / state rules]
3.1 Timestamp: find year in the text (e.g. "2023", "in 2024") and store it.
3.2 State change: emphasize verbs indicating state transitions.
3.3 Knowledge type: static / dynamic / obsolete.
3.5 Deduplicate identical triples.

[Output format - JSON only]
Do NOT include HTML tags in the JSON values, only plain text.

{
  "year": "2024",
  "knowledge_type": "dynamic",
  "triples": [
    {
      "entity1": "plain text entity 1",
      "relation": "atomic_relation",
      "entity2": "plain text entity 2",
      "relation_type": "hierarchical/causal/temporal/associative",
      "hierarchy_level": "1/2/3"
    }
  ]
}

[Example]
Text: In March 2024, OpenAI partnered with Figure to launch the Figure 01 humanoid robot.
Output:
{
  "year": "2024",
  "knowledge_type": "dynamic",
  "triples": [
    {"entity1": "OpenAI", "relation": "cooperate", "entity2": "Figure", "relation_type": "associative", "hierarchy_level": "1"},
    {"entity1": "OpenAI", "relation": "release", "entity2": "Figure 01 humanoid robot", "relation_type": "causal", "hierarchy_level": "1"}
  ]
}

Output ONLY the JSON object, no explanation. If nothing can be extracted, output {"year":"","knowledge_type":"static","triples":[]}."""


def normalize_entity(entity):
    if not entity:
        return entity
    return ENTITY_NORMALIZATION.get(entity, ENTITY_NORMALIZATION.get(entity.lower(), entity))


def normalize_relation(relation):
    if not relation:
        return relation
    r = relation.strip().rstrip('*').lower()
    return ATOMIC_RELATIONS.get(r, ATOMIC_RELATIONS.get(relation, relation))


def extract_year(text):
    if not text:
        return ""
    m = re.search(r'\b(19|20)\d{2}\b', str(text))
    return m.group(0) if m else ""


def dedupe_triples(triples):
    seen = set()
    result = []
    for t in triples:
        key = (normalize_entity(t["entity1"]), normalize_relation(t["relation"]), normalize_entity(t["entity2"]))
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def get_db_connection():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        charset=MYSQL_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor
    )


def init_db():
    import traceback
    try:
        print("Connecting to MySQL...")
        conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            charset=MYSQL_CONFIG["charset"]
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`")
        cursor.close()
        conn.close()
        print(f"Database `{MYSQL_CONFIG['database']}` ready.")
    except Exception as e:
        print(f"DB init failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)


def create_file_table(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        if safe_name[0].isdigit():
            safe_name = 't_' + safe_name

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{safe_name}_annotations` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num BIGINT NOT NULL,
                content TEXT,
                keyword VARCHAR(500),
                labeled_existing TEXT,
                labeled TEXT,
                year VARCHAR(10),
                knowledge_type VARCHAR(20) DEFAULT 'dynamic',
                domain_relevance VARCHAR(20) DEFAULT 'high',
                status VARCHAR(20) DEFAULT 'pending',
                confidence DECIMAL(3,2) DEFAULT 0.95,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_num (num),
                INDEX idx_year (year)
            )
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{safe_name}_triples` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num BIGINT NOT NULL,
                entity1 VARCHAR(500),
                entity1_normalized VARCHAR(255),
                relation VARCHAR(500),
                relation_normalized VARCHAR(100),
                entity2 VARCHAR(500),
                entity2_normalized VARCHAR(255),
                year VARCHAR(10),
                keyword VARCHAR(100),
                relation_type VARCHAR(20) DEFAULT 'associative',
                hierarchy_level INT DEFAULT 1,
                evolution_type VARCHAR(20) DEFAULT 'add',
                knowledge_type VARCHAR(20) DEFAULT 'dynamic',
                source VARCHAR(20) DEFAULT 'new_api',
                confidence DECIMAL(3,2) DEFAULT 0.95,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_num_entity (num, entity1_normalized, relation_normalized, entity2_normalized),
                INDEX idx_num (num),
                INDEX idx_year (year),
                INDEX idx_keyword (keyword),
                INDEX idx_source (source)
            )
        """)
        conn.commit()
        return safe_name
    except Exception as e:
        print(f"Create table failed {table_name}: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def parse_triples_from_json(labeled_text):
    try:
        if not labeled_text or labeled_text.strip() == "":
            return {"year": "", "knowledge_type": "static", "triples": []}

        json_match = re.search(r'\{[\s\S]*\}', labeled_text)
        json_str = json_match.group() if json_match else labeled_text

        data = json.loads(json_str)
        year = data.get("year", "")
        knowledge_type = data.get("knowledge_type", "dynamic")
        triples = data.get("triples", [])

        for t in triples:
            entity1_raw = str(t.get("entity1", "")).strip()
            entity2_raw = str(t.get("entity2", "")).strip()
            relation_raw = str(t.get("relation", "")).strip()

            entity1 = re.sub(r'<[^>]+>', '', entity1_raw).strip()
            entity2 = re.sub(r'<[^>]+>', '', entity2_raw).strip()
            relation = re.sub(r'<[^>]+>', '', relation_raw).strip()

            t["entity1"] = entity1_raw
            t["entity1_normalized"] = normalize_entity(entity1)
            t["entity2"] = entity2_raw
            t["entity2_normalized"] = normalize_entity(entity2)
            t["relation"] = relation_raw
            t["relation_normalized"] = normalize_relation(relation)
            t["relation_type"] = t.get("relation_type", "associative")
            t["hierarchy_level"] = t.get("hierarchy_level", 1)
            t["evolution_type"] = "add"
            t["confidence"] = t.get("confidence", 0.95)

        triples = dedupe_triples(triples)
        return {"year": year, "knowledge_type": knowledge_type, "triples": triples}
    except (json.JSONDecodeError, TypeError):
        return {"year": extract_year(labeled_text), "knowledge_type": "dynamic", "triples": []}


def save_to_file_table(safe_name, num, content, keyword, labeled, source="new_api"):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        parsed = parse_triples_from_json(labeled)
        year = parsed.get("year") or extract_year(content)
        knowledge_type = parsed.get("knowledge_type", "dynamic")
        triples = parsed.get("triples", [])

        cursor.execute(f"""
            INSERT INTO `{safe_name}_annotations` (num, content, keyword, labeled, year, knowledge_type, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'done')
            ON DUPLICATE KEY UPDATE
                labeled = VALUES(labeled),
                keyword = CASE WHEN keyword = '' OR keyword IS NULL THEN VALUES(keyword) ELSE keyword END,
                year = CASE WHEN year = '' OR year IS NULL THEN VALUES(year) ELSE year END,
                knowledge_type = VALUES(knowledge_type),
                status = 'done'
        """, (num, content, keyword, labeled, year, knowledge_type))

        cursor.execute(f"DELETE FROM `{safe_name}_triples` WHERE num = %s AND source = %s", (num, source))

        inserted = 0
        for t in triples:
            try:
                cursor.execute(f"""
                    INSERT INTO `{safe_name}_triples` (num, entity1, entity1_normalized, relation, relation_normalized,
                        entity2, entity2_normalized, year, keyword, relation_type, hierarchy_level,
                        evolution_type, knowledge_type, source, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    num,
                    t.get("entity1", ""),
                    t.get("entity1_normalized", ""),
                    t.get("relation", ""),
                    t.get("relation_normalized", ""),
                    t.get("entity2", ""),
                    t.get("entity2_normalized", ""),
                    year,
                    keyword,
                    t.get("relation_type", "associative"),
                    t.get("hierarchy_level", 1),
                    t.get("evolution_type", "add"),
                    knowledge_type,
                    source,
                    t.get("confidence", 0.95)
                ))
                inserted += 1
            except pymysql.err.IntegrityError:
                pass
            except Exception as e:
                print(f"  triple insert failed (num={num}): {e}")

        conn.commit()
        return inserted
    except Exception as e:
        print(f"DB error: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def import_to_file_table(safe_name, num, content, keyword, entity1, relation, entity2, source="existing"):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        year = extract_year(content)
        entity1_raw, entity2_raw = entity1, entity2
        entity1_clean = re.sub(r'<[^>]+>', '', entity1).strip()
        entity2_clean = re.sub(r'<[^>]+>', '', entity2).strip()
        relation_clean = re.sub(r'<[^>]+>', '', relation).strip()
        e1_norm = normalize_entity(entity1_clean)
        e2_norm = normalize_entity(entity2_clean)
        rel_norm = normalize_relation(relation_clean)

        relation_type = "associative"
        if rel_norm in ["is", "belongs_to", "contains"]:
            relation_type = "hierarchical"
        elif rel_norm in ["before", "after"]:
            relation_type = "temporal"
        elif rel_norm == "cause":
            relation_type = "causal"

        existing_text = f"{entity1_raw} | {relation} | {entity2_raw}"

        cursor.execute(f"""
            INSERT INTO `{safe_name}_annotations` (num, content, keyword, labeled_existing, year, status)
            VALUES (%s, %s, %s, %s, %s, 'done')
            ON DUPLICATE KEY UPDATE
                keyword = VALUES(keyword),
                labeled_existing = CONCAT(IFNULL(labeled_existing, ''), '\n', VALUES(labeled_existing)),
                year = CASE WHEN year = '' OR year IS NULL THEN VALUES(year) ELSE year END,
                status = 'done'
        """, (num, content, keyword, existing_text, year))

        try:
            cursor.execute(f"""
                INSERT INTO `{safe_name}_triples` (num, entity1, entity1_normalized, relation, relation_normalized,
                    entity2, entity2_normalized, year, keyword, relation_type, hierarchy_level,
                    evolution_type, knowledge_type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                num, entity1_raw, e1_norm, relation, rel_norm, entity2_raw, e2_norm,
                year, keyword, relation_type, 1, "add", "dynamic", source, 0.95
            ))
            conn.commit()
            return 1
        except pymysql.err.IntegrityError:
            conn.commit()
            return 0
        except Exception as e:
            print(f"  insert failed: {e}")
            conn.rollback()
            return 0
    except Exception as e:
        print(f"DB error: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def call_api(text, model="qwen-plus"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(text)[:1500]}
            ],
            temperature=0.1,
            max_tokens=800,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "Arrearage" in error_msg or "overdue" in error_msg.lower():
            return "[ACCOUNT_ARREARAGE]"
        return f"ERROR: {error_msg[:80]}"


def import_existing_to_file(df, safe_name):
    done = 0
    triple_count = 0
    skipped = 0
    for _, row in df.iterrows():
        try:
            num = int(row['id'])
            content = str(row['content']) if pd.notna(row['content']) else ""
            keyword = str(row['keyword']) if pd.notna(row.get('keyword')) else ""
            entity1 = row.get('entity1')
            relation = row.get('relation')
            entity2 = row.get('entity2')

            if pd.isna(entity1) or pd.isna(relation) or pd.isna(entity2):
                skipped += 1
                continue

            e1 = str(entity1).strip()
            rel = str(relation).strip()
            e2 = str(entity2).strip()
            if not e1 or not rel or not e2 or e1 == 'nan' or rel == 'nan' or e2 == 'nan':
                skipped += 1
                continue

            cnt = import_to_file_table(safe_name, num, content, keyword, e1, rel, e2, source="existing")
            triple_count += cnt
            done += 1
        except Exception as e:
            print(f"  import row failed (id={row.get('id', '?')}): {e}")
    if skipped > 0:
        print(f"    skipped: {skipped}")
    return done, triple_count


def process_api_for_file(df, safe_name, args):
    contents = df["content"].astype(str).tolist()
    nums = df["id"].tolist()
    keywords = (df["keyword"].astype(str).tolist() if "keyword" in df.columns
                else [""] * len(df))
    total = len(contents)

    test_result = call_api(contents[0], args.model)
    if "ACCOUNT_ARREARAGE" in test_result:
        print(f"    ERROR: API account arrearage")
        return 0, 0

    done = 0
    triple_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(call_api, c, args.model): i
            for i, c in enumerate(contents)
        }
        for f in as_completed(futures):
            i = futures[f]
            try:
                labeled = f.result()
                cnt = save_to_file_table(safe_name, nums[i], contents[i], keywords[i], labeled, source="new_api")
                triple_count += cnt
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = done / elapsed * 60 if elapsed > 0 else 0
                    print(f"    progress: {done}/{total} ({done/total*100:.0f}%) | {speed:.0f}/min")
                    sys.stdout.flush()
            except Exception as e:
                print(f"    API fail (num={nums[i]}): {e}")
    return done, triple_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='KG Annotation tool - English v2.0')
    parser.add_argument("-i", "--input", default="enlish", help="input directory")
    parser.add_argument("-m", "--model", default="qwen-plus",
                        choices=["qwen-plus", "qwen-turbo", "qwen-max"], help="Qwen model")
    parser.add_argument("-c", "--concurrent", type=int, default=10, help="concurrent requests")
    parser.add_argument("-s", "--start", type=int, default=0, help="start index (row offset)")
    parser.add_argument("-e", "--end", type=int, default=None, help="end index (row offset)")
    parser.add_argument("--file", default=None, help="single file mode")
    parser.add_argument("--mode", choices=["existing", "api", "all"], default="all",
                        help="existing=import labels only, api=call API, all=both")
    parser.add_argument("--limit", type=int, default=None, help="limit rows per file")
    args = parser.parse_args()

    print("Init DB...")
    init_db()

    data_dir = args.input
    csv_files = []
    if args.file:
        csv_files = [(args.file, os.path.basename(args.file).replace('.csv', ''))]
    else:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    # prefix with parent dir to disambiguate bbc_6g.csv etc. across topics
                    parent = os.path.basename(os.path.dirname(file_path))
                    table_name = f"{parent}_{file.replace('.csv', '')}"
                    csv_files.append((file_path, table_name))

    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} files")
    print("=" * 60)

    total_files = len(csv_files)
    total_docs_existing = 0
    total_triples_existing = 0
    total_docs_api = 0
    total_triples_api = 0

    for idx, (file_path, table_name) in enumerate(csv_files, 1):
        print(f"\n[{idx}/{total_files}] {table_name}")
        print("-" * 50)

        safe_name = create_file_table(table_name)
        if not safe_name:
            continue

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gb18030')
            except Exception as e:
                print(f"  read fail: {e}")
                continue
        except Exception as e:
            print(f"  read fail: {e}")
            continue

        # map English columns -> unified schema
        rename_map = {}
        if 'text' in df.columns:
            rename_map['text'] = 'content'
        if 'domain' in df.columns:
            rename_map['domain'] = 'keyword'
        if 'num' in df.columns and 'id' not in df.columns:
            rename_map['num'] = 'id'
        if rename_map:
            df = df.rename(columns=rename_map)

        if df.empty or 'content' not in df.columns or 'id' not in df.columns:
            print(f"  skip: missing content/id columns. have: {list(df.columns)}")
            continue

        # coerce id to int64 (bigint)
        try:
            df['id'] = pd.to_numeric(df['id'], errors='coerce').dropna().astype('int64')
            df = df.dropna(subset=['id'])
        except Exception as e:
            print(f"  id coerce warn: {e}")

        # slicing by row offset
        if args.start or args.end:
            df = df.iloc[args.start: (args.end if args.end else len(df))]
        if args.limit:
            df = df.head(args.limit)

        if df.empty:
            print("  skip: no rows after filter")
            continue

        print(f"  rows: {len(df)}")

        if args.mode in ["existing", "all"]:
            print(f"  [existing] importing labels...")
            cnt_doc, cnt_triple = import_existing_to_file(df, safe_name)
            total_docs_existing += cnt_doc
            total_triples_existing += cnt_triple
            print(f"    -> {cnt_doc} docs, {cnt_triple} triples")

        if args.mode in ["api", "all"]:
            print(f"  [api] model={args.model} concurrent={args.concurrent}")
            cnt_doc, cnt_triple = process_api_for_file(df, safe_name, args)
            total_docs_api += cnt_doc
            total_triples_api += cnt_triple
            print(f"    -> {cnt_doc} docs, {cnt_triple} triples")

    print("\n" + "=" * 60)
    print("=== Summary ===")
    print(f"files: {total_files}")
    print(f"[existing] docs={total_docs_existing}, triples={total_triples_existing}")
    print(f"[api]      docs={total_docs_api}, triples={total_triples_api}")


if __name__ == '__main__':
    main()
