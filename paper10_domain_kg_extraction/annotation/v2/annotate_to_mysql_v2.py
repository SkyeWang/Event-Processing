"""
高效标注脚本 v2.0 - 阿里云API + MySQL + 三元组拆分 + 知识图谱增强规范
新增规范：
- 实体抽取：最大匹配、类型优先级、实体归一化
- 关系抽取：原子关系定义表、方向不可逆
- 知识时序：时间戳锚定、状态变更、知识时效性、冲突检测、三元组去重
- 子结构：层次标注、关联子结构、动态演化、增量学习
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

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul')

# MySQL配置
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": "kg_annotation_v3",
    "charset": "utf8mb4"
}

# 设置API
# 阿里云
client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# DeepSeek
deepseek_client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

# ============================================================
# 原子关系定义表 - 规则2.1：禁止自由生成关系词
# ============================================================
ATOMIC_RELATIONS = {
    # 基础关系
    "是": "是", "属于": "属于", "包含": "包含", "等于": "等于",
    # 动作关系
    "发布": "发布", "推出": "发布", "公布了": "发布", "推出了": "发布",
    "提出": "提出", "发明": "发明", "创建": "创建", "建立": "创建",
    "研发": "研发", "开发": "开发", "设计": "设计", "制造": "制造",
    "收购": "收购", "并购": "收购", "投资": "投资", "融资": "投资",
    "合作": "合作", "联合": "合作", "签约": "合作",
    "应用": "应用", "用于": "应用", "赋能": "应用", "落地": "应用",
    "超越": "超越", "优于": "优于", "领先": "超越", "超过": "超越",
    "导致": "导致", "造成": "导致", "引起": "导致", "引发": "导致",
    "基于": "基于", "依赖": "基于",
    # 时序关系
    "早于": "早于", "晚于": "晚于", "先于": "早于", "随后": "晚于",
    # 状态关系
    "取代": "取代", "替代": "取代", "替换": "取代",
    "融合": "融合", "结合": "融合", "整合": "融合",
    "升级": "升级", "更新": "升级", "迭代": "升级",
}

# 实体归一化映射 - 规则1.3：同义异构实体统一映射
ENTITY_NORMALIZATION = {
    "LLM": "大语言模型",
    "大模型": "大语言模型",
    "大型语言模型": "大语言模型",
    "ChatGPT": "ChatGPT",
    "GPT-4": "GPT-4",
    "GPT4": "GPT-4",
    "transformer": "Transformer",
    "Transformer模型": "Transformer",
    "BERT": "BERT",
    "SVM": "SVM",
    "卷积神经网络": "CNN",
    "CNN模型": "CNN",
    "循环神经网络": "RNN",
    "RNN模型": "RNN",
    "长短期记忆网络": "LSTM",
    "LSTM模型": "LSTM",
    "生成式AI": "生成式AI",
    "AIGC": "AIGC",
    "人工智能": "人工智能",
    "机器学习": "机器学习",
    "深度学习": "深度学习",
}

# 知识时效类型 - 规则3.3
KNOWLEDGE_TYPES = {
    "static": "静态知识",    # 长期有效
    "dynamic": "动态知识",   # 随时间更新
    "obsolete": "过时知识",  # 已被替代
}

# 演化类型 - 规则4.3
EVOLUTION_TYPES = {
    "add": "新增",
    "strengthen": "强化",
    "weaken": "弱化",
    "disappear": "消失",
}

# 关系类型 - 规则4.2
RELATION_TYPES = {
    "hierarchical": "层次关系",
    "causal": "因果关系",
    "temporal": "时序关系",
    "associative": "关联关系",
}

# 冲突类型 - 规则3.4
CONFLICT_TYPES = {
    "temporal": "时间冲突",
    "state": "状态冲突",
    "logical": "逻辑冲突",
}

SYSTEM_PROMPT = """从科技新闻中抽取知识图谱三元组，严格遵循以下规范：

【标签格式】
- 头实体：用 <js>...</js> 包裹
- 关系：用 <R>...</R> 包裹
- 尾实体：用 <cp>...</cp> 包裹

【实体类型（可嵌套，遵循类型优先级规则1.2）】
- <gs>公司</gs> <cp>产品</cp> <js>技术</js> <ly>领域</ly>
- <yw>业务</yw> <jg>机构</jg> <rw>人物</rw>
- <dd>地点</dd> <xm>项目</xm>

【原子关系集（规则2.1：必须从以下关系中选择）】
发布、提出、发明、创建、研发、开发、设计、制造、收购、投资、合作、应用、超越、导致、取代、融合、升级、基于、是、属于、包含、融合、结合

【实体抽取规范】
1.1 最大匹配：实体应尽可能覆盖完整语义单元
1.2 类型优先级：具体实例(产品/人物) > 抽象概念(技术/领域) > 通用实体(机构/地点)
1.3 归一化处理：LLM/大模型/大型语言模型 → 大语言模型

【关系抽取规范】
2.1 原子关系：必须从预定义原子关系中选择，禁止自由生成
2.2 方向不可逆：严格区分头尾实体方向

【时序与状态规范】
3.1 时间戳：识别文本中的年份（如"2023年"、"2024年"），标注到三元组
3.2 状态变更：重点标注表示状态改变的动词
3.3 知识时效：区分静态/动态/过时知识
3.5 去重：完全相同的三元组只保留一条

【输出格式（JSON数组）】
JSON的value中不要包含任何HTML标签，只输出纯文本！

{
  "year": "2024",
  "knowledge_type": "dynamic",
  "triples": [
    {
      "entity1": "实体1纯文本（不加标签）",
      "relation": "关系",
      "entity2": "实体2纯文本（不加标签）",
      "relation_type": "hierarchical/causal/temporal/associative",
      "hierarchy_level": "1/2/3"
    }
  ]
}

【示例】
原文：2024年3月，OpenAI与Figure公司合作推出了Figure 01人形机器人。
输出：
{
  "year": "2024",
  "knowledge_type": "dynamic",
  "triples": [
    {"entity1": "OpenAI", "relation": "合作", "entity2": "Figure公司", "relation_type": "associative", "hierarchy_level": "1"},
    {"entity1": "OpenAI", "relation": "发布", "entity2": "Figure 01人形机器人", "relation_type": "causal", "hierarchy_level": "1"}
  ]
}

只输出JSON结果，不要解释。如果无法抽取，输出{"year":"","knowledge_type":"static","triples":[]}。"""


def normalize_entity(entity):
    """规则1.3：实体归一化"""
    return ENTITY_NORMALIZATION.get(entity, entity)


def normalize_relation(relation):
    """规则2.1：关系标准化"""
    return ATOMIC_RELATIONS.get(relation, relation)


def extract_year(text):
    """规则3.1：从文本中提取年份"""
    year_patterns = [
        r'(\d{4})年',
        r'(\d{4})年\d{1,2}月',
        r'(\d{4})年\d{1,2}月\d{1,2}日',
    ]
    for pattern in year_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def dedupe_triples(triples):
    """规则3.5：三元组去重"""
    seen = set()
    result = []
    for t in triples:
        key = (normalize_entity(t["entity1"]), normalize_relation(t["relation"]), normalize_entity(t["entity2"]))
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        charset=MYSQL_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor
    )


def init_db():
    """初始化MySQL数据库和表"""
    import traceback
    try:
        print("连接MySQL服务器...")
        conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            charset=MYSQL_CONFIG["charset"]
        )
        print("创建数据库...")
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_CONFIG['database']}`")
        cursor.close()
        conn.close()

        print("连接数据库...")
        conn = get_db_connection()
        cursor = conn.cursor()

        # 创建标注数据表 - 增强版
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num INT NOT NULL,
                content TEXT,
                keyword VARCHAR(500),
                labeled_existing TEXT COMMENT '已有标注（原始三元组）',
                labeled TEXT COMMENT 'API标注（DeepSeek/Qwen返回的JSON）',
                year VARCHAR(10),
                knowledge_type VARCHAR(20) DEFAULT 'dynamic',
                domain_relevance VARCHAR(20) DEFAULT 'high',
                status VARCHAR(20) DEFAULT 'pending',
                confidence DECIMAL(3,2) DEFAULT 0.95,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_num (num),
                INDEX idx_year (year),
                INDEX idx_knowledge_type (knowledge_type)
            )
        """)
        print("创建 annotations 表（增强版）")

        # 创建三元组子表 - 增强版
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num INT NOT NULL,
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
                conflict_type VARCHAR(20),
                knowledge_type VARCHAR(20) DEFAULT 'dynamic',
                source VARCHAR(20) DEFAULT 'new_api',
                confidence DECIMAL(3,2) DEFAULT 0.95,
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_num (num),
                INDEX idx_year (year),
                INDEX idx_keyword (keyword),
                INDEX idx_source (source),
                INDEX idx_entity1_norm (entity1_normalized(100)),
                INDEX idx_relation_norm (relation_normalized),
                INDEX idx_entity2_norm (entity2_normalized(100))
            )
        """)
        print("创建 triples 表（增强版）")

        # 创建知识冲突记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_conflicts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num INT NOT NULL,
                entity1 VARCHAR(500),
                entity2 VARCHAR(500),
                conflict_type VARCHAR(20),
                conflict_description TEXT,
                resolution VARCHAR(50),
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_num (num)
            )
        """)
        print("创建 knowledge_conflicts 表")

        conn.commit()
        conn.close()
        print("数据库初始化完成！")

    except Exception as e:
        print(f"数据库初始化失败: {e}")
        traceback.print_exc()
        raise SystemExit(1)


def create_file_table(table_name):
    """创建按文件命名的表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 表名清理：替换特殊字符
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        if safe_name[0].isdigit():
            safe_name = 't_' + safe_name

        # annotations表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{safe_name}_annotations` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num INT NOT NULL,
                content TEXT,
                keyword VARCHAR(500),
                labeled_existing TEXT COMMENT '已有标注（原始三元组）',
                labeled TEXT COMMENT 'API标注（DeepSeek/Qwen返回的JSON）',
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

        # triples表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{safe_name}_triples` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                num INT NOT NULL,
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
                INDEX idx_year (year),
                INDEX idx_keyword (keyword),
                INDEX idx_source (source)
            )
        """)

        conn.commit()
        return safe_name

    except Exception as e:
        print(f"创建表失败 {table_name}: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def save_to_file_table(safe_name, num, content, keyword, labeled, source="new_api"):
    """保存到按文件命名的表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        parsed = parse_triples_from_json(labeled)
        year = parsed.get("year", "")
        knowledge_type = parsed.get("knowledge_type", "dynamic")
        triples = parsed.get("triples", [])

        if not year:
            year = extract_year(content)

        # 插入标注（写入 labeled，不覆盖 labeled_existing）
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

        # 删除旧三元组
        cursor.execute(f"DELETE FROM `{safe_name}_triples` WHERE num = %s", (num,))

        # 插入新三元组
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
            except Exception as e:
                print(f"  插入三元组失败(num={num}): {e}")

        conn.commit()
        return len(triples)

    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def import_to_file_table(safe_name, num, content, keyword, entity1, relation, entity2, source="existing"):
    """导入已有标注到按文件命名的表（保留原始标签）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        year = extract_year(content)

        # entity1/entity2 保留原始值（含标签）
        entity1_raw = entity1
        entity2_raw = entity2

        # 清理标签用于归一化
        entity1_clean = re.sub(r'<[^>]+>', '', entity1).strip()
        entity2_clean = re.sub(r'<[^>]+>', '', entity2).strip()
        relation_clean = re.sub(r'<[^>]+>', '', relation).strip()

        # 归一化
        e1_norm = normalize_entity(entity1_clean)
        e2_norm = normalize_entity(entity2_clean)
        rel_norm = normalize_relation(relation_clean)

        # 判断关系类型
        relation_type = "associative"
        if relation_clean in ["是", "属于", "包含"] or rel_norm in ["是", "属于", "包含"]:
            relation_type = "hierarchical"
        elif relation_clean in ["早于", "晚于", "先于", "随后"]:
            relation_type = "temporal"
        elif relation_clean in ["导致", "造成", "引起", "引发"]:
            relation_type = "causal"

        # 构造已有标注的文本表示
        existing_text = f"{entity1_raw} | {relation} | {entity2_raw}"

        # 插入标注（写入 labeled_existing，不覆盖 labeled）
        cursor.execute(f"""
            INSERT INTO `{safe_name}_annotations` (num, content, keyword, labeled_existing, year, status)
            VALUES (%s, %s, %s, %s, %s, 'done')
            ON DUPLICATE KEY UPDATE
                keyword = VALUES(keyword),
                labeled_existing = CONCAT(IFNULL(labeled_existing, ''), '\n', VALUES(labeled_existing)),
                year = CASE WHEN year = '' OR year IS NULL THEN VALUES(year) ELSE year END,
                status = 'done'
        """, (num, content, keyword, existing_text, year))

        # 删除旧三元组
        cursor.execute(f"DELETE FROM `{safe_name}_triples` WHERE num = %s AND source = %s", (num, source))

        # 插入新三元组（entity1/entity2 保留原始标签）
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
        except Exception as e:
            print(f"  插入三元组失败: {e}")
            return 0

    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def parse_triples_from_json(labeled_text):
    """从API返回的JSON解析三元组"""
    try:
        if not labeled_text or labeled_text.strip() == "":
            return {"year": "", "knowledge_type": "static", "triples": []}

        # 尝试提取JSON部分（可能有额外文本）
        json_match = re.search(r'\{[\s\S]*\}', labeled_text)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = labeled_text

        # 尝试解析JSON
        data = json.loads(json_str)
        year = data.get("year", "")
        knowledge_type = data.get("knowledge_type", "dynamic")
        triples = data.get("triples", [])

        # 归一化处理（保留标签，只在 normalized 字段清理）
        for t in triples:
            # entity1/entity2 保留原始标签
            entity1_raw = t.get("entity1", "").strip()
            entity2_raw = t.get("entity2", "").strip()
            relation_raw = t.get("relation", "").strip()

            # 清理标签用于归一化
            entity1 = re.sub(r'<[^>]+>', '', entity1_raw).strip()
            entity2 = re.sub(r'<[^>]+>', '', entity2_raw).strip()
            relation = re.sub(r'<[^>]+>', '', relation_raw).strip()

            # 归一化后的纯文本
            entity1_norm = normalize_entity(entity1)
            entity2_norm = normalize_entity(entity2)
            relation_norm = normalize_relation(relation)

            t["entity1"] = entity1_raw      # 保留标签
            t["entity1_normalized"] = entity1_norm  # 归一化纯文本
            t["entity2"] = entity2_raw       # 保留标签
            t["entity2_normalized"] = entity2_norm  # 归一化纯文本
            t["relation"] = relation_raw     # 保留原始关系
            t["relation_normalized"] = relation_norm  # 归一化关系
            t["relation_type"] = t.get("relation_type", "associative")
            t["hierarchy_level"] = t.get("hierarchy_level", 1)
            t["evolution_type"] = "add"
            t["confidence"] = t.get("confidence", 0.95)

        # 去重
        triples = dedupe_triples(triples)

        return {
            "year": year,
            "knowledge_type": knowledge_type,
            "triples": triples
        }
    except (json.JSONDecodeError, TypeError) as e:
        # 兼容旧格式：直接解析标注文本
        return parse_triples_from_text(labeled_text)


def parse_triples_from_text(labeled_text):
    """解析标注内容，提取三元组（兼容旧格式）"""
    if not labeled_text or labeled_text.strip() == "" or labeled_text == "无":
        return {"year": "", "knowledge_type": "static", "triples": []}

    triples = []
    year = extract_year(labeled_text)

    # 分割多个三元组
    parts = re.split(r'[；;\n]', labeled_text)

    for part in parts:
        if not part.strip():
            continue

        # 提取头实体
        entity1 = None
        for tag in ['js', 'gs', 'ly', 'yw', 'jg', 'rw', 'dd', 'xm']:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', part, re.DOTALL)
            if match:
                entity1 = match.group(1).strip()
                break

        # 提取关系
        relation_match = re.search(r'<R>(.*?)</R>', part)
        relation = relation_match.group(1).strip() if relation_match else None

        # 提取尾实体
        entity2_match = re.search(r'<cp>(.*?)</cp>', part, re.DOTALL)
        entity2 = entity2_match.group(1).strip() if entity2_match else None

        if entity1 and relation and entity2:
            entity1_norm = normalize_entity(entity1)
            entity2_norm = normalize_entity(entity2)
            relation_norm = normalize_relation(relation)

            triples.append({
                "entity1": entity1,
                "entity1_normalized": entity1_norm,
                "relation": relation,
                "relation_normalized": relation_norm,
                "entity2": entity2,
                "entity2_normalized": entity2_norm,
                "relation_type": "associative",
                "hierarchy_level": 1,
                "evolution_type": "add",
                "confidence": 0.95
            })

    # 去重
    triples = dedupe_triples(triples)

    return {
        "year": year,
        "knowledge_type": "dynamic",
        "triples": triples
    }


def call_api(text, model="qwen-plus"):
    """调用API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:1000]}
            ],
            temperature=0.1,
            max_tokens=500,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "Arrearage" in error_msg or "overdue" in error_msg.lower():
            return "[账户欠费]"
        return f"错误: {error_msg[:50]}"


def call_deepseek_api(text, model="deepseek-chat"):
    """调用DeepSeek API"""
    try:
        response = deepseek_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:1000]}
            ],
            temperature=0.1,
            max_tokens=500,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "insufficient" in error_msg.lower() or "quota" in error_msg.lower():
            return "[账户欠费]"
        return f"错误: {error_msg[:50]}"


def check_conflicts(cursor, num, entity1, entity2):
    """规则3.4：检测知识冲突"""
    conflicts = []

    # 查询已有的相同实体对
    cursor.execute("""
        SELECT * FROM triples
        WHERE entity1_normalized = %s AND entity2_normalized = %s
    """, (normalize_entity(entity1), normalize_entity(entity2)))

    existing = cursor.fetchall()

    if len(existing) > 1:
        # 存在多个三元组，检查冲突
        for i, e1 in enumerate(existing):
            for e2 in existing[i+1:]:
                if e1['relation_normalized'] != e2['relation_normalized']:
                    conflicts.append({
                        "type": "logical",
                        "description": f"关系冲突: '{e1['relation_normalized']}' vs '{e2['relation_normalized']}'",
                        "source1": e1['id'],
                        "source2": e2['id']
                    })
                if e1['year'] and e2['year'] and e1['year'] != e2['year']:
                    conflicts.append({
                        "type": "temporal",
                        "description": f"时间冲突: {e1['year']} vs {e2['year']}",
                        "source1": e1['id'],
                        "source2": e2['id']
                    })

    return conflicts


def save_to_db(num, content, keyword, labeled, domain="tech", source="new_api"):
    """保存API标注结果到数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 解析三元组
        parsed = parse_triples_from_json(labeled)
        year = parsed.get("year", "")
        knowledge_type = parsed.get("knowledge_type", "dynamic")
        triples = parsed.get("triples", [])

        # 提取内容中的年份（如果API没有返回）
        if not year:
            year = extract_year(content)

        # 插入或更新标注数据（API结果写labeled，不覆盖labeled_existing）
        cursor.execute("""
            INSERT INTO annotations (num, content, keyword, labeled, year, knowledge_type, domain_relevance, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'done')
            ON DUPLICATE KEY UPDATE
                labeled = VALUES(labeled),
                keyword = CASE WHEN keyword = '' OR keyword IS NULL THEN VALUES(keyword) ELSE keyword END,
                year = CASE WHEN year = '' OR year IS NULL THEN VALUES(year) ELSE year END,
                knowledge_type = VALUES(knowledge_type),
                status = 'done'
        """, (num, content, keyword, labeled, year, knowledge_type, domain))

        # 先删除该num的旧三元组
        cursor.execute("DELETE FROM triples WHERE num = %s", (num,))

        # 插入新的三元组
        triple_inserted = 0
        for t in triples:
            try:
                cursor.execute("""
                    INSERT INTO triples (num, entity1, entity1_normalized, relation, relation_normalized,
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
                triple_inserted += 1
            except pymysql.err.IntegrityError:
                pass
            except Exception as e:
                print(f"  插入三元组失败: {e}, triple={t}")

        conn.commit()
        return triple_inserted

    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def load_newdata_files(data_dir="newdata"):
    """加载newdata目录下所有CSV文件（已标注数据）"""
    all_data = []

    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return pd.DataFrame()

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    # 添加来源字段
                    category = os.path.basename(os.path.dirname(file_path))
                    df['category'] = category
                    df['source_file'] = file
                    all_data.append(df)
                    print(f"加载: {file_path} ({len(df)} 条)")
                except Exception as e:
                    print(f"加载失败 {file_path}: {e}")

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        # 确保列名一致
        if 'num' in result.columns:
            result = result.rename(columns={'num': 'id'})
        return result
    return pd.DataFrame()


def import_existing_annotations_to_db(num, content, keyword, entity1, relation, entity2, category="tech"):
    """直接导入已有的标注三元组到数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        year = extract_year(content)

        existing_text = f"{entity1} | {relation} | {entity2}"

        # 插入或更新标注数据（已有标注写labeled_existing，不覆盖labeled）
        cursor.execute("""
            INSERT INTO annotations (num, content, keyword, labeled_existing, year, knowledge_type, domain_relevance, status)
            VALUES (%s, %s, %s, %s, %s, 'dynamic', 'high', 'done')
            ON DUPLICATE KEY UPDATE
                keyword = VALUES(keyword),
                labeled_existing = CONCAT(IFNULL(labeled_existing, ''), '\n', VALUES(labeled_existing)),
                year = CASE WHEN year = '' OR year IS NULL THEN VALUES(year) ELSE year END,
                status = 'done'
        """, (num, content, keyword, existing_text, year))

        # 先删除该num的旧三元组（只删existing的）
        cursor.execute("DELETE FROM triples WHERE num = %s AND source = 'existing'", (num,))

        # 归一化
        e1_norm = normalize_entity(entity1)
        rel_norm = normalize_relation(relation)
        e2_norm = normalize_entity(entity2)

        # 判断关系类型
        relation_type = "associative"
        if relation in ["是", "属于", "包含"] or rel_norm in ["是", "属于", "包含"]:
            relation_type = "hierarchical"
        elif relation in ["早于", "晚于", "先于", "随后"]:
            relation_type = "temporal"
        elif relation in ["导致", "造成", "引起", "引发"]:
            relation_type = "causal"

        # 插入三元组
        try:
            cursor.execute("""
                INSERT INTO triples (num, entity1, entity1_normalized, relation, relation_normalized,
                                    entity2, entity2_normalized, year, keyword, relation_type, hierarchy_level,
                                    evolution_type, knowledge_type, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'existing', %s)
            """, (
                num,
                entity1,
                e1_norm,
                relation,
                rel_norm,
                entity2,
                e2_norm,
                year,
                keyword,
                relation_type,
                1,
                "add",
                "dynamic",
                0.95
            ))
            return 1
        except Exception as e:
            print(f"  插入失败: {e}")
            return 0

    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()
        return 0
    finally:
        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='知识图谱标注工具 v2.0')
    parser.add_argument("-i", "--input", default="newdata", help="输入目录(newdata)")
    parser.add_argument("-m", "--model", choices=["qwen-plus", "deepseek-chat"], default="deepseek-chat", help="模型")
    parser.add_argument("-c", "--concurrent", type=int, default=5, help="并发数")
    parser.add_argument("-s", "--start", type=int, default=1, help="起始编号")
    parser.add_argument("-e", "--end", type=int, default=None, help="结束编号")
    parser.add_argument("--file", default=None, help="单文件模式")
    parser.add_argument("--mode", choices=["existing", "api", "all"], default="all",
                        help="模式: existing=只导入已有数据, api=API重新标注, all=两种都做")
    args = parser.parse_args()

    # 初始化全局表
    print("初始化数据库...")
    init_db()

    # 收集所有CSV文件
    data_dir = args.input
    csv_files = []

    if args.file:
        # 单文件模式
        csv_files = [(args.file, os.path.basename(args.file).replace('.csv', ''))]
    else:
        # 目录模式 - 遍历所有CSV文件
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    table_name = file.replace('.csv', '')
                    csv_files.append((file_path, table_name))

    if not csv_files:
        print("没有找到CSV文件，退出")
        return

    print(f"找到 {len(csv_files)} 个文件")
    print("=" * 60)

    # 统计汇总
    total_files = len(csv_files)
    total_docs_existing = 0
    total_triples_existing = 0
    total_docs_api = 0
    total_triples_api = 0

    for idx, (file_path, table_name) in enumerate(csv_files, 1):
        print(f"\n[{idx}/{total_files}] 处理文件: {table_name}")
        print("-" * 50)

        # 创建文件专属表
        safe_name = create_file_table(table_name)
        if not safe_name:
            print(f"  跳过: 表创建失败")
            continue

        # 读取文件
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gb18030')
            except Exception as e:
                print(f"  读取失败(gb18030): {e}")
                continue
        except Exception as e:
            print(f"  读取失败: {e}")
            continue

        if 'num' in df.columns:
            df = df.rename(columns={'num': 'id'})

        if df.empty or 'content' not in df.columns:
            print(f"  跳过: 无数据")
            continue

        if 'id' not in df.columns:
            print(f"  跳过: 缺少 id/num 列")
            continue

        # 筛选范围
        if args.end:
            df = df[(df['id'] >= args.start) & (df['id'] <= args.end)]
        else:
            df = df[df['id'] >= args.start]

        if df.empty:
            print(f"  跳过: 无有效数据")
            continue

        print(f"  数据: {len(df)} 条")

        # 模式1: 导入已有标注
        if args.mode in ["existing", "all"]:
            print(f"  [existing] 导入已有标注...")
            cnt_doc, cnt_triple = import_existing_to_file(df, safe_name)
            total_docs_existing += cnt_doc
            total_triples_existing += cnt_triple
            print(f"    -> {cnt_doc}文档, {cnt_triple}三元组")

        # 模式2: API重新标注
        if args.mode in ["api", "all"]:
            print(f"  [api] API标注 (model={args.model})...")
            cnt_doc, cnt_triple = process_api_for_file(df, safe_name, args)
            total_docs_api += cnt_doc
            total_triples_api += cnt_triple
            print(f"    -> {cnt_doc}文档, {cnt_triple}三元组")

    # 最终汇总
    print("\n" + "=" * 60)
    print("=== 最终统计 ===")
    print(f"处理文件: {total_files} 个")
    print(f"\n[已有标注] 文档: {total_docs_existing}, 三元组: {total_triples_existing}")
    print(f"[API标注] 文档: {total_docs_api}, 三元组: {total_triples_api}")

    # 列出所有创建的表
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [list(t.values())[0] for t in cursor.fetchall() if '_triples' in list(t.values())[0] or '_annotations' in list(t.values())[0]]
    print(f"\n创建的表: {len(tables)} 个")
    for t in sorted(tables):
        print(f"  - {t}")
    conn.close()


def import_existing_to_file(df, safe_name):
    """导入已有标注到文件专属表"""
    total = len(df)
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
            print(f"  导入行失败(id={row.get('id', '?')}): {e}")

    if skipped > 0:
        print(f"    跳过: {skipped} 条")
    return done, triple_count


def process_api_for_file(df, safe_name, args):
    """使用API标注文件专属表"""
    contents = df["content"].astype(str).tolist()
    nums = df["id"].tolist()
    keywords = (df["keyword"].astype(str).tolist() if "keyword" in df.columns
                else [""] * len(df))

    total = len(contents)

    # 测试API
    if args.model == "deepseek-chat":
        test_result = call_deepseek_api(contents[0])
    else:
        test_result = call_api(contents[0], args.model)

    if "账户欠费" in test_result:
        print(f"    错误: API账户欠费")
        return 0, 0

    done = 0
    triple_count = 0
    start_time = time.time()

    api_func = call_deepseek_api if args.model == "deepseek-chat" else call_api

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(api_func, c): i
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
                    print(f"    进度: {done}/{total} ({done/total*100:.0f}%) | {speed:.0f}条/分钟")
                    sys.stdout.flush()
            except Exception as e:
                print(f"    API标注失败(num={nums[i]}): {e}")

    return done, triple_count


if __name__ == '__main__':
    main()
