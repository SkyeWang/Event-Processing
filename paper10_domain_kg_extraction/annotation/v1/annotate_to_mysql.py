"""
高效标注脚本 - 阿里云API + MySQL实时写入 + 三元组拆分
"""
import os
import sys
import time
import re
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
    "database": "kg_annotation2",
    "charset": "utf8mb4"
}

# 设置API
client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """从新闻中抽取知识图谱三元组。

标签格式：
- 头实体：用 <js>...</js> 包裹
- 关系：用 <R>...</R> 包裹
- 尾实体：用 <cp>...</cp> 包裹

实体类型（可嵌套）：
- <gs>公司</gs> <cp>产品</cp> <js>技术</js> <ly>领域</ly>
- <yw>业务</yw> <jg>机构</jg> <rw>人物</rw>
- <dd>地点</dd> <xm>项目</xm>

示例：
原文：傅利叶创始人顾捷表示将坚持技术创新。
标注：<js><rw>顾捷</rw></js><R>是</R><cp><gs>傅利叶</gs>创始人</cp>

原文：特斯拉发布Optimus Gen-2机器人。
标注：<js><gs>特斯拉</gs></js><R>发布</R><cp><sb>Optimus Gen-2</sb></cp>

只输出标注结果，不要解释。如果无法抽取，输出"无"。"""


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
    # 先创建数据库
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        charset=MYSQL_CONFIG["charset"]
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
    cursor.close()
    conn.close()

    # 获取连接创建表
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建标注数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            num INT NOT NULL,
            content TEXT,
            keyword VARCHAR(500),
            labeled TEXT,
            status VARCHAR(20) DEFAULT 'pending',
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_num (num)
        )
    """)
    print("创建 annotations 表")

    # 创建三元组子表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triples (
            id INT AUTO_INCREMENT PRIMARY KEY,
            num INT NOT NULL,
            entity1 VARCHAR(500),
            relation VARCHAR(500),
            entity2 VARCHAR(500),
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_num (num)
        )
    """)
    print("创建 triples 表")

    conn.commit()
    conn.close()


def parse_triples(labeled_text):
    """解析标注内容，提取三元组（一个数据可能拆分成多个）"""
    if not labeled_text or labeled_text.strip() == "" or labeled_text == "无":
        return []

    triples = []

    # 分割多个三元组（用中文分号分隔）
    parts = re.split(r'[；;\n]', labeled_text)

    for part in parts:
        if not part.strip():
            continue

        # 提取头实体（尝试多种标签）
        entity1 = None
        for tag in ['js', 'gs', 'ly', 'yw', 'jg', 'rw', 'dd', 'xm']:
            match = re.search(rf'<{tag}>(.*?)</{tag}>', part)
            if match:
                entity1 = match.group(1).strip()
                break

        # 提取关系
        relation_match = re.search(r'<R>(.*?)</R>', part)
        relation = relation_match.group(1).strip() if relation_match else None

        # 提取尾实体
        entity2_match = re.search(r'<cp>(.*?)</cp>', part)
        entity2 = entity2_match.group(1).strip() if entity2_match else None

        if entity1 and relation and entity2:
            triples.append({
                "entity1": entity1,
                "relation": relation,
                "entity2": entity2
            })

    return triples


def call_api(text, model="qwen3.5-plus"):
    """调用API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:800]}
            ],
            temperature=0.1,
            max_tokens=300,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "Arrearage" in error_msg or "overdue" in error_msg.lower():
            return "[账户欠费]"
        return f"错误: {error_msg[:50]}"


def save_to_db(num, content, keyword, labeled):
    """保存标注结果到数据库"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 插入或更新标注数据
        cursor.execute("""
            INSERT INTO annotations (num, content, keyword, labeled, status)
            VALUES (%s, %s, %s, %s, 'done')
            ON DUPLICATE KEY UPDATE
                labeled = VALUES(labeled), keyword = VALUES(keyword), status = 'done'
        """, (num, content, keyword, labeled))

        # 解析三元组
        triples = parse_triples(labeled)

        # 先删除该num的旧三元组
        cursor.execute("DELETE FROM triples WHERE num = %s", (num,))

        # 插入新的三元组
        for t in triples:
            cursor.execute("""
                INSERT INTO triples (num, entity1, relation, entity2)
                VALUES (%s, %s, %s, %s)
            """, (num, t["entity1"], t["relation"], t["entity2"]))

        conn.commit()
    except Exception as e:
        print(f"数据库错误: {e}")
        conn.rollback()
    finally:
        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="alldata.csv", help="输入文件")
    parser.add_argument("-m", "--model", default="qwen-plus", help="模型")
    parser.add_argument("-c", "--concurrent", type=int, default=5, help="并发数")
    parser.add_argument("-s", "--start", type=int, default=1, help="起始编号")
    parser.add_argument("-e", "--end", type=int, default=None, help="结束编号")
    args = parser.parse_args()

    # 初始化数据库
    print("初始化数据库...")
    init_db()

    # 读取数据
    print(f"读取数据: {args.input}")
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    # 筛选范围
    if args.end:
        df = df[df['id'] >= args.start][df['id'] <= args.end]
    else:
        df = df[df['id'] >= args.start]

    contents = df["content"].astype(str).tolist()
    nums = df["id"].tolist()
    keywords = df.get("keyword", [""] * len(df)).astype(str).tolist()

    total = len(contents)
    print(f"总数据: {total} 条 (num: {nums[0]}-{nums[-1]})")

    # 测试API
    print("测试API...")
    test_result = call_api(contents[0])
    print(f"测试结果: {test_result[:100]}...")

    if "账户欠费" in test_result:
        print("\n!!! 错误: 阿里云账户欠费 !!!")
        return

    results = [None] * total
    start_time = time.time()
    done = 0

    print(f"\n开始处理 (并发={args.concurrent})...")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(call_api, c, args.model): i
            for i, c in enumerate(contents)
        }

        for f in as_completed(futures):
            i = futures[f]
            try:
                labeled = f.result()
                results[i] = labeled

                # 实时写入数据库
                save_to_db(nums[i], contents[i], keywords[i], labeled)

                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = done / elapsed * 60
                    eta = (total - done) / speed if speed > 0 else 0
                    print(f"进度: {done}/{total} ({done/total*100:.1f}%) | 速度: {speed:.0f}条/分钟 | 预计剩余: {eta:.0f}分钟")
                    sys.stdout.flush()
            except Exception as e:
                print(f"处理错误: {e}")
                results[i] = f"错误: {str(e)[:50]}"

    elapsed = time.time() - start_time
    print(f"\n完成! 耗时: {elapsed/60:.1f}分钟，速度: {total/elapsed*60:.0f}条/分钟")

    # 统计
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM annotations WHERE status='done'")
    done_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM triples")
    triple_count = cursor.fetchone()[0]
    conn.close()

    print(f"\n=== 统计 ===")
    print(f"已标注: {done_count} 条")
    print(f"三元组总数: {triple_count} 条")


if __name__ == "__main__":
    main()
