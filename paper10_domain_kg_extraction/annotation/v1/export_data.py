# -*- coding: utf-8 -*-
import pymysql
import pandas as pd

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": "kg_annotation2",
    "charset": "utf8mb4"
}

conn = pymysql.connect(**MYSQL_CONFIG)

# ?????б?
cursor = conn.cursor()
cursor.execute('SHOW TABLES')
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

# ???????????
df_anno = pd.read_sql("SELECT * FROM annotations", conn)
df_anno.to_csv("annotations_result.csv", index=False, encoding="utf-8-sig")
print(f"annotations: {len(df_anno)} rows")

# ?????????
df_triples = pd.read_sql("SELECT * FROM triples", conn)
df_triples.to_csv("triples_result.csv", index=False, encoding="utf-8-sig")
print(f"triples: {len(df_triples)} rows")

conn.close()
print("\nExported: annotations_result.csv and triples_result.csv")
