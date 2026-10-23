# -*- coding: utf-8 -*-
import pymysql

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": "kg_annotation2",
    "charset": "utf8mb4"
}

conn = pymysql.connect(**MYSQL_CONFIG)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM annotations WHERE status='done'")
done = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM annotations")
total = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM triples")
triples = cursor.fetchone()[0]

conn.close()

print(f"Total records: {total}")
print(f"Done: {done}")
print(f"Triples: {triples}")
