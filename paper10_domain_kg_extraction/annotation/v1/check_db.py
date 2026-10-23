# -*- coding: utf-8 -*-
import pymysql

try:
    conn = pymysql.connect(host='localhost', user='root', password='123456', charset='utf8mb4')
    cursor = conn.cursor()
    cursor.execute('SHOW DATABASES')
    dbs = cursor.fetchall()
    print('Databases:', [d[0] for d in dbs])
    cursor.close()
    conn.close()
except Exception as e:
    print('Error:', e)
