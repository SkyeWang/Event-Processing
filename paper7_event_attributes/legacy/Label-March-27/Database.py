# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/2/27 14:38
# @Author : Ben Gao
# @File : Database.py
# @Software : PyCharm


import pymssql


# 连接数据库
def ConnectToDB():
    try:
        db = pymssql.connect("服务器名称", "用户名", "密码", "库名")
        return db
    except:
        print("连接失败。")


# 数据库插入
def InsertToDB(sql, db, cursor):
    # db = ConnectToDB()
    # cursor = db.cursor()
    try:
        cursor.execute(sql)
        db.commit()
        # print("插入成功。")
    except:
        db.rollback()


def IsExist_Ins(article):
    db = ConnectToDB()
    cursor = db.cursor()
    sql = "select * from Instagram where Article = '" + article + "';"
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        db.close()
        return False
    else:
        cursor.close()
        db.close()
        return True


def IsExist_Facebook(article):
    db = ConnectToDB()
    cursor = db.cursor()
    sql = "select * from Facebook where Article = '" + article + "';"
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        db.close()
        return False
    else:
        cursor.close()
        db.close()
        return True


def IsExist_CNN(article):
    db = ConnectToDB()
    cursor = db.cursor()
    sql = "select * from CNN where Article = '" + article + "';"
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        db.close()
        return False
    else:
        cursor.close()
        db.close()
        return True


def IsExist_Guardian(article):
    db = ConnectToDB()
    cursor = db.cursor()
    sql = "select * from Guardian where Article = '" + article + "';"
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        db.close()
        return False
    else:
        cursor.close()
        db.close()
        return True


def IsExist_Times(article):
    db = ConnectToDB()
    cursor = db.cursor()
    sql = "select * from Times where Article = '" + article + "';"
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        cursor.close()
        db.close()
        return False
    else:
        cursor.close()
        db.close()
        return True