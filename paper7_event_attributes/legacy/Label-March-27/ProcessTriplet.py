# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:02
# @Author : Ben Gao
# @File : ProcessTriplet.py
# @Software : PyCharm


import time
import Database


def RebuildTriplet():
    f = open("Data/AIoT_Triplet.txt", "r", encoding="utf-8")
    triplets = f.readlines()
    f.close()
    new_triplets = []
    for triplet in triplets:
        item = triplet.split(",")
        try:
            item[0] = str(item[0]).lower().replace(" <", "<").replace("< ", "<").replace(" >", ">").replace("> ", ">").replace("\"", "").strip()
            item[1] = str(item[1]).lower().replace(" <", "<").replace("< ", "<").replace(" >", ">").replace("> ", ">").replace(" *", "*").replace("\"", "").strip()
            item[2] = str(item[2]).lower().replace(" <", "<").replace("< ", "<").replace(" >", ">").replace("> ", ">").replace("\"", "").strip()
            if not str(item[2]).endswith("*") and not str(item[2]) == "" and not str(item[0]).endswith("*"):
                if not str(item[1]).endswith("*"):
                    new_triplets.append(item[0] + "," + item[1] + "*," + item[2])
                else:
                    new_triplets.append(item[0] + "," + item[1] + "," + item[2])
        except Exception as e:
            continue
    f = open("New_Data/AIoT_Triplet.txt", "a", encoding="utf-8")
    for triplet in new_triplets:
        f.write(str(triplet) + "\n")
    f.close()


def RebuildEntity():
    f = open("Rebuild_Data/集成电路_Triplet.txt", "r", encoding="utf-8")
    triplets = f.readlines()
    f.close()
    entities = set()
    for triplet in triplets:
        item = triplet.split(",")
        entities.add(item[0])
        entities.add(item[2].replace("\n", ""))
    f = open("Rebuild_Data/集成电路_Entity.txt", "a", encoding="utf-8")
    for entity in entities:
        f.write(str(entity) + "\n")
    f.close()


def RebuildRelation():
    f = open("Rebuild_Data/集成电路_Triplet.txt", "r", encoding="utf-8")
    triplets = f.readlines()
    f.close()
    relations = set()
    for triplet in triplets:
        item = triplet.split(",")
        relations.add(item[1])
    f = open("Rebuild_Data/集成电路_Relation.txt", "a", encoding="utf-8")
    for relation in relations:
        f.write(str(relation) + "\n")
    f.close()


def Insert():
    f = open("New_Data/AIoT_Triplet.txt", "r", encoding="utf-8")
    triplets = f.readlines()
    f.close()
    entities = set()
    for triplet in triplets:
        item = triplet.split(",")
        sql = "insert into New_Triplets values('AIoT', '" + str(item[0].replace("'", "''")) + "', '" + str(item[1].replace("'", "''")) + "', '" + str(item[2].replace("\n", "").replace("'", "''")) + "');"
        print(sql)
        time.sleep(0.5)
        Database.InsertToDB(sql)


if __name__ == "__main__":
    RebuildTriplet()
    # RebuildEntity()
    # RebuildRelation()
    Insert()