# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/3/28 15:26
# @Author : Ben Gao
# @File : main.py
# @Software : PyCharm


import re
import time
import Database
db = Database.ConnectToDB()
cursor = db.cursor()
from pattern.text.en import pluralize
from tqdm import tqdm


def BioesLabel(sequence: list, position: int, target):
    sequence[position] = target


def Label():
    start = 1
    end = 5557
    sql = "select entity1, relation, entity2 from New_Triplets;"
    cursor.execute(sql)
    triplets = cursor.fetchall()
    sql = "select entity1 from New_Triplets;"
    cursor.execute(sql)
    entities = cursor.fetchall()
    sql = "select entity2 from New_Triplets;"
    cursor.execute(sql)
    entities = entities + cursor.fetchall()
    entities = list(set(entities))
    entity_len = []
    for entity in entities:
        entity_len.append(len(str(entity).split(" ")))
    for i in range(0, len(entities) - 1):
        for j in range(0, len(entities) - i - 1):
            if entity_len[j] > entity_len[j + 1]:
                temp = entity_len[j]
                entity_len[j] = entity_len[j + 1]
                entity_len[j + 1] = temp
                temp = entities[j]
                entities[j] = entities[j + 1]
                entities[j + 1] = temp

    for r in tqdm(range(start, end), desc="进行中"):
        # sql = "select relation from Triplets;"
        # cursor.execute(sql)
        # relations = cursor.fetchall()
        # relations = list(set(relations))
        sql = "select Article from Facebook where id = " + str(r) + ";"
        cursor.execute(sql)
        article = cursor.fetchone()
        article = str(article).lower().strip("(").strip(")").strip(",").strip("'").strip('"')
        texts = article.split(".")
        while " " in texts:
            texts.remove(" ")
        for i in range(0, len(texts)):
            texts[i] = texts[i].strip().strip("?")


        seqRes = []         # 所有序列标记的list
        relRes = []         # 所有关系的list
        triRes = []         # 所有三元组/二元组的list
        enRes = []
        for text in texts:
            sequence = []
            entityList = []
            temp = text.split(" ")
            for i in range(len(temp)):
                sequence.append('O')
            for entity in entities:
                entity = entity[0]
                # flag = False
                # for e in entityList:            # 按照字符串长度降序匹配,子字符串不再匹配
                #     if e.find(entity) != -1:
                #         flag = True
                # if flag:
                #     continue
                # entityLabelList = re.findall("<.*>", entity)
                # entityLabel = ""
                # if len(entityLabelList) != 0:
                #     entityLabel = entityLabelList[0].replace("<", "").replace(">", "")
                pureEntity = re.sub("<.*>", "", entity)         # 不含标签的实体
                pureEntity = pureEntity.split(" ")
                try:
                    position = [i for i, x in enumerate(temp) if x == pureEntity[0]]
                    for i in range(0, len(position)):
                        p = position[i]
                        for pE in pureEntity:
                            if pE == temp[p] or pluralize(pE) == temp[p]:
                                p = p + 1
                            else:
                                position[i] = -1
                                break
                except Exception as e:
                    position = []
                for pos in position:
                    if pos != -1:
                        if entity not in entityList:
                            entityList.append(entity)
                        # flag2 = True
                        # for i in range(pos, len(pureEntity)):
                        #     if sequence[i] != 'O':
                        #         flag2 = False
                        # if flag2:
                        if len(pureEntity) == 1:
                            BioesLabel(sequence, pos, 'S')
                        else:
                            BioesLabel(sequence, pos, 'B')
                            for m in range(1, len(pureEntity) - 1):
                                BioesLabel(sequence, pos + m, 'I')
                            BioesLabel(sequence, pos + len(pureEntity) - 1, 'E')

            relationList = []
            tripletList = []
            tempRelation = ""
            for triplet in triplets:
                item = [triplet[0], triplet[1], triplet[2]]
                for i in item:
                    if i.find("*") != -1:           # 根据是否存在*判断是关系还是实体
                        tempRelation = i
                tempEntity = []
                for i in item:
                    if i != tempRelation:
                        tempEntity.append(i)
                if len(item) == 3 and len(tempEntity) == 2:         # 三元组 两个实体一个关系
                    if (tempEntity[0] in entityList) & (tempEntity[1] in entityList):
                        entityLabelList = re.findall("<.*>", tempEntity[0])
                        entityLabel0 = ""
                        if len(entityLabelList) != 0:
                            entityLabel0 = entityLabelList[0].replace("<", "").replace(">", "")
                        entityLabelList = re.findall("<.*>", tempEntity[1])
                        entityLabel1 = ""
                        if len(entityLabelList) != 0:
                            entityLabel1 = entityLabelList[0].replace("<", "").replace(">", "")
                        tempEntity[0] = re.sub("<.*>", "", tempEntity[0])
                        temp0 = tempEntity[0].split(" ")
                        po = temp.index(temp0[0])
                        for s in range(po, len(temp0)):
                            if entityLabel0 != "" and sequence[s].find("-") == -1:
                                sequence[s] = sequence[s] + "-" + entityLabel0
                        tempEntity[1] = re.sub("<.*>", "", tempEntity[1])
                        temp1 = tempEntity[1].split(" ")
                        po = temp.index(temp1[0])
                        for s in range(po, po + len(temp1)):
                            if entityLabel1 != "" and sequence[s].find("-") == -1:
                                sequence[s] = sequence[s] + "-" + entityLabel1
                        tripletList.append(triplet)
                        relationList.append(item[1])

            seqRes.append(sequence)
            enRes.append(entityList)
            triRes.append(tripletList)
            relationList = list(set(relationList))
            relRes.append(relationList)

        for i in range(0, len(texts)):
            try:
                sequence = ""
                for j in range(0, len(seqRes[i])):
                    sequence += str(seqRes[i][j]) + " "
                triple = ""
                for j in range(0, len(triRes[i])):
                    triple += "<" + str(triRes[i][j]) + ">"
                if len(str(texts[i])) > 50:
                    sql = "insert into Sentences values('" + str(texts[i]).replace("'", "''") + "', '" + sequence + "', '" + str(relRes[i]).replace("'", "''") + "', '" + triple.replace("'", "''") + "', '" + str(enRes[i]).replace("'", "''") + "', '" + str(r) + "', '手工');"
                    # print(sql)
                    Database.InsertToDB(sql, db, cursor)
                # time.sleep(0.1)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    Label()
    cursor.close()
    db.close()
