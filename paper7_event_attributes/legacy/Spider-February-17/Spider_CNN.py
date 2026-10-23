# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/3/14 14:10
# @Author : Ben Gao
# @File : Spider_CNN.py
# @Software : PyCharm


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re
import time
import Database


def GetUrl(keyword, driver):
    try:
        url_list = []
        begin = 0
        for page in range(1, 6):
            data = {
                "q": keyword,
                "from": begin,
                "size": 50,
                "page": page,
                "sort": "relevance",
                "types": "all",
                "section": ""
            }
            url = "https://edition.cnn.com/search?" + urlencode(data)
            driver.get(url)
            time.sleep(10)
            content = driver.page_source
            soup = BeautifulSoup(content, "lxml")
            list = soup.select("div.card.container__item.container__item--type-.__item.__item--type- > a")
            for a_tag in list:
                if a_tag.get("href") not in url_list:
                    url_list.append(a_tag.get("href"))
            begin = begin + 50
        print(url_list)
        print(len(url_list))
    except Exception as e:
        url_list = []
        print("请求失败。")
    return url_list


def GetWebContent(url, driver):
    try:
        driver.get(url)
        time.sleep(5)
        content = driver.page_source
    except Exception as e:
        content = ""
        print("请求失败。", url)
    return content


def ParseWebContent(content):
    soup = BeautifulSoup(content, "lxml")
    try:
        try:
            title = soup.select_one("h1.headline__text.inline-placeholder").string
        except:
            title = soup.select_one("h1.article-title.speakable").string
        title = title.replace("\n", "")
        title = title.strip()
        title = title.replace("'", "''")
        try:
            publish = soup.select_one("div.timestamp").string
        except:
            publish = soup.select_one("span.cnnDateStamp").string
        publish = publish.replace("\n", "")
        article_list = soup.select("p.paragraph.inline-placeholder")
        if len(article_list) == 0:
            article_list = soup.select("article.module > div#storycontent > div#storytext > p")
        article = ""
        for at in article_list:
            at = str(at)
            at = re.sub("<.*?>", "", at)        # 去除标签
            at = at.replace("\n", "")
            at = at.strip()
            article = article + at
        article = article.replace("'", "''")
        sql = "insert into CNN values('" + title + "', '" + publish + "', '" + article + "');"
        print(sql)
        if not Database.IsExist_CNN(article):
            Database.InsertToDB(sql)
    except Exception as e:
        print("Error.")


def CNNSpider(keyword):
    chrome_options = Options()
    chrome_options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=chrome_options)
    url_list = GetUrl(keyword, driver)
    for url in url_list:
        content = GetWebContent(url, driver)
        ParseWebContent(content)


if __name__ == "__main__":
    CNNSpider("US Entity List")