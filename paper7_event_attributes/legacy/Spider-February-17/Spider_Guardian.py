# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/3/19 13:48
# @Author : Ben Gao
# @File : Spider_Guardian.py
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
        for page in range(1, 109):
            data = {
                "s": keyword,
                "page": page,
            }
            url = "https://guardian.ng/?" + urlencode(data)
            driver.get(url)
            time.sleep(10)
            content = driver.page_source
            soup = BeautifulSoup(content, "lxml")
            list = soup.select("span.title > a")
            for a_tag in list:
                if a_tag.get("href") not in url_list:
                    url_list.append(a_tag.get("href"))
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
        title = soup.select_one("h1.title").string
        title = title.strip()
        title = title.replace("'", "''")
        publish = soup.select_one("div.date").string
        publish = publish.replace("\n", "")
        publish = re.sub("\s+", " ", publish)
        publish = publish.strip()
        article_list = soup.select("article > p")
        article = ""
        for at in article_list:
            at = str(at)
            at = re.sub("<.*?>", "", at)        # 去除标签
            at = at.replace("\n", "")
            at = at.strip()
            article = article + at
        article = article.replace("'", "''")
        sql = "insert into Guardian values('" + title + "', '" + publish + "', '" + article + "');"
        print(sql)
        if not Database.IsExist_Guardian(article):
            Database.InsertToDB(sql)
    except Exception as e:
        print("Error.")


def GuardianSpider(keyword):
    chrome_options = Options()
    chrome_options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=chrome_options)
    url_list = GetUrl(keyword, driver)
    for url in url_list:
        content = GetWebContent(url, driver)
        ParseWebContent(content)


if __name__ == "__main__":
    GuardianSpider("5G")