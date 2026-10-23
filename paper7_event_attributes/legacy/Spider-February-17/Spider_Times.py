# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/3/21 18:14
# @Author : Ben Gao
# @File : Spider_Times.py
# @Software : PyCharm


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import time
import Database


def GetUrl(driver):
    try:
        url_list = []
        for page in range(151, 336):
            js = "document.getElementById('page_no').value = " + str(page) + ";"
            driver.execute_script(js)
            driver.find_element(by=By.NAME, value="search_txt").send_keys(Keys.RETURN)
            time.sleep(5)
            content = driver.page_source
            soup = BeautifulSoup(content, "lxml")
            list = soup.select("blockquote > a")
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
        title = soup.select_one("div.article_title").string
        title = title.replace("'", "''")
        publish = soup.select_one("span.pub_time").string
        article = soup.select_one("div.article_content")
        article = str(article)
        article = re.sub("<.*?>", "", article)
        article = article.replace("\n", "")
        article = article.replace("'", "''")
        sql = "insert into Times values('" + title + "', '" + publish + "', '" + article + "');"
        print(sql)
        if not Database.IsExist_Times(article):
            Database.InsertToDB(sql)
    except Exception as e:
        print("Error.")


def TimesSpider(keyword):
    driver = webdriver.Chrome()
    driver.get("https://search.globaltimes.cn/QuickSearchCtrl")
    time.sleep(5)
    driver.find_element(by=By.NAME, value="search_txt").send_keys(keyword)
    driver.find_element(by=By.NAME, value="search_txt").send_keys(Keys.RETURN)
    url_list = GetUrl(driver)
    for url in url_list:
        content = GetWebContent(url, driver)
        ParseWebContent(content)


if __name__ == "__main__":
    TimesSpider("5G")