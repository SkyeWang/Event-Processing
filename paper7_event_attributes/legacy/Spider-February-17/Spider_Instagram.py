# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/2/17 14:22
# @Author : Ben Gao
# @File : Spider_Instagram.py
# @Software : PyCharm


from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import quote
from bs4 import BeautifulSoup
import re
import time
import Database


def GetUrl(keyword, username, password, driver):
    try:
        url = "https://www.instagram.com/explore/tags/" + quote(keyword) + "/"
        driver.get(url)
        time.sleep(5)
        driver.find_element(by=By.CLASS_NAME, value="_ab8w ._ab94._ab99._ab9f._ab9m._ab9p._abb_._abcm").click()
        time.sleep(5)
        driver.find_element(by=By.NAME, value="username").send_keys(username)
        driver.find_element(by=By.NAME, value="password").send_keys(password)
        driver.find_element(by=By.CLASS_NAME, value="_acan._acap._acas._aj1-").click()
        time.sleep(5)
        driver.get(url)
        time.sleep(5)
        url_list = ScrollToBottom(driver)
        print(url_list)
        print(len(url_list))
    except Exception as e:
        url_list = []
        print("请求失败。")
    return url_list


def ScrollToBottom(driver):
    js = "return action = document.body.scrollHeight"
    height = 0       # 初始化现在滚动条所在高度为0
    new_height = driver.execute_script(js)      # 当前窗口总高度
    result = []
    while height < new_height:
        # 将滚动条调整至页面底部
        for i in range(height, new_height, 100):
            driver.execute_script("window.scrollTo(0, {})".format(i))
            time.sleep(0.3)
            content = driver.page_source
            soup = BeautifulSoup(content, "lxml")
            article_list = soup.select("div._aabd._aa8k._aanf > a")
            for a_tag in article_list:
                if a_tag.get("href") not in result:
                    result.append(a_tag.get("href"))
        height = new_height
        time.sleep(0.3)
        new_height = driver.execute_script(js)
    return result

# 获取网页代码
def GetWebContent(url, driver):
    try:
        driver.get(url)
        time.sleep(5)
        content = driver.page_source
    except Exception as e:
        content = ""
        print("请求失败。", url)
    return content


# 根据标签解析网页内容
def ParseWebContent(content):
    soup = BeautifulSoup(content, "lxml")
    try:
        nickname = soup.select_one("a.x1i10hfl.xjbqb8w.x6umtig.x1b1mbwd.xaqea5y.xav7gou.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz._acan._acao._acat._acaw._aj1-._a6hd").string
        nickname = nickname.replace("'", "''")
        publish = soup.select_one("time").get("datetime")
        # emoji = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
        tags = soup.select("h1 > a")
        for k in range(0, len(tags)):
            tags[k] = str(tags[k])
            tags[k] = re.sub("<.*?>", "", tags[k])
            # tags[k] = re.sub(emoji, "", tags[k])
            tags[k] = tags[k].replace("#", "")
        tags = str(tags).replace("'", "''")
        try:
            favour = soup.select_one("div._aacl._aaco._aacw._aacx._aada._aade > span").string
        except Exception as e:
            favour = "0"
        article = str(soup.select_one("h1"))
        article = re.sub("<.*?>", "", article)        # 去除标签
        # article = re.sub(emoji, "", article)
        article = article.replace("'", "''")
        # f = open("data_Instagram.txt", "a", encoding="utf-8")
        # result = "昵称：" + nickname + "\n" + "发布时间：" + publish + "\n" + "标签：" + str(tags) + "\n" + "点赞数：" + favour + "\n" + "文章内容：" + article + "\n"
        # f.write(result)
        # f.close()
        sql = "insert into Instagram values('" + nickname + "', '" + publish + "', '" + tags + "', '" + favour + "', '" + article + "');"
        print(sql)
        if not Database.IsExist_Ins(article):
            Database.InsertToDB(sql)
    except Exception as e:
        print("Error.")


def InstagramSpider(keyword, username, password):
    driver = webdriver.Chrome()
    url_list = GetUrl(keyword, username, password, driver)
    for url in url_list:
        content = GetWebContent("https://www.instagram.com" + url, driver)
        ParseWebContent(content)


if __name__ == "__main__":
    InstagramSpider("5G", "123456", "password")

