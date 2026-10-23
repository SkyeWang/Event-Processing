# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/2/20 13:20
# @Author : Ben Gao
# @File : Spider_Facebook.py
# @Software : PyCharm


from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re
import time
import Database


def Login(username, password, driver):
    try:
        login_url = "https://www.facebook.com/"
        driver.get(login_url)
        time.sleep(5)
        driver.find_element(by=By.NAME, value="email").send_keys(username)
        driver.find_element(by=By.NAME, value="pass").send_keys(password)
        driver.find_element(by=By.NAME, value="login").click()
        time.sleep(5)
    except Exception as e:
        print("请求失败。")


def ScrollToBottom(driver):
    js = "return action = document.body.scrollHeight"
    height = 0       # 初始化现在滚动条所在高度为0
    new_height = driver.execute_script(js)      # 当前窗口总高度
    k = 1
    while k < 15:
        # 将滚动条调整至页面底部
        for i in range(height, new_height, 100):
            driver.execute_script("window.scrollTo(0, {})".format(i))
            time.sleep(0.3)
            try:
                driver.find_element(by=By.XPATH, value="//div[text() = '展开']").click()
            except:
                continue
        height = new_height
        time.sleep(0.3)
        new_height = driver.execute_script(js)
        k = k + 1


# 根据标签解析网页内容
def ParseWebContent(driver):
    content = driver.page_source
    soup = BeautifulSoup(content, "lxml")
    try:
        nickname = soup.select("span.xt0psk2 > a.x1i10hfl.xjbqb8w.x6umtig.x1b1mbwd.xaqea5y.xav7gou.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xt0b8zv.xzsf02u.x1s688f")
        publish = soup.select("span.x4k7w5x.x1h91t0o.x1h9r5lt.xv2umb2.x1beo9mf.xaigb6o.x12ejxvf.x3igimt.xarpa2k.xedcshv.x1lytzrv.x1t2pt76.x7ja8zs.x1qrby5j.x1jfb8zj > a.x1i10hfl.xjbqb8w.x6umtig.x1b1mbwd.xaqea5y.xav7gou.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.x1heor9g.xt0b8zv.xo1l8bm > span")
        article = soup.select("div.xu06os2.x1ok221b > span.x193iq5w.xeuugli.x13faqbe.x1vvkbs.x1xmvt09.x1lliihq.x1s928wv.xhkezso.x1gmr53x.x1cpjm7i.x1fgarty.x1943h6x.xudqn12.x3x7a5m.x6prxxf.xvq8zen.xo1l8bm.xzsf02u.x1yc453h")
        for k in range(0, len(nickname)):
            nickname[k] = str(nickname[k])
            nickname[k] = re.sub("<.*?>", "", nickname[k])
        for k in range(0, len(article)):
            if k > len(article) - 1:
                break
            article[k] = str(article[k])
            article[k] = re.sub("<.*?>", "", article[k])
            if article[k] in nickname:
                del article[k]
        # f = open("data_Facebook.txt", "a", encoding="utf-8")
        # total = "共计：" + str(len(article)) + "条" + "\n"
        # f.write(total)
        for k in range(0, len(article)):
            publish[k] = str(publish[k])
            publish[k] = re.sub("<.*?>", "", publish[k])
            emoji = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
            article[k] = str(article[k])
            article[k] = re.sub("<.*?>", "", article[k])
            # article[k] = re.sub(emoji, "", article[k])
            article[k] = article[k].replace("'", "''")
            result = str(k + 1) + "\n" + "发布时间：" + str(publish[k]) + "\n" + "文章内容：" + str(article[k]) + "\n"
            # f.write(result)
            sql = "insert into Facebook values('" + str(publish[k]) + "', '" + str(article[k]) + "');"
            print(sql)
            if not Database.IsExist_Facebook(str(article[k])):
                Database.InsertToDB(sql)
        # f.close()
    except Exception as e:
        print("Error.")


def FacebookSpider(username, password):
    driver = webdriver.Chrome()
    Login(username, password, driver)
    file = open("Keywords-Facebook.txt", "r", encoding="utf-8")
    keywords = file.readlines()
    for keyword in keywords:
        keyword = keyword.replace("\n", "")
        print(keyword)
        data = {
            "q": keyword
        }
        url = "https://www.facebook.com/search/posts/?" + urlencode(data)
        driver.get(url)
        time.sleep(5)
        ScrollToBottom(driver)
        ParseWebContent(driver)


if __name__ == "__main__":
        FacebookSpider("abc@xxx.com", "password")


