# 原附件历史爬虫/标注代码；与本次固定语料实验入口独立。
# 外部网页、数据库表及浏览器驱动须另行配置，详见本目录 README。
# -*- coding = utf-8 -*-
# @Time : 2023/2/17 13:03
# @Author : Ben Gao
# @File : Spider.py
# @Software : PyCharm

import requests
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import re
import utils
import time
from tqdm import tqdm
import socket

headers = {
    "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate",     # 删除br，br类型无法解压
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "max-age=0",
    "Cookie": "",  # 原附件的会话 Cookie 已移除；运行时自行配置。
}


# 获取网页代码
def GetWebContent(url):
    try:
        time.sleep(3)
        respond = requests.get(url=url, headers=headers, timeout=15)
        content = respond.text
        # print(content)
        respond.close()
    except requests.exceptions.RequestException as e:
        content = ""
        print("请求失败。", url)
    return content


# 根据标签解析网页内容
def ParseWebContent(content):
    soup = BeautifulSoup(content, "lxml")
    try:
        # data1 = soup.select_one("body > div:nth-child(2) > div > div > div.x9f619 x1n2onr6 x1ja2u2z > div.x78zum5 xdt5ytf x1n2onr6 x1ja2u2z > div.x78zum5 xdt5ytf x1n2onr6 > div.x78zum5 xdt5ytf x1n2onr6 xat3117 xxzkxad > div.x78zum5 xdt5ytf x10cihs4 x1t2pt76 x1n2onr6 x1ja2u2z > div.x9f619 xnz67gz x78zum5 x168nmei x13lgxp2 x5pf9jr xo71vjh x1uhb9sk x1plvlek xryxfnj x1c4vz4f x2lah0s xdt5ytf xqjyukv x1qjc9v5 x1oa3qoh x1qughib > div.xh8yej3 x1gryazu x10o80wk x14k21rp x1porb0y x17snn68 x6osk4m > section.x78zum5 xdt5ytf x1iyjqo2 x6ikm8r xg6iff7 > main.x78zum5 xdt5ytf x1iyjqo2 x182iqb8 xnz67gz > div._aa6b _ad9f _aa6d > div._aa6e > article._aa6a _aatb _aatd _aatf > div._ab8w  _ab94 _ab99 _ab9h _ab9m _ab9p _abcm > div._ab8w  _ab94 _ab99 _ab9f _ab9m _ab9p _abcm > div._ae1h > div._ae2s > div._ae5q _akdn > ul._a9z6 _a9za > div.x1qjc9v5 x6umtig x1b1mbwd xaqea5y xav7gou x9f619 x78zum5 xdt5ytf x2lah0s xk390pu xdj266r x11i5rnm xat24cr x1mh8g0r xexx8yu x4uap5 x18d9i69 xkhd6sd x1n2onr6 xggy1nq x11njtxf > li._a9zj _a9zl  _a9z5> div._a9zm > div._a9zn._a9zo > div._a9zr > div._a9zs > h1._aacl _aaco _aacu _aacx _aad7 _aade")
        data2 = soup.select_one("body > div:nth-child(2)")
        # print(data1)
        print(data2)
    except Exception as e:
        data1 = ""


if __name__ == "__main__":
    content = GetWebContent("https://www.instagram.com/p/Cor_q6drlhn/")
    ParseWebContent(content)
    # for num in tqdm(range(10)):
    #     # 构建请求数据
    #     data = {
    #
    #     }
    #     url = "https://www,baidu.com/s?" + urlencode(data)
    #     content = GetWebContent(url)