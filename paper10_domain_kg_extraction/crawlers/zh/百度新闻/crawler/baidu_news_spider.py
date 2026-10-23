# crawler/baidu_news_spider.py

import re
import time
import random
import csv
import hashlib
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# 随机 User-Agent 列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
]

# CSV 文件路径
CSV_FILE = 'baidunews2.csv'
CSV_ALL_FILE = 'baidunewsal2.csv'

class BaiduNewsScraper:
    def __init__(self, keywords, max_results=100, proxy=None):
        options = ChromeOptions()
        options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
        if proxy:
            options.add_argument(f"--proxy-server={proxy}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        self.wait = WebDriverWait(self.driver, 30)
        self.keywords = [kw.lower() for kw in keywords]
        self.max_results = max_results
        self.crawled_count = 0
        self.crawled_urls = set()
        self.should_stop = False
        
        # 加载已爬取的URL
        self._load_crawled_urls()
        
        # 初始化CSV文件
        self._init_csv_files()

    def _load_crawled_urls(self):
        """加载已爬取的URL用于去重"""
        if os.path.exists(CSV_FILE):
            try:
                with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('url'):
                            self.crawled_urls.add(row['url'])
            except Exception as e:
                print(f"加载已爬取URL失败: {e}")

    def _init_csv_files(self):
        """初始化CSV文件"""
        fieldnames = ['title', 'url', 'summary', 'source_time', 'matched_keywords', 'content', 'crawl_time']
        
        # 确保文件存在并包含表头
        for file_path in [CSV_FILE, CSV_ALL_FILE]:
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

    def simulate_mouse_movements(self):
        """模拟鼠标移动"""
        actions = ActionChains(self.driver)
        for _ in range(random.randint(3, 8)):
            actions.move_by_offset(random.randint(-30, 30), random.randint(-30, 30)).pause(random.uniform(0.1, 0.3))
        try:
            actions.perform()
        except Exception:
            pass

    def check_captcha(self):
        """检查验证码"""
        try:
            # 检查常见的验证码元素
            captcha_elements = [
                "//div[contains(@class, 'captcha')]",
                "//img[contains(@src, 'captcha')]",
                "//iframe[contains(@src, 'captcha')]",
                "//div[contains(text(), '验证')]"
            ]
            
            for xpath in captcha_elements:
                try:
                    element = self.driver.find_element(By.XPATH, xpath)
                    if element.is_displayed():
                        return True
                except:
                    continue
            return False
        except:
            return False

    def wait_for_captcha_manual(self):
        """等待手动处理验证码"""
        print("检测到验证码，请在浏览器中手动完成验证...")
        while self.check_captcha() and not self.should_stop:
            time.sleep(2)
        if not self.should_stop:
            print("验证码处理完成，继续爬取...")
            time.sleep(random.uniform(2, 4))

    def is_chinese_text(self, text, threshold=0.3):
        """检测是否为中文文本"""
        if not text:
            return False
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))
        return total_chars > 0 and (chinese_chars / total_chars) >= threshold

    def extract_news_content(self, url):
        """提取新闻详细内容"""
        try:
            # 打开新标签页
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            
            self.driver.get(url)
            time.sleep(random.uniform(2, 4))
            
            # 等待页面加载
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 尝试多种常见的内容选择器
            content_selectors = [
                'div.article-content',
                'div.content',
                'div.article-body',
                'div.post-content',
                'div.news-content',
                'div.detail-content',
                'article',
                'div.main-content',
                'div.article',
                '.article-body',
                '.content-text',
                '.article-main'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)
                    if len(content) > 100:  # 内容长度足够
                        break
            
            # 如果没有找到内容，尝试获取所有段落
            if not content or len(content) < 100:
                paragraphs = soup.find_all('p')
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # 清理内容：移除常见的无关文本
            if content:
                # 移除登录、注册等无关信息
                noise_patterns = [
                    r'下次自动登录',
                    r'其它账号登录',
                    r'新浪QQ微信',
                    r'老马点金：',
                    r'多空大动作',
                    r'淘气天尊：',
                    r'美元贬值',
                    r'常长亭：',
                    r'股心：',
                    r'皮球胖胖：',
                    r'热点精选：',
                    r'商业观察：',
                    r'登录|注册',
                    r'点击.*登录',
                    r'账号.*密码',
                    r'手机.*验证',
                    r'微信.*扫码',
                    r'QQ.*登录',
                    r'微博.*登录',
                    r'新浪.*登录',
                    r'返回.*首页',
                    r'跳转.*首页',
                    r'更多.*新闻',
                    r'相关.*推荐',
                    r'热门.*资讯',
                    r'最新.*消息',
                    r'点击.*查看',
                    r'阅读.*原文',
                    r'分享.*好友',
                    r'收藏.*文章',
                    r'点赞.*评论',
                    r'用户.*评论',
                    r'发表.*评论',
                    r'网友.*评论',
                    r'读者.*评论',
                    r'欢迎.*评论',
                    r'请.*登录',
                    r'登录.*后',
                    r'注册.*账号',
                    r'忘记.*密码',
                    r'找回.*密码',
                    r'验证.*手机',
                    r'手机.*验证码',
                    r'邮箱.*验证',
                    r'第三方.*登录',
                    r'快速.*登录',
                    r'安全.*登录',
                    r'记住.*密码',
                    r'自动.*登录',
                    r'下次.*登录',
                    r'立即.*注册',
                    r'免费.*注册',
                    r'新用户.*注册',
                    r'已有.*账号',
                    r'没有.*账号',
                    r'点击.*注册',
                    r'账号.*注册',
                    r'手机.*注册',
                    r'邮箱.*注册',
                    r'微信.*注册',
                    r'QQ.*注册',
                    r'微博.*注册',
                    r'新浪.*注册',
                    r'一键.*登录',
                    r'快速.*注册',
                    r'社交.*登录',
                    r'社交.*注册',
                    r'第三方.*账号',
                    r'授权.*登录',
                    r'OAuth.*登录',
                    r'开放.*平台',
                    r'应用.*授权',
                    r'用户.*授权',
                    r'隐私.*政策',
                    r'用户.*协议',
                    r'服务.*条款',
                    r'使用.*协议',
                    r'法律.*声明',
                    r'版权.*所有',
                    r'免责.*声明',
                    r'风险.*提示',
                    r'投资.*有风险',
                    r'入市.*需谨慎',
                    r'股市.*有风险',
                    r'投资.*谨慎',
                    r'市场.*风险',
                    r'交易.*风险',
                    r'理财.*风险',
                    r'金融.*风险',
                    r'产品.*风险',
                    r'平台.*风险',
                    r'技术.*风险',
                    r'操作.*风险',
                    r'系统.*风险',
                    r'网络.*风险',
                    r'安全.*风险',
                    r'数据.*风险',
                    r'信息.*风险',
                    r'内容.*风险',
                    r'服务.*风险',
                    r'使用.*风险',
                    r'访问.*风险',
                    r'连接.*风险',
                    r'传输.*风险',
                    r'存储.*风险',
                    r'处理.*风险',
                    r'分析.*风险',
                    r'展示.*风险',
                    r'分享.*风险',
                    r'传播.*风险',
                    r'转载.*风险',
                    r'引用.*风险',
                    r'链接.*风险',
                    r'跳转.*风险',
                    r'下载.*风险',
                    r'安装.*风险',
                    r'运行.*风险',
                    r'更新.*风险',
                    r'升级.*风险',
                    r'维护.*风险',
                    r'修复.*风险',
                    r'优化.*风险',
                    r'改进.*风险',
                    r'完善.*风险',
                    r'提升.*风险',
                    r'改善.*风险',
                    r'增强.*风险',
                    r'加强.*风险',
                    r'提高.*风险',
                    r'降低.*风险',
                    r'减少.*风险',
                    r'避免.*风险',
                    r'防止.*风险',
                    r'预防.*风险',
                    r'控制.*风险',
                    r'管理.*风险',
                    r'监控.*风险',
                    r'检测.*风险',
                    r'识别.*风险',
                    r'评估.*风险',
                    r'分析.*风险',
                    r'应对.*风险',
                    r'处理.*风险',
                    r'解决.*风险',
                    r'消除.*风险',
                    r'减轻.*风险',
                    r'缓解.*风险',
                    r'降低.*影响',
                    r'减少.*损失',
                    r'避免.*损失',
                    r'防止.*损失',
                    r'预防.*损失',
                    r'控制.*损失',
                    r'管理.*损失',
                    r'监控.*损失',
                    r'检测.*损失',
                    r'识别.*损失',
                    r'评估.*损失',
                    r'分析.*损失',
                    r'应对.*损失',
                    r'处理.*损失',
                    r'解决.*损失',
                    r'消除.*损失',
                    r'减轻.*损失',
                    r'缓解.*损失'
                ]
                
                # 应用过滤规则
                for pattern in noise_patterns:
                    content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                
                # 移除多余的空白字符
                content = re.sub(r'\s+', ' ', content).strip()
                
                # 移除过短的行（可能是标题或导航）
                lines = content.split('\n')
                filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
                content = '\n'.join(filtered_lines)
            
            # 关闭当前标签页，返回主页面
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            
            return content if content else "内容提取失败"
            
        except Exception as e:
            try:
                # 确保回到主页面
                if len(self.driver.window_handles) > 1:
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
            except:
                pass
            return f"内容提取失败: {str(e)}"

    def save_to_csv(self, data_list, file_path):
        """保存数据到CSV文件"""
        if not data_list:
            return
            
        fieldnames = ['title', 'url', 'summary', 'source_time', 'matched_keywords', 'content', 'crawl_time']
        
        # 检查文件是否存在，不存在则创建并写入表头
        file_exists = os.path.exists(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for data in data_list:
                writer.writerow(data)

    def get_matched_keywords(self, text):
        """获取匹配的关键词"""
        if not text:
            return ""
        
        text_lower = text.lower()
        matched = [kw for kw in self.keywords if kw in text_lower]
        return ', '.join(matched) if matched else ""

    def scrape_news(self):
        self.driver.get('https://news.baidu.com')
        time.sleep(random.uniform(2, 4))
        
        # 查找搜索框 - 使用正确的选择器
        try:
            search_box = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='word']")))
            # 优先使用id为'ww'的搜索框
            try:
                search_box = self.driver.find_element(By.ID, "ww")
            except:
                pass  # 如果找不到ww，使用之前找到的搜索框
        except Exception as e:
            print(f"找不到搜索框: {e}")
            # 尝试其他选择器
            try:
                search_box = self.wait.until(EC.presence_of_element_located((By.ID, "kw")))
            except:
                print("无法找到任何搜索框")
                return
        
        print(f"找到搜索框，准备搜索关键词: {self.keywords}")
        
        # 输入关键词
        query = ' '.join(self.keywords)
        search_box.clear()
        search_box.send_keys(query)
        time.sleep(random.uniform(0.5, 1.5))
        
        # 点击搜索按钮或按回车
        try:
            # 尝试点击搜索按钮
            search_button = self.driver.find_element(By.CSS_SELECTOR, "input[value*='搜索']")
            search_button.click()
        except:
            # 如果找不到按钮，按回车
            search_box.send_keys(Keys.RETURN)
        
        time.sleep(random.uniform(3, 5))
        print(f"搜索完成，当前URL: {self.driver.current_url}")
        
        # 点击按时间排序
        try:
            sort_by_time = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), '按时间') or contains(text(), '时间')]")))
            sort_by_time.click()
            time.sleep(random.uniform(2, 3))
            print("已切换到时间排序")
        except:
            print("未找到时间排序按钮，继续爬取...")
        
        batch_data = []
        page_num = 1
        
        while self.crawled_count < self.max_results and not self.should_stop:
            print(f"正在爬取第 {page_num} 页...")
            
            # 检查验证码
            if self.check_captcha():
                self.wait_for_captcha_manual()
                if self.should_stop:
                    break
            
            # 模拟人类行为
            self.simulate_mouse_movements()
            
            # 获取页面源码
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找新闻条目 - 使用更准确的选择器
            news_items = []
            
            # 尝试多种百度新闻搜索结果的选择器
            selectors = [
                'div.result',  # 通用结果容器
                'div.c-result',  # 百度新闻特定类名
                'div[data-log-click]',  # 带有日志点击数据的元素
                'li.result',
                'div.news-item',
                'article',
                'div[class*="result"]',
                'div[class*="news"]',
                'li[class*="result"]'
            ]
            
            for selector in selectors:
                items = soup.select(selector)
                if items:
                    print(f"使用选择器 '{selector}' 找到 {len(items)} 个新闻条目")
                    news_items = items
                    break
            
            if not news_items:
                print("尝试查找所有链接元素...")
                # 最后尝试：查找所有包含链接的元素
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    # 如果链接指向外部网站且有文本内容，可能是新闻链接
                    if href.startswith('http') and len(text) > 10 and not any(x in href for x in ['baidu.com', 'news.baidu.com']):
                        parent = link.find_parent()
                        if parent:
                            news_items.append(parent)
                            if len(news_items) >= 10:  # 限制数量避免过多无关元素
                                break
                
                print(f"通过链接查找找到 {len(news_items)} 个可能的新闻条目")
            
            found_new = False
            
            for item in news_items:
                if self.crawled_count >= self.max_results or self.should_stop:
                    break
                
                try:
                    # 提取标题和链接
                    title_elem = item.find('a') or item.find('h3')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    if not url or not title:
                        continue
                    
                    # 补全相对URL
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        url = 'https://news.baidu.com' + url
                    
                    # 去重检查
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    if url in self.crawled_urls:
                        continue
                    
                    # 提取摘要
                    summary_elem = item.find('div', class_=re.compile(r'summary|desc|content'))
                    summary = summary_elem.get_text(strip=True) if summary_elem else ""
                    
                    # 提取来源和时间
                    source_time_elem = item.find('span', class_=re.compile(r'time|date|source'))
                    source_time = source_time_elem.get_text(strip=True) if source_time_elem else ""
                    
                    # 检查是否包含关键词
                    full_text = f"{title} {summary}"
                    matched_keywords = self.get_matched_keywords(full_text)
                    
                    if not matched_keywords:
                        continue
                    
                    # 提取详细内容
                    content = self.extract_news_content(url)
                    
                    # 创建数据记录
                    news_data = {
                        'title': title,
                        'url': url,
                        'summary': summary,
                        'source_time': source_time,
                        'matched_keywords': matched_keywords,
                        'content': content,
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # 添加到批次
                    batch_data.append(news_data)
                    self.crawled_urls.add(url)
                    self.crawled_count += 1
                    found_new = True
                    
                    print(f"已爬取 {self.crawled_count}/{self.max_results}: {title[:50]}...")
                    
                    # 实时推送到前端
                    yield news_data
                    
                    # 每5条保存一次
                    if len(batch_data) >= 5:
                        self.save_to_csv(batch_data, CSV_FILE)
                        batch_data = []
                    
                    # 智能休息
                    if self.crawled_count % 30 == 0:
                        rest_time = random.uniform(15, 30)
                        print(f"休息 {rest_time:.1f} 秒...")
                        time.sleep(rest_time)
                    
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    print(f"处理新闻条目时出错: {e}")
                    continue
            
            # 保存剩余数据
            if batch_data:
                self.save_to_csv(batch_data, CSV_FILE)
                batch_data = []
            
            # 如果没有找到新内容，尝试翻页
            if not found_new:
                print("本页没有找到新内容，尝试翻页...")
            
            # 尝试点击下一页
            try:
                next_page = self.driver.find_element(By.XPATH, "//a[contains(text(), '下一页') or contains(@class, 'next')]")
                next_page.click()
                time.sleep(random.uniform(3, 5))
                page_num += 1
            except:
                # 尝试点击页码
                try:
                    page_links = self.driver.find_elements(By.XPATH, "//a[contains(@class, 'page') or contains(@href, 'pn=')]")
                    if page_links and page_num < len(page_links):
                        page_links[page_num].click()
                        time.sleep(random.uniform(3, 5))
                        page_num += 1
                    else:
                        print("没有更多页面了，爬取结束")
                        break
                except:
                    print("无法翻页，爬取结束")
                    break
        
        # 保存完整数据到all文件
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            with open(CSV_ALL_FILE, 'w', encoding='utf-8-sig') as f:
                f.write(content)
        
        print(f"爬取完成！共爬取 {self.crawled_count} 条新闻")

    def stop(self):
        """停止爬虫"""
        self.should_stop = True

    def close(self):
        """关闭浏览器"""
        if hasattr(self, 'driver'):
            self.driver.quit()


def run_baidu_news_crawler(keywords, max_results=100, proxy=None):
    """运行百度新闻爬虫"""
    scraper = BaiduNewsScraper(keywords, max_results, proxy)
    try:
        yield from scraper.scrape_news()
    finally:
        scraper.close()


def run_baidu_news_crawler_stream(keywords, max_results=100, proxy=None):
    """运行百度新闻爬虫（流式）"""
    scraper = None
    try:
        scraper = BaiduNewsScraper(keywords, max_results, proxy)
        for data in scraper.scrape_news():
            yield data
    except GeneratorExit:
        # 正常的流结束，确保清理资源
        if scraper:
            scraper.stop()
    except Exception as e:
        # 发生异常时也要清理资源
        if scraper:
            scraper.stop()
        raise e
    finally:
        if scraper:
            scraper.close()
