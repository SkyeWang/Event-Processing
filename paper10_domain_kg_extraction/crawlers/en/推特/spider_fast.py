# crawler/spider_fast.py - 快速版本，减少等待时间
#使用的是默认的Edge浏览器，需要先登录，连接代理，在开始代码的时候需要先关闭所有Edge浏览器进程
#直接python spider_fast.py 运行
import time
import random
import string
import csv
import os
import subprocess
import json
import logging
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawling.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

quantum_csv_path = 'twitter.csv'
checkpoint_path = 'checkpoint.json'
banned_keywords_path = 'banned_keywords.txt'


def load_keywords_from_file(filepath):
    """从文件加载关键词"""
    keywords = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                # 提取关键词（格式：数字. 关键词 或 纯关键词）
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    keyword = match.group(1).strip()
                else:
                    keyword = line  # 直接使用整行作为关键词
                if keyword and len(keyword) > 1:
                    keywords.append(keyword)
        print(f"成功加载 {len(keywords)} 个关键词")
        return keywords
    except FileNotFoundError:
        print(f"错误: 关键词文件未找到: {filepath}")
        return []
    except Exception as e:
        print(f"加载关键词文件失败: {e}")
        return []


def load_checkpoint():
    """加载断点信息"""
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            logger.info(f"已加载断点: 已爬取 {len(checkpoint.get('crawled_keywords', []))} 个关键词")
            return checkpoint
    except Exception as e:
        logger.warning(f"加载断点文件失败: {e}")
    return None


def save_checkpoint(crawled_keywords):
    """保存断点信息"""
    try:
        checkpoint = {
            'crawled_keywords': crawled_keywords,
            'last_update': datetime.now().isoformat(),
            'total_crawled': len(crawled_keywords)
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.debug(f"断点已保存: {len(crawled_keywords)} 个关键词")
    except Exception as e:
        logger.error(f"保存断点文件失败: {e}")


def kill_edge_processes():
    """关闭所有Edge浏览器进程"""
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'msedge.exe'], capture_output=True)
        print("已关闭所有Edge浏览器进程")
        time.sleep(2)
    except:
        pass


def save_banned_keyword(keyword):
    """保存被封禁的关键词到文件"""
    try:
        with open(banned_keywords_path, 'a', encoding='utf-8') as f:
            f.write(f"{keyword}\n")
        logger.info(f"已保存被封禁关键词: {keyword} -> {banned_keywords_path}")
    except Exception as e:
        logger.error(f"保存被封禁关键词失败: {e}")


def wait_and_retry(min_wait=300, max_wait=600, banned_level=0):
    """等待随机时间后重新开始"""
    # 根据封禁级别调整等待时间
    if banned_level == 0:
        wait_time = random.uniform(min_wait, max_wait)
        message = f"检测到IP可能被封禁，等待 {wait_time/60:.1f} 分钟后重试..."
    else:
        wait_min = min_wait * (2 ** (banned_level - 1))
        wait_max = max_wait * (2 ** (banned_level - 1))
        wait_min = max(wait_min, 1200)  # 最少等待20分钟
        wait_max = min(wait_max, 1800)  # 最多等待30分钟
        wait_time = random.uniform(wait_min, wait_max)
        message = f"检测到IP封禁级别 {banned_level}，等待 {wait_time/60:.1f} 分钟后重试..."

    logger.info(message)
    time.sleep(wait_time)


def restart_edge_and_continue():
    """重启Edge并继续爬取"""
    logger.info("重新启动Edge浏览器...")
    kill_edge_processes()
    time.sleep(5)
    logger.info("Edge浏览器已重启，继续爬取...")


class FastTwitterScraper:
    def __init__(self, keywords, proxy=None, save_interval=30):
        options = EdgeOptions()
        options.use_chromium = True

        # 使用已登录的配置文件
        user_data_dir = r"C:\Users\30751\AppData\Local\Microsoft\Edge\User Data"
        options.add_argument(f"--user-data-dir={user_data_dir}")
        options.add_argument("--profile-directory=Default")

        # 反检测选项
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-infobars")

        options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")

        if proxy:
            options.add_argument(f"--proxy-server={proxy}")

        options.binary_location = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

        edge_service = EdgeService(executable_path=r"D:\edgedriver\msedgedriver.exe")
        self.driver = webdriver.Edge(service=edge_service, options=options)

        # 隐藏webdriver属性
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            '''
        })

        self.wait = WebDriverWait(self.driver, 20)
        self.keywords = [kw.lower() for kw in keywords]
        self.save_interval = save_interval
        self.last_save_time = time.time()
        self.buffer = []
        self.seen_tweets = set()  # 去重集合
        self.crawled_keywords = []  # 记录已爬取的关键词
        self.consecutive_zero_count = 0  # 连续零结果的计数器
        self.banned_keywords = set()  # 记录被封禁的关键词
        self.first_banned_keyword = None  # 记录此次被封禁的第一个关键词
        self.restart_zero_count = 0  # 重新开始后连续零结果的计数器
        self.banned_level = 0  # 封禁级别，决定等待时间长短

        # 检查CSV文件是否存在，如果不存在则创建并写入表头
        file_exists = os.path.exists(quantum_csv_path)
        if not file_exists:
            with open(quantum_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['内容', '作者', '日期', '关键词'])
            self.count = 0
        else:
            # 读取已有数据，建立去重集合
            try:
                with open(quantum_csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过表头
                    for row in reader:
                        if row:
                            tweet_id = hash(row[0][:100])  # 使用内容前100字符作为唯一标识
                            self.seen_tweets.add(tweet_id)
                self.count = len(self.seen_tweets)
                logger.info(f"已加载 {self.count} 条历史数据，将继续追加新数据")
            except:
                self.count = 0

    def manual_login(self):
        print("\n正在打开Twitter...")
        self.driver.get('https://twitter.com')
        time.sleep(3)
        
        current_url = self.driver.current_url
        if 'login' in current_url.lower():
            print("请在浏览器中登录Twitter，然后按回车键继续...")
            input()
        else:
            print("已检测到登录状态！")
        
        time.sleep(2)

    def _save_buffer(self):
        """保存缓冲区数据"""
        if not self.buffer:
            return

        with open(quantum_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)

        logger.info(f"[保存] {len(self.buffer)} 条数据 (总计: {self.count})")
        self.buffer = []
        self.last_save_time = time.time()

        # 同时保存断点
        save_checkpoint(self.crawled_keywords)

    def search_keyword(self, keyword):
        """搜索单个关键词"""
        start_time = time.time()
        logger.info(f"开始搜索关键词: {keyword}")

        encoded = quote(keyword)
        # 添加近两年的时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 近两年
        since = start_date.strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')
        url = f'https://twitter.com/search?q={encoded}%20since%3A{since}%20until%3A{until}&src=typed_query&f=live'

        try:
            self.driver.get(url)
            time.sleep(5)  # 增加初始等待时间，减少人机验证风险
        except Exception as e:
            logger.error(f"访问失败: {e}")
            return 0

        found_total = 0
        last_height = 0
        no_change_count = 0

        for scroll in range(30):  # 每个关键词滚动30次
            # 获取页面内容
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            articles = soup.find_all('article')

            found_this_scroll = 0
            for art in articles:
                try:
                    # 提取推文内容
                    txt_elem = art.find('div', {'data-testid': 'tweetText'})
                    if not txt_elem:
                        continue
                    txt = txt_elem.text.strip()

                    # 去重检查
                    tweet_id = hash(txt[:100])  # 使用前100字符作为唯一标识
                    if tweet_id in self.seen_tweets:
                        continue
                    self.seen_tweets.add(tweet_id)

                    # 检查是否包含量子科技关键词
                    txt_lower = txt.lower()
                    matched_kws = [kw for kw in self.keywords if kw in txt_lower]
                    if not matched_kws:
                        continue

                    # 提取作者
                    author = "未知"
                    try:
                        author_elem = art.find('a', {'href': re.compile(r'^/\w+$')})
                        if author_elem:
                            author = author_elem.get('href', '').strip('/')
                    except:
                        pass

                    # 提取日期
                    date_str = "未知"
                    try:
                        time_elem = art.find('time')
                        if time_elem:
                            date_str = time_elem.get('datetime', '')
                    except:
                        pass

                    # 添加到缓冲区
                    self.buffer.append([txt, author, date_str, ', '.join(matched_kws)])
                    self.count += 1
                    found_this_scroll += 1
                    found_total += 1

                    # 检查保存
                    if (time.time() - self.last_save_time >= self.save_interval) or len(self.buffer) >= 200:
                        self._save_buffer()

                except Exception:
                    continue

            if found_this_scroll > 0:
                logger.info(f"  滚动 {scroll+1}/30: 找到 {found_this_scroll} 条新推文")

            # 慢速滚动，避免触发人机验证
            # 先滚动到中间位置，再滚动到底部，促进内容加载
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight * 0.7);')
            time.sleep(random.uniform(1.0, 2.0))
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(random.uniform(4.0, 8.0))  # 随机等待4-8秒

            # 检查是否到达底部
            new_height = self.driver.execute_script('return document.body.scrollHeight;')
            # 只有当找到的新推文数量为0且连续多次滚动高度不变时才判定到底
            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= 3:  # 增加阈值，避免误判
                    logger.info("  已到达页面底部（连续5次无新内容）")
                    break
            else:
                no_change_count = 0
                if new_height > last_height:
                    last_height = new_height

        elapsed = time.time() - start_time
        logger.info(f"关键词 '{keyword}' 爬取完成: 找到 {found_total} 条推文, 耗时 {elapsed:.1f} 秒")

        # 记录已爬取的关键词
        self.crawled_keywords.append(keyword)

        # IP封禁检测
        if found_total == 0:
            self.consecutive_zero_count += 1
            logger.warning(f"连续 {self.consecutive_zero_count} 个关键词返回0条结果")

            # 连续2个或以上关键词返回0，可能IP被封禁
            if self.consecutive_zero_count >= 2:
                logger.error(f"连续 {self.consecutive_zero_count} 个关键词返回0条结果，IP可能被封禁！")
                save_banned_keyword(keyword)
                self.banned_keywords.add(keyword)
                return -1  # 返回-1表示IP被封禁
        else:
            self.consecutive_zero_count = 0  # 重置计数器

        return found_total

    def run(self, start_keyword=None, use_checkpoint=True):
        """运行爬虫

        Args:
            start_keyword: 从哪个关键词开始爬取（默认从头开始）
            use_checkpoint: 是否使用断点功能（默认True）
        """
        total_keywords = len(self.keywords)

        # 如果指定了开始关键词，找到其位置
        start_idx = 0
        if start_keyword:
            start_lower = start_keyword.lower()
            for i, kw in enumerate(self.keywords):
                if kw.lower() == start_lower:
                    start_idx = i + 1  # 从该关键词之后开始
                    logger.info(f"从关键词 '{start_keyword}' 之后开始爬取（从位置: {i+2}/{total_keywords}）")
                    break
            else:
                logger.warning(f"未找到关键词 '{start_keyword}'，从头开始爬取")

        # 尝试从断点继续
        if use_checkpoint:
            checkpoint = load_checkpoint()
            if checkpoint:
                crawled_keywords = checkpoint.get('crawled_keywords', [])
                if crawled_keywords:
                    # 找到最后一个已爬取关键词的位置
                    last_crawled = crawled_keywords[-1]
                    for i, kw in enumerate(self.keywords):
                        if kw.lower() == last_crawled.lower():
                            start_idx = max(start_idx, i + 1)
                            logger.info(f"从断点继续: 已爬取 {len(crawled_keywords)} 个关键词，从位置 {start_idx+1} 开始")
                            break

        remaining_keywords = self.keywords[start_idx:]
        logger.info(f"开始爬取，共 {len(remaining_keywords)} 个关键词（总计 {total_keywords} 个）")
        logger.info(f"目标: 近180天数据")
        logger.info(f"爬取速度: 初始等待5秒，滚动间隔4-8秒，每关键词滚动30次，关键词间隔休息20-40秒")

        start_time = time.time()

        for idx, keyword in enumerate(remaining_keywords, start_idx + 1):
            logger.info(f"[{idx}/{total_keywords}] 开始搜索: {keyword}")
            found = self.search_keyword(keyword)

            # 检查是否IP被封禁
            if found == -1:
                logger.error(f"IP被封禁，当前关键词 '{keyword}' 爬取失败")

                # 记录第一个被封禁的关键词
                if self.first_banned_keyword is None:
                    self.first_banned_keyword = keyword
                    logger.info(f"记录此次封禁的第一个关键词: {self.first_banned_keyword}")

                # 保存当前进度
                self._save_buffer()
                logger.info("断点已保存，准备重启...")

                # 关闭浏览器
                self.driver.quit()

                # 增加封禁级别
                self.banned_level += 1
                logger.info(f"当前封禁级别: {self.banned_level}")

                # 根据封禁级别调整等待时间
                if self.banned_level == 1:
                    # 第一次被封禁，等待5-10分钟
                    wait_time = random.uniform(300, 600)
                else:
                    # 后续封禁，等待20-30分钟（指数增长）
                    wait_min = 1200 * (2 ** (self.banned_level - 2))
                    wait_max = 1800 * (2 ** (self.banned_level - 2))
                    wait_min = min(wait_min, 3600)  # 最多1小时
                    wait_max = min(wait_max, 5400)  # 最多1.5小时
                    wait_time = random.uniform(wait_min, wait_max)

                logger.info(f"等待 {wait_time/60:.1f} 分钟后重试...")
                time.sleep(wait_time)

                # 重新启动Edge
                restart_edge_and_continue()

                # 重新登录
                self.manual_login()

                # 记录重试次数
                retry_count = 0
                retry_found = 0

                # 从第一个被封禁的关键词开始重新爬取
                current_idx = max(0, idx - 1)  # 找到第一个被封禁关键词的位置
                for retry_idx in range(current_idx, min(current_idx + 4, len(self.keywords))):
                    retry_keyword = self.keywords[retry_idx]
                    logger.info(f"重新尝试爬取关键词: {retry_keyword}")
                    retry_found = self.search_keyword(retry_keyword)

                    # 更新重试计数
                    if retry_found == 0:
                        retry_count += 1
                    else:
                        retry_count = 0  # 重置重试计数

                    # 检查重试结果
                    if retry_found > 0:
                        logger.info(f"重新爬取成功，找到 {retry_found} 条推文")
                        # 重置封禁级别和计数器
                        self.banned_level = 0
                        self.restart_zero_count = 0
                        self.first_banned_keyword = None
                        break

                    # 如果连续4个关键词都为0，继续等待
                    if retry_idx - current_idx + 1 >= 4:
                        logger.error(f"连续4个关键词都为0，继续等待...")
                        wait_time = random.uniform(1200, 1800)  # 20-30分钟
                        logger.info(f"等待 {wait_time/60:.1f} 分钟后重试...")
                        time.sleep(wait_time)
                        restart_edge_and_continue()
                        self.manual_login()
                        retry_idx = current_idx - 1  # 重新开始循环

                # 如果这次重试没有成功，设置连续0结果计数器
                if retry_found == 0:
                    self.restart_zero_count = retry_count
                else:
                    # 如果中间有成功的关键词，重置所有计数器
                    self.consecutive_zero_count = 0
                    self.restart_zero_count = 0

            logger.info(f"[{idx}/{total_keywords}] 该关键词共找到 {found} 条推文\n")

            # 关键词之间短暂休息，避免触发人机验证
            if idx < total_keywords:
                rest_time = random.uniform(20, 40)
                logger.info(f"休息 {rest_time:.1f} 秒...")
                time.sleep(rest_time)

        # 最后保存
        self._save_buffer()

        # 统计
        elapsed = time.time() - start_time
        logger.info("="*60)
        logger.info("爬取完成!")
        logger.info(f"总数据量: {self.count} 条")
        logger.info(f"已爬取关键词: {len(self.crawled_keywords)} 个")
        logger.info(f"用时: {elapsed/60:.1f} 分钟")
        logger.info(f"平均速度: {self.count/(elapsed/60):.0f} 条/分钟")
        logger.info(f"数据保存至: {quantum_csv_path}")
        logger.info(f"断点保存至: {checkpoint_path}")

        # 如果有被封禁的关键词，显示被封禁的关键词文件
        if self.banned_keywords:
            logger.info(f"被封禁的关键词已保存至: {banned_keywords_path}")
            logger.info(f"共封禁 {len(self.banned_keywords)} 个关键词")

        logger.info("="*60)

        self.driver.quit()


if __name__ == "__main__":
    # 从 keywords.txt 文件加载关键词
    keywords_file = 'keywords.txt'

    # 检查文件是否存在
    if not os.path.exists(keywords_file):
        logger.error(f"找不到关键词文件 '{keywords_file}'")
        logger.error(f"当前目录: {os.getcwd()}")
        exit(1)

    keywords = load_keywords_from_file(keywords_file)
    if not keywords:
        logger.error("无可用关键词，退出")
        exit(1)

    # 检查被封禁关键词文件，如果存在则过滤掉
    banned_keywords = set()
    if os.path.exists(banned_keywords_path):
        try:
            with open(banned_keywords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        banned_keywords.add(line.lower())
            logger.info(f"检测到 {len(banned_keywords)} 个已被封禁的关键词")

            # 显示被封禁的关键词
            if banned_keywords:
                logger.info("被封禁的关键词如下：")
                for i, kw in enumerate(banned_keywords, 1):
                    logger.info(f"  {i}. {kw}")

            # 从关键词列表中移除被封禁的关键词
            original_count = len(keywords)
            keywords = [kw for kw in keywords if kw.lower() not in banned_keywords]
            removed_count = original_count - len(keywords)
            logger.info(f"已移除 {removed_count} 个被封禁的关键词，剩余 {len(keywords)} 个关键词")

            if not keywords:
                logger.error("所有关键词都已被封禁，请稍后再试或更换关键词")
                logger.info(f"被封禁的关键词列表请查看: {banned_keywords_path}")
                exit(1)

        except Exception as e:
            logger.warning(f"读取被封禁关键词文件失败: {e}")

    logger.info(f"关键词数量: {len(keywords)}")

    # 检查是否有断点文件
    if os.path.exists(checkpoint_path):
        logger.info(f"检测到断点文件 {checkpoint_path}，将从断点继续爬取")

    # 关闭Edge进程
    kill_edge_processes()

    # 运行爬虫 - 从断点继续（可以修改 start_keyword 指定起始位置）
    scraper = FastTwitterScraper(keywords, save_interval=30)
    scraper.manual_login()
    scraper.run(start_keyword=None, use_checkpoint=True)  # 设置为 None 从断点开始，或指定关键词
