#python youtube_scraper_chrome.py运行
import json
import time
import re
import os
import csv
import threading
from typing import List, Dict, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


class YouTubeScraperChrome:
    """
    YouTube爬虫类 - 使用Chrome浏览器自动化爬取
    支持自动保存、增量爬取、多线程采集
    """

    def __init__(self, output_file: str = "youtubebio.csv", headless: bool = True):
        """
        初始化YouTube爬虫

        Args:
            output_file: 输出CSV文件路径
            headless: 是否使用无头模式（不显示浏览器窗口）
        """
        self.output_file = os.path.abspath(output_file)
        self.headless = headless

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 数据存储
        self.results: List[Dict] = []
        self.seen_video_ids: Set[str] = set()
        self.lock = threading.Lock()

        # 自动保存控制
        self.last_save_time = time.time()
        self.save_interval = 30  # 30秒保存一次
        self.total_target = 30000  # 目标数据量

        # 统计信息
        self.keywords_processed = 0
        self.total_keywords = 0
        self.start_time = None

        # 初始化浏览器
        self.driver = None
        self._init_browser()

        # 加载已存在的数据（断点续传）
        self._load_existing_data()

    def _init_browser(self):
        """初始化Chrome浏览器"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # 设置User-Agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        # 禁用图片加载以提高速度
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # 手动指定Chrome浏览器路径
        chrome_binary_path = r"D:\Gemini Chrome下载\chrome-win64\chrome-win64\chrome.exe"
        if os.path.exists(chrome_binary_path):
            chrome_options.binary_location = chrome_binary_path
            print(f"使用指定Chrome路径: {chrome_binary_path}")
        else:
            print("警告: 未找到指定路径的Chrome浏览器，将使用系统默认路径")

        # 手动指定ChromeDriver路径
        chromedriver_path = r"D:\Gemini Chrome下载\chrome-win64\chromedriver-win64\chromedriver-win64\chromedriver.exe"
        
        try:
            if os.path.exists(chromedriver_path):
                print(f"使用指定ChromeDriver: {chromedriver_path}")
                service = Service(chromedriver_path)
                print("正在启动Chrome浏览器...")
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("Chrome浏览器启动成功")
            else:
                print("未找到指定ChromeDriver，尝试自动下载...")
                # 尝试使用webdriver-manager自动下载ChromeDriver
                from webdriver_manager.chrome import ChromeDriverManager
                
                os.environ['WDM_LOCAL'] = '1'
                os.environ['WDM_SSL_VERIFY'] = '0'
                os.environ['WDM_PROGRESS_BAR'] = '0'
                
                driver_manager = ChromeDriverManager()
                driver_path = driver_manager.install()
                print(f"ChromeDriver路径: {driver_path}")
                
                service = Service(driver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)

            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("Chrome浏览器初始化成功")
        except Exception as e:
            print(f"浏览器初始化失败: {e}")
            print("\n请确保:")
            print("1. 已安装Chrome浏览器")
            print("2. 已安装ChromeDriver并添加到系统PATH")
            print("3. 或手动下载ChromeDriver并放在脚本同目录")
            print("\n下载地址: https://chromedriver.chromium.org/downloads")
            print("\n当前Chrome路径:")
            print(f"  {chrome_binary_path}")
            print("请确认此路径是否正确")
            raise

    def _load_existing_data(self):
        """加载已存在的CSV数据，实现断点续传"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        video_id = row.get('video_id', '')
                        if video_id and video_id not in self.seen_video_ids:
                            self.seen_video_ids.add(video_id)
                print(f"已加载现有数据: {len(self.seen_video_ids)} 条记录")
            except Exception as e:
                print(f"加载现有数据时出错: {e}")

    def load_keywords(self, filepath: str) -> List[str]:
        """从文件加载关键词列表"""
        keywords = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        keyword = re.sub(r'^\d+\.\s*', '', line)
                        if keyword:
                            keywords.append(keyword)
            print(f"成功加载 {len(keywords)} 个关键词")
            return keywords
        except FileNotFoundError:
            print(f"文件未找到: {filepath}")
            return []
        except Exception as e:
            print(f"加载关键词时出错: {e}")
            return []

    def search_videos(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """
        使用Chrome浏览器搜索YouTube视频

        Args:
            keyword: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            视频信息列表
        """
        videos = []
        search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}&sp=EgIQAQ%253D%253D"

        try:
            print(f"  正在访问: {search_url[:60]}...")
            self.driver.get(search_url)

            # 等待页面加载
            time.sleep(3)

            # 滚动页面加载更多视频
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)

            # 等待视频元素加载
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer"))
                )
            except TimeoutException:
                print("  等待视频元素超时")
                return videos

            # 获取视频元素
            video_elements = self.driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
            print(f"  找到 {len(video_elements)} 个视频元素")

            for element in video_elements[:max_results]:
                try:
                    # 提取视频ID
                    link_element = element.find_element(By.CSS_SELECTOR, "a#video-title")
                    href = link_element.get_attribute("href")
                    video_id = self._extract_video_id(href)

                    # 提取标题
                    title = link_element.get_attribute("title") or link_element.text

                    # 提取作者
                    try:
                        author_element = element.find_element(By.CSS_SELECTOR, "ytd-channel-name a")
                        author = author_element.text
                    except NoSuchElementException:
                        author = "Unknown"

                    # 提取发布时间
                    try:
                        time_element = element.find_element(By.CSS_SELECTOR, "span.style-scope.ytd-video-meta-block")
                        published_time = time_element.text
                    except NoSuchElementException:
                        published_time = ""

                    # 提取观看次数
                    try:
                        view_element = element.find_element(By.CSS_SELECTOR, "span.inline-metadata-item")
                        view_count = view_element.text
                    except NoSuchElementException:
                        view_count = ""

                    # 提取缩略图
                    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else ""

                    # 检查是否为英文视频
                    if video_id and title and self._is_english_content(title):
                        with self.lock:
                            if video_id not in self.seen_video_ids:
                                self.seen_video_ids.add(video_id)
                                video_info = {
                                    "keyword": keyword,
                                    "title": title,
                                    "author": author,
                                    "video_id": video_id,
                                    "url": f"https://www.youtube.com/watch?v={video_id}",
                                    "published_at": published_time,
                                    "view_count": view_count,
                                    "description": "",
                                    "thumbnail": thumbnail,
                                    "scraped_at": datetime.now().isoformat()
                                }
                                videos.append(video_info)

                except Exception as e:
                    continue

            print(f"  成功提取 {len(videos)} 个英文视频")
            return videos

        except Exception as e:
            print(f"  搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return videos

    def _extract_video_id(self, url: str) -> str:
        """从URL中提取视频ID"""
        if not url:
            return ""
        match = re.search(r'[?&]v=([^&]+)', url)
        if match:
            return match.group(1)
        # 处理短链接格式
        match = re.search(r'youtu\.be/([^?&]+)', url)
        if match:
            return match.group(1)
        return ""

    def _is_english_content(self, text: str) -> bool:
        """判断文本是否为英文内容"""
        if not text:
            return False
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
        if total_chars == 0:
            return True
        return english_chars / total_chars > 0.6

    def _auto_save(self):
        """自动保存数据（每30秒）"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self._save_to_csv_incremental()
            self.last_save_time = current_time

    def _save_to_csv_incremental(self):
        """增量保存到CSV文件"""
        with self.lock:
            if not self.results:
                return

            try:
                fieldnames = ["keyword", "title", "author", "url", "published_at", "video_id", "view_count", "description", "thumbnail", "scraped_at"]

                file_exists = os.path.exists(self.output_file)
                mode = 'a' if file_exists else 'w'

                with open(self.output_file, mode, newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)

                    if not file_exists or os.path.getsize(self.output_file) == 0:
                        writer.writeheader()

                    writer.writerows(self.results)

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 自动保存: 累计 {len(self.seen_video_ids)} 条数据 -> {self.output_file}")
                self.results = []

            except Exception as e:
                print(f"自动保存失败: {e}")
                import traceback
                traceback.print_exc()

    def process_keyword(self, keyword: str, max_results: int = 15) -> int:
        """处理单个关键词"""
        try:
            videos = self.search_videos(keyword, max_results)

            with self.lock:
                self.results.extend(videos)

            self.keywords_processed += 1

            # 显示进度
            progress = (self.keywords_processed / self.total_keywords) * 100 if self.total_keywords > 0 else 0
            print(f"[{self.keywords_processed}/{self.total_keywords}] {keyword}: +{len(videos)} 条 "
                  f"(累计: {len(self.seen_video_ids)} 条, 进度: {progress:.1f}%)")

            # 自动保存检查
            self._auto_save()

            # 随机延迟，避免请求过快
            time.sleep(random.uniform(3, 6))

            return len(videos)

        except Exception as e:
            print(f"处理关键词 '{keyword}' 时出错: {e}")
            return 0

    def scrape_by_keywords(self, keywords: List[str], max_results_per_keyword: int = 15,
                          max_total: int = 30000) -> int:
        """
        批量爬取视频

        Args:
            keywords: 关键词列表
            max_results_per_keyword: 每个关键词最大结果数
            max_total: 目标总数据量

        Returns:
            总数据条数
        """
        self.total_keywords = len(keywords)
        self.start_time = time.time()

        print(f"\n{'='*60}")
        print(f"开始爬取YouTube视频")
        print(f"{'='*60}")
        print(f"目标数据量: {max_total} 条")
        print(f"关键词数量: {len(keywords)}")
        print(f"每30秒自动保存到: {self.output_file}")
        print(f"{'='*60}\n")

        for keyword in keywords:
            if len(self.seen_video_ids) >= max_total:
                print(f"\n已达到目标数据量 {max_total} 条，停止爬取")
                break

            self.process_keyword(keyword, max_results_per_keyword)

        # 最终保存
        self._save_to_csv_incremental()

        # 关闭浏览器
        if self.driver:
            self.driver.quit()

        # 统计信息
        elapsed_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"爬取完成!")
        print(f"总数据量: {len(self.seen_video_ids)} 条")
        print(f"处理关键词: {self.keywords_processed}/{self.total_keywords}")
        print(f"耗时: {elapsed_time/60:.1f} 分钟")
        if elapsed_time > 0:
            print(f"平均速度: {len(self.seen_video_ids)/(elapsed_time/60):.0f} 条/分钟")
        print(f"数据保存至: {self.output_file}")
        print(f"{'='*60}")

        return len(self.seen_video_ids)

    def generate_additional_keywords(self, base_keywords: List[str]) -> List[str]:
        """基于基础关键词生成更多组合关键词"""
        prefixes = ["", "advanced ", "industrial ", "modern ", "sustainable ", "green "]
        suffixes = ["", " technology", " process", " production", " industry", " research"]

        extended_keywords = set(base_keywords)

        for keyword in base_keywords:
            for prefix in prefixes:
                for suffix in suffixes:
                    new_keyword = f"{prefix}{keyword}{suffix}".strip()
                    if new_keyword != keyword:
                        extended_keywords.add(new_keyword)

        # 添加组合关键词
        combinations = [
            "biomanufacturing tutorial",
            "biomanufacturing explained",
            "how does biomanufacturing work",
            "biomanufacturing applications",
            "future of biomanufacturing",
            "biomanufacturing vs traditional manufacturing"
        ]
        extended_keywords.update(combinations)

        result = list(extended_keywords)
        print(f"关键词扩展: {len(base_keywords)} -> {len(result)}")
        return result


def test_scraper():
    """测试爬虫是否能正常工作"""
    print("="*60)
    print("测试YouTube爬虫 (Chrome版本)")
    print("="*60)

    scraper = YouTubeScraperChrome(output_file="youtubebio.csv", headless=False)  # 显示浏览器窗口便于调试

    # 测试几个关键词
    test_keywords = ["biomanufacturing", "synthetic biology"]

    print(f"\n输出文件路径: {scraper.output_file}")
    print(f"测试关键词: {test_keywords}\n")

    for keyword in test_keywords:
        print(f"正在测试关键词: {keyword}")
        videos = scraper.search_videos(keyword, max_results=5)

        with scraper.lock:
            for video in videos:
                if video["video_id"] not in scraper.seen_video_ids:
                    scraper.seen_video_ids.add(video["video_id"])
                    scraper.results.append(video)

        print(f"  累计找到 {len(scraper.seen_video_ids)} 个唯一视频\n")
        time.sleep(3)

    # 立即保存
    scraper._save_to_csv_incremental()

    # 关闭浏览器
    if scraper.driver:
        scraper.driver.quit()

    # 检查文件是否生成
    if os.path.exists(scraper.output_file):
        file_size = os.path.getsize(scraper.output_file)
        print(f"\n✓ CSV文件已生成!")
        print(f"  文件路径: {scraper.output_file}")
        print(f"  文件大小: {file_size} 字节")

        # 读取并显示内容
        with open(scraper.output_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"  数据条数: {len(rows)}")

            if rows:
                print(f"\n前3条数据预览:")
                for i, row in enumerate(rows[:3], 1):
                    print(f"  {i}. {row.get('title', 'N/A')[:50]}...")
                    print(f"     作者: {row.get('author', 'N/A')}")
                    print(f"     链接: {row.get('url', 'N/A')}")
    else:
        print(f"\n✗ CSV文件未生成，请检查错误信息")


def main():
    """主函数"""
    import sys

    # 检查是否有测试参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_scraper()
        return

    # 初始化爬虫
    scraper = YouTubeScraperChrome(output_file="youtubebio.csv", headless=True)

    print(f"输出文件路径: {scraper.output_file}")

    # 加载基础关键词
    base_keywords = scraper.load_keywords("biomanufacturing_keywords.txt")

    if not base_keywords:
        print("未找到关键词文件，请先运行测试: python youtube_scraper_chrome.py --test")
        return

    # 扩展关键词以获取更多数据
    all_keywords = scraper.generate_additional_keywords(base_keywords)

    # 如果关键词不够，循环使用
    while len(all_keywords) < 2000:
        all_keywords.extend(base_keywords)

    print(f"最终关键词数量: {len(all_keywords)}")

    # 开始爬取
    total_records = scraper.scrape_by_keywords(
        keywords=all_keywords,
        max_results_per_keyword=10,  # 每个关键词获取10条
        max_total=30000  # 目标3万条
    )

    print(f"\n爬取结束，共获取 {total_records} 条数据")
    print(f"数据保存在: {scraper.output_file}")


if __name__ == "__main__":
    main()
