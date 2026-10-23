#python bbc_news_crawler.py- coding: utf-8 -*-
"""
BBC News Crawler for Biomanufacturing Keywords
根据CSDN教程使用requests和BeautifulSoup抓取BBC新闻

Features:
- 读取biomanufacturing_keywords.txt中的关键词
- 通过BBC搜索API获取新闻
- 提取标题、发布时间、关键词
- 保存到bbc_bio.csv
- 目标：不少于3万条数据
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re
from datetime import datetime
from urllib.parse import quote_plus
import os
import json
import threading


class BBCNewsCrawler:
    """BBC新闻爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        self.session.headers.update(self.headers)
        self.results = []
        self.seen_urls = set()  # 去重
        self.last_save_time = time.time()
        self.save_interval = 30  # 每30秒保存一次
        self.csv_filepath = 'bbc_bio.csv'
        self.lock = threading.Lock()
        
    def load_keywords(self, filepath):
        """从文件加载关键词"""
        keywords = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue
                    # 提取关键词（格式：数字. 关键词）
                    match = re.match(r'^\d+\.\s*(.+)$', line)
                    if match:
                        keyword = match.group(1).strip()
                        if keyword and len(keyword) > 1:
                            keywords.append(keyword)
            print(f"成功加载 {len(keywords)} 个关键词")
            return keywords
        except Exception as e:
            print(f"加载关键词文件失败: {e}")
            return []
    
    def search_bbc(self, keyword, page=1):
        """
        搜索BBC新闻
        使用BBC搜索页面
        """
        try:
            encoded_keyword = quote_plus(keyword)
            # BBC搜索URL
            url = f"https://www.bbc.co.uk/search?q={encoded_keyword}&page={page}"
            
            print(f"  搜索: '{keyword}' (第{page}页)")
            
            response = self.session.get(url, timeout=30, allow_redirects=True)
            
            if response.status_code != 200:
                print(f"  HTTP错误: {response.status_code}")
                return None
                
            return response.text
            
        except requests.exceptions.Timeout:
            print(f"  请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  请求失败: {e}")
            return None
    
    def parse_search_results(self, html, keyword):
        """解析搜索结果页面"""
        articles = []
        
        if not html:
            return articles
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # BBC搜索结果页面结构
        # 尝试多种可能的选择器
        
        # 方法1: 查找data-testid属性
        result_items = soup.find_all('div', {'data-testid': 'card-text-wrapper'})
        
        # 方法2: 查找article标签
        if not result_items:
            result_items = soup.find_all('article')
        
        # 方法3: 查找包含promo的class
        if not result_items:
            result_items = soup.find_all('div', class_=re.compile('promo|card', re.I))
        
        # 方法4: 查找所有包含标题的div
        if not result_items:
            result_items = soup.find_all('div', class_=re.compile('ssrcss-', re.I))
        
        print(f"    找到 {len(result_items)} 个结果项")
        
        for item in result_items:
            try:
                # 提取标题 - 尝试多种选择器
                title_elem = None
                for selector in ['h2', 'h3', 'h1', 'a[data-testid="card-headline"]', 'a']:
                    title_elem = item.find(selector) if isinstance(selector, str) and not selector.startswith('a[') else item.select_one(selector)
                    if title_elem:
                        break
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 5:
                    continue
                
                # 提取链接
                link = ''
                if title_elem.name == 'a' and title_elem.get('href'):
                    link = title_elem['href']
                else:
                    link_elem = item.find('a', href=True)
                    if link_elem:
                        link = link_elem.get('href', '')
                
                if link and link.startswith('/'):
                    link = f"https://www.bbc.co.uk{link}"
                
                # 去重检查
                if link in self.seen_urls:
                    continue
                
                # 提取发布时间 - 尝试多种方式
                pub_time = ''
                time_elem = item.find('time')
                if time_elem:
                    pub_time = time_elem.get_text(strip=True)
                    if not pub_time and time_elem.get('datetime'):
                        pub_time = time_elem['datetime']
                
                # 如果time标签没有文本，尝试其他方式
                if not pub_time:
                    time_elem = item.find('span', class_=re.compile('date|time', re.I))
                    if time_elem:
                        pub_time = time_elem.get_text(strip=True)
                
                # 提取摘要/描述
                description = ''
                desc_elem = item.find('p') or item.find('div', class_=re.compile('description|summary|content', re.I))
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                
                articles.append({
                    'title': title,
                    'url': link,
                    'publish_time': pub_time,
                    'keyword': keyword,
                    'description': description[:200]  # 限制长度
                })
                
                if link:
                    self.seen_urls.add(link)
                
            except Exception as e:
                continue
        
        return articles
    
    def crawl_by_keyword(self, keyword, max_pages=3):
        """根据关键词爬取新闻"""
        all_articles = []
        
        for page in range(1, max_pages + 1):
            html = self.search_bbc(keyword, page)
            
            if not html:
                break
            
            # 检查是否是有效页面（不是"无结果"页面）
            if "No results found" in html or "did not match any documents" in html:
                print(f"    无搜索结果")
                break
                
            articles = self.parse_search_results(html, keyword)
            
            if not articles:
                print(f"    第{page}页无有效数据")
                # 再试一次，可能是页面结构不同
                if page == 1:
                    break
            else:
                all_articles.extend(articles)
                print(f"    第{page}页获取 {len(articles)} 条")
            
            # 随机延迟，避免被封
            time.sleep(random.uniform(3, 6))
        
        return all_articles
    
    def save_to_csv(self, filepath=None, show_message=True):
        """保存结果到CSV"""
        if not self.results:
            if show_message:
                print("没有数据可保存")
            return
        
        if filepath is None:
            filepath = self.csv_filepath
        
        with self.lock:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filepath)) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['title', 'publish_time', 'keyword', 'url', 'description'])
                writer.writeheader()
                writer.writerows(self.results)
            
            if show_message:
                print(f"\n数据已保存到: {filepath}")
                print(f"总计: {len(self.results)} 条新闻")
                print(f"去重后: {len(self.seen_urls)} 条唯一URL")
    
    def check_and_save(self, force=False):
        """检查是否需要保存（每30秒自动保存）"""
        current_time = time.time()
        if force or (current_time - self.last_save_time >= self.save_interval):
            self.save_to_csv(show_message=False)
            self.last_save_time = current_time
            print(f"  [自动保存] {len(self.results)} 条数据已保存到 {self.csv_filepath}")
    
    def save_progress(self, keyword_idx):
        """保存进度"""
        progress_file = 'crawler_progress.json'
        progress = {
            'last_keyword_idx': keyword_idx,
            'total_results': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f)
    
    def load_progress(self):
        """加载进度"""
        progress_file = 'crawler_progress.json'
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def run(self, keywords_file, target_count=30000, resume=False):
        """运行爬虫"""
        print("=" * 70)
        print("BBC News Crawler for Biomanufacturing")
        print("=" * 70)
        
        # 加载关键词
        keywords = self.load_keywords(keywords_file)
        if not keywords:
            print("无可用关键词，退出")
            return
        
        # 检查是否有进度可恢复
        start_idx = 0
        if resume:
            progress = self.load_progress()
            if progress:
                start_idx = progress.get('last_keyword_idx', 0)
                print(f"\n恢复进度: 从第 {start_idx+1} 个关键词继续")
        
        print(f"\n目标: 获取不少于 {target_count} 条新闻")
        print(f"关键词数量: {len(keywords)}")
        print(f"开始位置: 第 {start_idx+1} 个关键词")
        print("=" * 70)
        
        # 逐个关键词爬取
        for idx, keyword in enumerate(keywords[start_idx:], start=start_idx+1):
            print(f"\n[{idx}/{len(keywords)}] 处理关键词: {keyword}")
            
            try:
                articles = self.crawl_by_keyword(keyword, max_pages=3)
                self.results.extend(articles)
                
                print(f"  当前总计: {len(self.results)} 条")
                
                # 每10个关键词保存一次进度
                if idx % 10 == 0:
                    self.save_progress(idx)
                    print(f"  进度已保存")
                
                # 检查是否需要自动保存（每30秒）
                self.check_and_save()
                
                # 达到目标数量时停止
                if len(self.results) >= target_count:
                    print(f"\n✓ 已达到目标数量 {target_count}，停止爬取")
                    break
                
                # 关键词之间延迟
                delay = random.uniform(5, 10)
                print(f"  等待 {delay:.1f} 秒...")
                time.sleep(delay)
                
            except Exception as e:
                print(f"  处理关键词时出错: {e}")
                # 出错时也尝试保存
                self.check_and_save(force=True)
                continue
        
        # 最终保存
        self.save_to_csv()
        self.save_progress(idx)


def main():
    """主函数"""
    crawler = BBCNewsCrawler()
    
    # 关键词文件路径
    keywords_file = 'biomanufacturing_keywords.txt'
    
    # 检查文件是否存在
    if not os.path.exists(keywords_file):
        print(f"错误: 找不到关键词文件 '{keywords_file}'")
        print(f"当前目录: {os.getcwd()}")
        print(f"文件列表: {os.listdir('.')}")
        return
    
    # 运行爬虫，目标3万条
    # resume=True 可以从上次中断的地方继续
    crawler.run(keywords_file, target_count=30000, resume=False)


if __name__ == '__main__':
    main()
