#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bing学术搜索爬虫程序
用于爬取中文关键词相关的学术文献信息
"""

import os
import time
import random
import csv
import logging
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException


class BingAcademicCrawler:
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.scraped_urls = set()
        self.driver = None
        self.current_keyword = None
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs('keywords_fin', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def get_random_user_agent(self):
        """获取随机User-Agent"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
        ]
        return random.choice(user_agents)
        
    def setup_driver(self):
        """配置Edge浏览器"""
        try:
            options = Options()
            options.add_argument(f'--user-agent={self.get_random_user_agent()}')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            
            # 使用Edge浏览器
            self.driver = webdriver.Edge(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.logger.info("浏览器配置成功")
            return True
            
        except Exception as e:
            self.logger.error(f"浏览器配置失败: {e}")
            return False
            
    def simulate_mouse_movements(self):
        """模拟鼠标随机移动"""
        try:
            actions = ActionChains(self.driver)
            # 先移动到页面中心位置
            actions.move_to_element(self.driver.find_element(By.TAG_NAME, "body"))
            actions.perform()
            
            # 然后进行小幅随机移动
            actions = ActionChains(self.driver)
            for _ in range(random.randint(3, 6)):
                x_offset = random.randint(-20, 20)
                y_offset = random.randint(-20, 20)
                actions.move_by_offset(x_offset, y_offset)
                actions.pause(random.uniform(0.1, 0.3))
            actions.perform()
        except Exception as e:
            # 鼠标移动失败不影响主要功能，只记录日志
            pass
            
    def read_keywords(self, file_path):
        """读取关键词文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            return keywords
        except Exception as e:
            self.logger.error(f"读取关键词文件失败 {file_path}: {e}")
            return []
            
    def load_existing_urls(self, csv_filename):
        """加载已存在的URL到去重集合"""
        try:
            file_path = os.path.join('data', csv_filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row.get('url'):
                            self.scraped_urls.add(row['url'])
                self.logger.info(f"已加载 {len(self.scraped_urls)} 个历史URL到去重集合")
        except Exception as e:
            self.logger.error(f"加载历史URL失败: {e}")
            
    def move_keyword_to_finished(self, file_path, keyword):
        """将已完成的关键词移动到完成文件夹"""
        try:
            # 构建完成文件路径
            filename = os.path.basename(file_path)
            finished_file_path = os.path.join('keywords_fin', filename)
            
            # 读取原文件所有关键词
            with open(file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            
            # 移除已完成的关键词
            if keyword in keywords:
                keywords.remove(keyword)
            
            # 如果还有关键词，重写原文件
            if keywords:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for kw in keywords:
                        f.write(kw + '\n')
            else:
                # 如果没有关键词了，删除原文件
                os.remove(file_path)
            
            # 添加到完成文件
            fin_filename = os.path.basename(file_path)
            fin_path = os.path.join('keywords_fin', fin_filename)
            with open(fin_path, 'a', encoding='utf-8') as f:
                f.write(keyword + '\n')
                
            self.logger.info(f"关键词 '{keyword}' 已移动到完成文件夹")
            
        except Exception as e:
            self.logger.error(f"移动关键词失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                self.logger.info("浏览器已关闭")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
            
    def search_keyword(self, keyword):
        """搜索关键词"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 检查浏览器会话是否有效
                if attempt > 0:
                    self.logger.info(f"重新初始化浏览器 (尝试 {attempt + 1}/{max_retries})")
                    self.setup_driver()
                
                # 构建搜索URL
                encoded_keyword = quote(keyword)
                search_url = f"https://cn.bing.com/academic/search?q={encoded_keyword}&sort=1&first=1&FORM=PENR"
                
                self.logger.info(f"开始搜索关键词: {keyword}")
                self.driver.get(search_url)
                
                # 等待页面加载
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # 模拟人类操作
                self.simulate_mouse_movements()
                time.sleep(random.uniform(2, 4))
                
                return True
                
            except Exception as e:
                self.logger.error(f"搜索关键词失败 '{keyword}' (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # 等待5秒后重试
                else:
                    return False
            
    def extract_search_results(self):
        """提取搜索结果元素"""
        results = []
        try:
            # 等待页面加载完成
            time.sleep(2)
            
            # 使用更精确的选择器来查找搜索结果容器
            result_selectors = [
                '.b_algo',          # Bing搜索结果容器
                '.b_title',         # 标题容器
                'h2',               # 标题标签
                'h3',               # 副标题标签
                '[data-bm]',        # Bing结果元素
                '.b_result',        # 结果容器
            ]
            
            all_elements = []
            for selector in result_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    all_elements.extend(elements)
                except:
                    continue
                    
            # 过滤出包含链接的有效结果
            seen_urls = set()
            for element in all_elements:
                try:
                    # 查找元素内的链接
                    link_element = element.find_element(By.TAG_NAME, "a")
                    url = link_element.get_attribute('href')
                    
                    if not url or url in seen_urls or url in self.scraped_urls:
                        continue
                        
                    # 过滤掉一些明显不是学术内容的链接
                    if any(skip in url.lower() for skip in [
                        'javascript:', 'mailto:', '#', 'bing.com', 'microsoft.com',
                        'cache:', 'webcache.googleusercontent.com', 'translate.google.com'
                    ]):
                        continue
                        
                    # 获取标题
                    title = link_element.text.strip() or element.text.strip()
                    if not title or len(title) < 3:
                        continue
                        
                    # 过滤掉一些明显不是学术内容的标题
                    skip_keywords = ['登录', '注册', '首页', '更多', '搜索', '下载', '购买', '价格']
                    if any(skip in title for skip in skip_keywords):
                        continue
                        
                    seen_urls.add(url)
                    results.append(element)
                    
                except Exception as e:
                    continue
                    
            # 如果没有找到结果，尝试更通用的方法
            if not results:
                try:
                    # 直接查找所有外部链接
                    all_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='http']")
                    
                    for link in all_links:
                        try:
                            url = link.get_attribute('href')
                            if (url and 
                                not any(domain in url.lower() for domain in ['bing.com', 'microsoft.com']) and
                                url not in seen_urls and url not in self.scraped_urls):
                                
                                title = link.text.strip()
                                if title and len(title) > 3:
                                    seen_urls.add(url)
                                    results.append(link)
                        except:
                            continue
                                
                except Exception as e:
                    self.logger.error(f"备用提取方法失败: {e}")
                    
            return results[:10]  # 限制每页最多10个结果
            
        except Exception as e:
            self.logger.error(f"提取搜索结果失败: {e}")
            return []
            
    def extract_content_by_clicking(self, result_element):
        """通过点击条目来获取内容"""
        try:
            # 获取原始窗口句柄
            original_window = self.driver.current_window_handle
            
            # 在新标签页中打开链接
            link_element = result_element.find_element(By.TAG_NAME, "a")
            url = link_element.get_attribute('href')
            
            if not url or url in self.scraped_urls:
                return None, None
                
            # 使用Ctrl+Click在新标签页中打开
            actions = ActionChains(self.driver)
            actions.key_down('\ue009').click(link_element).key_up('\ue009').perform()
            time.sleep(2)
            
            # 切换到新窗口
            WebDriverWait(self.driver, 10).until(
                lambda driver: len(driver.window_handles) > 1
            )
            
            for window_handle in self.driver.window_handles:
                if window_handle != original_window:
                    self.driver.switch_to.window(window_handle)
                    break
                    
            # 等待页面加载，最多50秒
            try:
                WebDriverWait(self.driver, 50).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                self.logger.warning(f"页面加载超时: {url}")
                self.driver.close()
                self.driver.switch_to.window(original_window)
                return None, "无有效内容（页面加载超时）"
                
            time.sleep(2)  # 额外等待确保内容加载完成
            
            # 获取页面标题
            title = self.driver.title.strip()
            if not title:
                title = link_element.text.strip()
                
            # 提取页面内容
            content = self.extract_page_content()
            
            # 关闭当前标签页并返回原窗口
            self.driver.close()
            self.driver.switch_to.window(original_window)
            
            return title, content
            
        except TimeoutException:
            self.logger.warning(f"页面加载超时: {url}")
            try:
                self.driver.close()
                self.driver.switch_to.window(original_window)
            except:
                pass
            return None, "无有效内容（页面加载超时）"
        except Exception as e:
            self.logger.error(f"点击获取内容失败: {e}")
            try:
                self.driver.close()
                self.driver.switch_to.window(original_window)
            except:
                pass
            return None, "无有效内容"
            
    def extract_page_content(self):
        """提取当前页面的主要内容"""
        try:
            # 尝试多种方法提取内容
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.article-content', '.entry-content', '#content',
                '.article-body', '.post-body', '.news-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                try:
                    main_content = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if main_content and main_content.text.strip():
                        break
                except:
                    continue
                    
            # 如果没找到主要内容区域，使用body
            if not main_content:
                main_content = self.driver.find_element(By.TAG_NAME, "body")
                
            # 获取文本内容
            text = main_content.text.strip()
            
            # 清理文本
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = ' '.join(lines)
            #最大30000字
            if len(text) > 30000:
                text = text[:30000] + "..."
                

            return text if text else "无有效内容"
            
        except Exception as e:
            self.logger.error(f"提取页面内容失败: {e}")
            return "无有效内容"
            
    def navigate_to_next_page(self, current_page):
        """导航到下一页"""
        try:
            # 计算下一页的first参数
            next_first = current_page * 10 + 1
            current_keyword_encoded = quote(self.current_keyword)
            next_url = f"https://cn.bing.com/academic/search?q={current_keyword_encoded}&sort=1&first={next_first}&FORM=PENR"
            
            self.driver.get(next_url)
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # 随机延迟
            time.sleep(random.uniform(20, 30))
            
            return True
            
        except Exception as e:
            self.logger.error(f"导航到下一页失败: {e}")
            return False
            
    def save_to_csv(self, data, filename):
        """保存数据到CSV文件"""
        try:
            file_path = os.path.join('data', filename)
            
            # 检查文件是否存在，决定是否写入表头
            file_exists = os.path.exists(file_path)
            
            with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['keyword', 'title', 'url', 'content', 'scraped_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    
                for item in data:
                    writer.writerow({
                        'keyword': self.current_keyword,
                        'title': item['title'],
                        'url': item['url'],
                        'content': item['content'],
                        'scraped_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
            self.logger.info(f"数据已保存到 {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存CSV文件失败: {e}")
            
    def crawl_keyword(self, keyword, csv_filename):
        """爬取单个关键词的所有结果"""
        self.current_keyword = keyword
        all_results = []
        page = 0
        max_pages = 50  # 限制最大页数，避免无限循环
        empty_page_count = 0  # 连续空页面计数器
        
        if not self.search_keyword(keyword):
            return False
            
        while page < max_pages:
            try:
                self.logger.info(f"正在爬取关键词 '{keyword}' 第 {page + 1} 页")
                
                # 提取当前页结果元素
                result_elements = self.extract_search_results()
                
                if not result_elements:
                    empty_page_count += 1
                    self.logger.info(f"关键词 '{keyword}' 第 {page + 1} 页没有找到结果，连续空页面数: {empty_page_count}")
                    
                    # 如果连续5页都没有结果，停止爬取
                    if empty_page_count >= 5:
                        self.logger.info(f"关键词 '{keyword}' 连续5页无有效结果，停止爬取")
                        break
                        
                    # 导航到下一页
                    page += 1
                    if not self.navigate_to_next_page(page):
                        break
                    continue
                else:
                    # 重置空页面计数器
                    empty_page_count = 0
                    
                # 点击每个结果条目获取内容
                page_results = []
                for element in result_elements:
                    try:
                        # 滚动到元素可见位置
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(1)
                        
                        # 模拟鼠标悬停
                        actions = ActionChains(self.driver)
                        actions.move_to_element(element).perform()
                        time.sleep(random.uniform(0.5, 1.5))
                        
                        # 通过点击获取内容
                        title, content = self.extract_content_by_clicking(element)
                        
                        if title and content:
                            # 获取URL用于去重
                            link_element = element.find_element(By.TAG_NAME, "a")
                            url = link_element.get_attribute('href')
                            
                            if url and url not in self.scraped_urls:
                                self.scraped_urls.add(url)
                                page_results.append({
                                    'title': title,
                                    'url': url,
                                    'content': content
                                })
                                
                        # 随机延迟，模拟人类操作
                        time.sleep(random.uniform(2, 4))
                        
                    except Exception as e:
                        self.logger.warning(f"处理结果条目时出错: {e}")
                        continue
                        
                # 保存当前页数据
                if page_results:
                    all_results.extend(page_results)
                    self.save_to_csv(page_results, csv_filename)
                    self.logger.info(f"第 {page + 1} 页成功获取 {len(page_results)} 条结果")
                else:
                    self.logger.info(f"第 {page + 1} 页没有获取到有效结果")
                    
                # 导航到下一页
                page += 1
                if not self.navigate_to_next_page(page):
                    break
                    
            except Exception as e:
                self.logger.error(f"爬取关键词 '{keyword}' 第 {page + 1} 页时出错: {e}")
                break
                
        self.logger.info(f"关键词 '{keyword}' 爬取完成，共获取 {len(all_results)} 条结果")
        return True
        
    def run(self, keyword_file_path):
        """运行爬虫，处理指定的关键词文件"""
        try:
            # 设置浏览器
            self.setup_driver()
            
            # 检查关键词文件是否存在
            if not os.path.exists(keyword_file_path):
                self.logger.error(f"关键词文件不存在: {keyword_file_path}")
                return
                
            self.logger.info(f"处理关键词文件: {keyword_file_path}")
            
            # 生成CSV文件名（去掉.txt后缀）
            filename = os.path.basename(keyword_file_path)
            csv_filename = filename.replace('.txt', '.csv')
            
            # 加载已存在的URL到去重集合
            self.load_existing_urls(csv_filename)
            
            # 读取关键词
            keywords = self.read_keywords(keyword_file_path)
            
            for keyword in keywords:
                try:
                    # 爬取关键词
                    if self.crawl_keyword(keyword, csv_filename):
                        # 移动到已完成
                        self.move_keyword_to_finished(keyword_file_path, keyword)
                    else:
                        self.logger.error(f"关键词 '{keyword}' 爬取失败")
                        
                except Exception as e:
                    self.logger.error(f"处理关键词 '{keyword}' 时出错: {e}")
                    continue
                    
            # 检查文件是否还存在
            if not os.path.exists(keyword_file_path):
                self.logger.info(f"关键词文件 {filename} 已处理完成")
                
        except Exception as e:
            self.logger.error(f"运行爬虫时出错: {e}")
        finally:
            self.cleanup()
            if self.driver:
                self.driver.quit()
            self.logger.info("爬虫运行结束")


if __name__ == "__main__":
    import sys
    
    # 检查是否提供了关键词文件路径参数
    if len(sys.argv) < 2:
        print("使用方法: python bing_academic_crawler.py <关键词文件路径>")
        print("示例: python bing_academic_crawler.py keywords/第六代移动通讯.txt")
        print("示例: python bing_academic_crawler.py keywords/脑机接口.txt")
        sys.exit(1)
    
    keyword_file_path = sys.argv[1]
    crawler = BingAcademicCrawler()
    crawler.run(keyword_file_path)
