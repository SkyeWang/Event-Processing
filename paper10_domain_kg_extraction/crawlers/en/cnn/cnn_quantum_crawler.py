#python cnn_quantum_crawler.py
import time
import random
import csv
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

cnn_csv_path = 'cnn_quantum.csv'
save_interval = 30
max_pages_per_keyword = 5

CHROME_DRIVER_PATH = r'D:\Gemini Chrome下载\chrome-win64\chromedriver-win64\chromedriver-win64\chromedriver.exe'
CHROME_BINARY_PATH = r'D:\Gemini Chrome下载\chrome-win64\chrome-win64\chrome.exe'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


def load_keywords():
    """从txt文件加载关键词"""
    keywords = []
    with open('quantum_keywords_english.txt', 'r', encoding='utf-8') as f:
        for line in f:
            keyword = line.strip()
            if keyword:
                keywords.append(keyword)
    return keywords


def setup_chrome_driver():
    """配置Chrome浏览器"""
    options = ChromeOptions()
    
    options.binary_location = CHROME_BINARY_PATH
    
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-infobars")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-breakpad")
    options.add_argument("--disable-component-extensions-with-background-pages")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-domain-reliability")
    options.add_argument("--disable-features=TranslateUI")
    options.add_argument("--disable-hang-monitor")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-prompt-on-repost")
    options.add_argument("--disable-sync")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--no-first-run")
    options.add_argument("--safebrowsing-disable-auto-update")
    options.add_argument("--enable-automation=false")
    options.add_argument("--password-store=basic")
    options.add_argument("--use-mock-keychain")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    
    chrome_service = ChromeService(executable_path=CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=chrome_service, options=options)
    
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
            Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
            window.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'permissions', {get: () => ({query: () => Promise.resolve({state: 'prompt'})})});
        '''
    })
    
    return driver


def check_captcha(driver):
    """检测是否出现人机验证"""
    try:
        page_source = driver.page_source.lower()
        captcha_indicators = [
            'captcha',
            'verify you are human',
            'i\'m not a robot',
            'unusual traffic',
            'security check',
            'human verification'
        ]
        
        for indicator in captcha_indicators:
            if indicator in page_source:
                return True
        
        return False
    except:
        return False


def search_cnn(driver, keyword):
    """使用Google搜索CNN相关文章，支持翻页"""
    search_url = f'https://www.google.com/search?q={keyword}+site:cnn.com'
    
    try:
        driver.get(search_url)
        time.sleep(random.uniform(8, 12))
        
        if check_captcha(driver):
            print(f"  ⚠️  检测到人机验证！请手动完成验证后按回车继续...")
            input()
            time.sleep(2)
            
    except Exception as e:
        print(f"  访问搜索页面失败: {e}")
        return []
    
    all_articles = []
    page_num = 1
    
    while page_num <= max_pages_per_keyword:
        print(f"  第 {page_num} 页...")
        
        if check_captcha(driver):
            print(f"  ⚠️  检测到人机验证！请手动完成验证后按回车继续...")
            input()
            time.sleep(2)
        
        html = driver.page_source
        
        soup = BeautifulSoup(html, 'html.parser')
        
        selectors = [
            ('div', 'g'),
            ('div', 'MjjYud'),
            ('div', 'tF2Cxc'),
            ('div', 'yuRUbf'),
        ]
        
        search_results = []
        for tag, class_name in selectors:
            results = soup.find_all(tag, class_=class_name)
            if results:
                search_results = results
                print(f"    使用选择器: {tag}.{class_name}, 找到 {len(results)} 个结果")
                break
        
        if not search_results:
            print(f"    未找到搜索结果，尝试其他方法...")
            all_divs = soup.find_all('div')
            for div in all_divs:
                links = div.find_all('a')
                if links and 'cnn.com' in str(div):
                    search_results.append(div)
            print(f"    通过链接查找找到 {len(search_results)} 个结果")
        
        for result in search_results:
            try:
                title_elem = result.find('h3')
                if not title_elem:
                    title_elem = result.find('div', class_='DKV0Md')
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                
                link_elem = title_elem.find('a') if title_elem.find('a') else result.find('a')
                if not link_elem:
                    continue
                
                article_url = link_elem.get('href', '')
                
                if '/url?q=' in article_url:
                    article_url = article_url.split('/url?q=')[1].split('&')[0]
                
                if 'cnn.com' not in article_url:
                    continue
                
                publish_time = "未知时间"
                time_elem = result.find('span', class_='MUxGbd')
                if time_elem:
                    publish_time = time_elem.get_text(strip=True)
                
                all_articles.append({
                    'title': title,
                    'url': article_url,
                    'publish_time': publish_time,
                    'keyword': keyword
                })
                
                print(f"    找到CNN文章: {title[:60]}...")
                
            except Exception as e:
                continue
        
        if page_num < max_pages_per_keyword:
            try:
                next_button_selectors = [
                    '//a[@aria-label="Next"]',
                    '//a[@id="pnnext"]',
                    '//span[text()="Next"]/parent::a',
                ]
                
                next_button = None
                for selector in next_button_selectors:
                    try:
                        next_button = driver.find_element(By.XPATH, selector)
                        if next_button and next_button.is_displayed():
                            break
                    except:
                        continue
                
                if next_button:
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(random.uniform(8, 12))
                    
                    if check_captcha(driver):
                        print(f"  ⚠️  检测到人机验证！请手动完成验证后按回车继续...")
                        input()
                        time.sleep(2)
                    
                    page_num += 1
                else:
                    print(f"    没有找到下一页按钮，停止翻页")
                    break
            except Exception as e:
                print(f"    翻页失败: {e}")
                break
        else:
            break
    
    print(f"  总共找到 {len(all_articles)} 篇CNN文章")
    return all_articles


def save_data_to_csv(data, file_path):
    """保存数据到CSV文件"""
    if not data:
        return
    
    file_exists = False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_exists = True
    except:
        pass
    
    mode = 'a' if file_exists else 'w'
    
    with open(file_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['标题', '发布时间', '匹配关键词', '文章链接'])
        
        for item in data:
            writer.writerow([
                item['title'],
                item['publish_time'],
                item['keyword'],
                item['url']
            ])
    
    print(f"已保存 {len(data)} 条数据到 {file_path}")


def get_crawled_keywords(file_path):
    """获取已经爬取过的关键词"""
    crawled_keywords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    crawled_keywords.add(row[2])
    except:
        pass
    return crawled_keywords


def crawl_cnn_quantum():
    """爬取CNN量子计算相关文章"""
    
    file_exists = False
    try:
        with open(cnn_csv_path, 'r', encoding='utf-8') as f:
            file_exists = True
    except:
        pass
    
    if not file_exists:
        with open(cnn_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['标题', '发布时间', '匹配关键词', '文章链接'])
    
    keywords = load_keywords()
    
    crawled_keywords = get_crawled_keywords(cnn_csv_path)
    print(f"已爬取关键词数: {len(crawled_keywords)}")
    
    remaining_keywords = [k for k in keywords if k not in crawled_keywords]
    print(f"剩余待爬取关键词数: {len(remaining_keywords)}")
    
    if not remaining_keywords:
        print("所有关键词已爬取完成！")
        return
    
    print(f"CNN量子计算爬虫启动（Chrome浏览器版本）")
    print(f"总关键词数: {len(keywords)}")
    print(f"已爬取: {len(crawled_keywords)}")
    print(f"待爬取: {len(remaining_keywords)}")
    print(f"保存间隔: {save_interval}秒")
    print(f"每个关键词最多翻页: {max_pages_per_keyword}页")
    print(f"输出文件: {cnn_csv_path}")
    print(f"使用Google搜索CNN文章")
    print(f"检测到人机验证时会自动暂停，请手动完成验证后按回车继续")
    print("="*60)
    
    driver = setup_chrome_driver()
    
    all_data = []
    last_save_time = time.time()
    total_articles = 0
    failed_count = 0
    
    try:
        for idx, keyword in enumerate(remaining_keywords, 1):
            print(f"\n[{idx}/{len(remaining_keywords)}] 搜索关键词: {keyword}")
            
            articles = search_cnn(driver, keyword)
            print(f"  找到 {len(articles)} 篇CNN文章")
            
            all_data.extend(articles)
            total_articles += len(articles)
            
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                if all_data:
                    save_data_to_csv(all_data, cnn_csv_path)
                    all_data = []
                    last_save_time = current_time
            
            delay = random.uniform(15, 30)
            print(f"  等待 {delay:.1f} 秒后继续...")
            time.sleep(delay)
            
            if failed_count > 10:
                print(f"  连续失败{failed_count}次，暂停60秒...")
                time.sleep(60)
                failed_count = 0
        
        if all_data:
            save_data_to_csv(all_data, cnn_csv_path)
        
        print("\n" + "="*60)
        print(f"爬取完成!")
        print(f"总文章数: {total_articles}")
        print(f"数据保存至: {cnn_csv_path}")
        print("="*60)
    
    finally:
        driver.quit()


if __name__ == "__main__":
    crawl_cnn_quantum()
