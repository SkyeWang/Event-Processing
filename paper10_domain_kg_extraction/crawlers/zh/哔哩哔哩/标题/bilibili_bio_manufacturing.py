import requests
import time
import csv
import re
import random
import signal
import sys
from typing import List, Dict
from bio_manufacturing_keywords import bio_manufacturing_keywords

class BilibiliBioManufacturingAPI:
    """B站生物制造视频爬虫类，使用API接口爬取生物制造相关视频"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site"
        })
        self.video_data: List[Dict[str, str | int]] = []
        self.crawled_urls = set()
        self.last_save_time = time.time()
        self.save_interval = 30  # 30秒保存一次，方便测试
        
        # 注册信号处理函数
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """信号处理函数，在用户中断时保存数据"""
        print("\n正在保存数据，请稍候...")
        self.save_to_csv()
        print("数据保存成功！")
        sys.exit(0)
    
    def format_number(self, num_str: str) -> int:
        """格式化数字字符串为整数"""
        if not num_str or num_str == "0":
            return 0
        if "万" in num_str:
            num = re.findall(r"\d+\.?\d*", num_str)[0]
            return int(float(num) * 10000)
        elif "亿" in num_str:
            num = re.findall(r"\d+\.?\d*", num_str)[0]
            return int(float(num) * 100000000)
        else:
            return int(re.sub(r"\D", "", num_str))
    
    def search_videos(self, keyword: str, page: int = 1) -> List[Dict]:
        """通过API搜索视频"""
        url = f"https://api.bilibili.com/x/web-interface/search/all?keyword={keyword}&page={page}"
        
        try:
            time.sleep(random.uniform(1.0, 3.0))
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == 0:
                result_data = data.get("data", {})
                result = result_data.get("result", {})
                
                if isinstance(result, dict):
                    video_results = result.get("video", [])
                    return video_results
                else:
                    return []
            else:
                print(f"API返回错误：{data.get('message', '未知错误')}")
                return []
        except Exception as e:
            print(f"搜索请求失败：{e}")
            return []
    
    def parse_video_data(self, video_item: Dict) -> None:
        """解析视频数据"""
        try:
            
            bvid = video_item.get("bvid", "")
            if not bvid:
                return
            
            video_link = f"https://www.bilibili.com/video/{bvid}"
            
            if video_link in self.crawled_urls:
                return
            
            self.crawled_urls.add(video_link)
            
            video_title = video_item.get("title", "")
            if not video_title:
                video_title = video_item.get("arcurl", "")
            
            if not video_title or len(video_title) < 3:
                return
            
            author = video_item.get("author", "未知UP主")
            play_count = self.format_number(str(video_item.get("play", "0")))
            danmu_count = self.format_number(str(video_item.get("video_review", "0")))
            
            self.video_data.append({
                "视频标题": video_title,
                "UP主": author,
                "播放量": play_count,
                "弹幕数": danmu_count,
                "视频链接": video_link
            })
            print(f"已提取第{len(self.video_data)}条：{video_title[:30]}...")
            
            # 定期保存数据
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save_to_csv()
                self.last_save_time = current_time
            
        except Exception as e:
            print(f"解析视频数据时出错：{e}")
    
    def save_to_csv(self) -> None:
        """将爬取的视频数据保存为CSV文件"""
        if not self.video_data:
            print("无视频数据可保存")
            return
        
        csv_headers = ["视频标题", "UP主", "播放量", "弹幕数", "视频链接"]
        save_path = "bilibili_bio_manufacturing.csv"
        
        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(self.video_data)
            print(f"数据保存成功！共 {len(self.video_data)} 条视频数据，文件路径：{save_path}")
        except Exception as e:
            print(f"数据保存失败：{e}")
    
    def run(self) -> None:
        """爬虫主执行函数"""
        print("开始爬取B站生物制造相关视频...")
        
        keywords = bio_manufacturing_keywords
        
        for keyword in keywords:
            print(f"\n正在搜索关键词：{keyword}")
            page = 1
            
            while True:
                print(f"正在爬取第 {page} 页... (当前已获取 {len(self.video_data)} 条)")
                
                results = self.search_videos(keyword, page)
                
                if not results:
                    print(f"第 {page} 页没有数据，停止翻页")
                    break
                
                current_count = len(self.video_data)
                
                for result in results:
                    self.parse_video_data(result)
                
                if len(self.video_data) == current_count:
                    print(f"第 {page} 页没有新数据，停止翻页")
                    break
                
                page += 1
                
                if page > 200:
                    print(f"已翻页200次，停止搜索关键词：{keyword}")
                    break
        
        self.save_to_csv()
        time.sleep(2)
        print("B站生物制造视频数据爬取完成！")

if __name__ == "__main__":
    bilibili_spider = BilibiliBioManufacturingAPI()
    bilibili_spider.run()
