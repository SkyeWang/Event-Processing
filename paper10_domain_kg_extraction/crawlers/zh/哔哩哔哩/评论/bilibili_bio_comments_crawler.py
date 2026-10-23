import time
import random
import csv
import os
import json
import requests
from datetime import datetime

csv_file_path = 'bilibili_bio_manufacturing.csv'
output_file = 'bilibili_bio_comments.csv'
save_interval = 30
max_comments_per_video = 1000

API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

PROXIES = {
}


def load_video_links():
    """从CSV文件加载视频链接"""
    video_links = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    video_links.append({
                        'title': row[0],
                        'up': row[1],
                        'views': row[2],
                        'danmaku': row[3],
                        'url': row[4]
                    })
    except Exception as e:
        print(f"加载视频链接失败: {e}")
    return video_links


def get_crawled_videos():
    """获取已经爬取过的视频"""
    crawled_videos = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 6:
                    crawled_videos.add(row[5])
    except:
        pass
    return crawled_videos


def bv_to_aid(bv):
    """将BV号转换为aid"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bv}"
    try:
        response = requests.get(url, headers=API_HEADERS, proxies=PROXIES, timeout=10)
        data = response.json()
        if data.get('code') == 0:
            return data['data']['aid']
        else:
            print(f"BV号转换失败: {data.get('message')}")
            return None
    except Exception as e:
        print(f"BV号转换异常: {e}")
        return None


def extract_bv_from_url(url):
    """从视频链接中提取BV号"""
    if 'bilibili.com/video/' in url:
        bv = url.split('bilibili.com/video/')[1].split('?')[0]
        return bv
    return None


def get_video_comments(aid):
    """获取视频评论"""
    comments = []
    page = 0
    count = 0
    
    while count < max_comments_per_video:
        url = f"https://api.bilibili.com/x/v2/reply/main?oid={aid}&type=1&next={page}"
        try:
            response = requests.get(url, headers=API_HEADERS, proxies=PROXIES, timeout=15)
            data = response.json()
            
            if data.get('code') != 0:
                print(f"获取评论失败: {data.get('message')}")
                break
            
            reply_data = data.get('data', {})
            replies = reply_data.get('replies', [])
            
            if not replies:
                break
            
            for reply in replies:
                comment = {
                    'content': reply.get('content', {}).get('message', ''),
                    'user': reply.get('member', {}).get('uname', ''),
                    'like': reply.get('like', 0),
                    'time': datetime.fromtimestamp(reply.get('ctime', 0)).strftime('%Y-%m-%d %H:%M:%S')
                }
                comments.append(comment)
                count += 1
                
                if count >= max_comments_per_video:
                    break
            
            page += 1
            
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"获取评论异常: {e}")
            time.sleep(5)
            continue
    
    return comments


def save_comments_to_csv(comments, file_path):
    """保存评论到CSV文件"""
    if not comments:
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
            writer.writerow(['视频标题', 'UP主', '播放量', '弹幕数', '评论内容', '评论用户', '点赞数', '评论时间', '视频链接'])
        
        for comment in comments:
            writer.writerow([
                comment['video_title'],
                comment['video_up'],
                comment['video_views'],
                comment['video_danmaku'],
                comment['content'],
                comment['user'],
                comment['like'],
                comment['time'],
                comment['video_url']
            ])
    
    print(f"已保存 {len(comments)} 条评论到 {file_path}")


def crawl_bilibili_comments():
    """爬取哔哩哔哩视频评论"""
    
    video_links = load_video_links()
    print(f"总视频数: {len(video_links)}")
    
    crawled_videos = get_crawled_videos()
    print(f"已爬取视频数: {len(crawled_videos)}")
    
    remaining_videos = [video for video in video_links if video['url'] not in crawled_videos]
    print(f"待爬取视频数: {len(remaining_videos)}")
    
    if not remaining_videos:
        print("所有视频已爬取完成！")
        return
    
    print(f"哔哩哔哩生物制造评论爬虫启动")
    print(f"保存间隔: {save_interval}秒")
    print(f"每个视频最多评论数: {max_comments_per_video}")
    print(f"输出文件: {output_file}")
    print("="*60)
    
    all_comments = []
    last_save_time = time.time()
    total_comments = 0
    failed_count = 0
    
    try:
        for idx, video in enumerate(remaining_videos, 1):
            print(f"\n[{idx}/{len(remaining_videos)}] 处理视频: {video['title'][:50]}...")
            print(f"  UP主: {video['up']}")
            print(f"  播放量: {video['views']}")
            print(f"  弹幕数: {video['danmaku']}")
            print(f"  链接: {video['url']}")
            
            bv = extract_bv_from_url(video['url'])
            if not bv:
                print(f"  ❌ 无法提取BV号")
                failed_count += 1
                continue
            
            aid = bv_to_aid(bv)
            if not aid:
                print(f"  ❌ 无法转换为aid")
                failed_count += 1
                continue
            
            print(f"  BV号: {bv}")
            print(f"  aid: {aid}")
            
            comments = get_video_comments(aid)
            print(f"  找到 {len(comments)} 条评论")
            
            for comment in comments:
                comment['video_title'] = video['title']
                comment['video_up'] = video['up']
                comment['video_views'] = video['views']
                comment['video_danmaku'] = video['danmaku']
                comment['video_url'] = video['url']
            
            all_comments.extend(comments)
            total_comments += len(comments)
            
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                if all_comments:
                    save_comments_to_csv(all_comments, output_file)
                    all_comments = []
                    last_save_time = current_time
            
            delay = random.uniform(5, 10)
            print(f"  等待 {delay:.1f} 秒后继续...")
            time.sleep(delay)
            
            if failed_count > 5:
                print(f"  连续失败{failed_count}次，暂停30秒...")
                time.sleep(30)
                failed_count = 0
        
        if all_comments:
            save_comments_to_csv(all_comments, output_file)
        
        print("\n" + "="*60)
        print(f"爬取完成!")
        print(f"总评论数: {total_comments}")
        print(f"数据保存至: {output_file}")
        print("="*60)
    
    except KeyboardInterrupt:
        print("\n爬取被用户中断")
        if all_comments:
            save_comments_to_csv(all_comments, output_file)
            print(f"已保存 {len(all_comments)} 条评论")
    
    except Exception as e:
        print(f"爬取过程中发生错误: {e}")
        if all_comments:
            save_comments_to_csv(all_comments, output_file)
            print(f"已保存 {len(all_comments)} 条评论")


if __name__ == "__main__":
    crawl_bilibili_comments()
