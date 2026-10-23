from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime
from crawler.spider import run_crawler, run_crawler_stream, get_random_proxy
from crawler.baidu_news_spider import run_baidu_news_crawler_stream
import json
import os
import threading

app = Flask(__name__)

# 全局变量存储当前运行的爬虫实例
current_crawler = None
crawler_lock = threading.Lock()

@app.route('/')
def index():
    # 渲染首页模板
    return render_template('index.html')

@app.route('/data')
def data():
    # 批量接口：一次性返回全部爬取结果
    kw = [k.strip() for k in request.args.get('kw', '').split(',') if k.strip()]
    start = request.args.get('start', '')
    end   = request.args.get('end', '')
    user  = request.args.get('user', '')
    pwd   = request.args.get('pwd', '')
    auto_proxy = request.args.get('auto_proxy', 'false').lower() == 'true'

    # 参数校验
    if not kw or not start or not user or not pwd:
        return jsonify({'error': '参数缺失'}), 400
    if not end:
        end = datetime.today().strftime('%Y-%m-%d')
    try:
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date   = datetime.strptime(end,   '%Y-%m-%d')
    except ValueError as e:
        return jsonify({'error': f'日期格式错误: {e}'}), 400

    proxy = get_random_proxy()
    try:
        results = run_crawler(kw, start_date, end_date, user, pwd, proxy, auto_proxy)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify(results)

@app.route('/stream')
def stream():
    # SSE 流式接口：实时推送爬取到的数据
    source = request.args.get('source', 'twitter')  # twitter 或 baidu
    
    if source == 'baidu':
        return baidu_news_stream()
    else:
        return twitter_stream()

def twitter_stream():
    # Twitter爬虫流式接口
    kw = [k.strip() for k in request.args.get('kw', '').split(',') if k.strip()]
    start = request.args.get('start', '')
    end   = request.args.get('end', '')
    user  = request.args.get('user', '')
    pwd   = request.args.get('pwd', '')
    auto_proxy = request.args.get('auto_proxy', 'false').lower() == 'true'

    if not kw or not start or not user or not pwd:
        return jsonify({'error': '参数缺失'}), 400
    if not end:
        end = datetime.today().strftime('%Y-%m-%d')
    try:
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date   = datetime.strptime(end,   '%Y-%m-%d')
    except ValueError as e:
        return jsonify({'error': f'日期格式错误: {e}'}), 400

    proxy = get_random_proxy()
    def event_stream():
        # 逐条推送
        for item in run_crawler_stream(kw, start_date, end_date, user, pwd, proxy, auto_proxy):
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        # 结束事件
        yield "event: end\n"
        yield "data: {}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

def baidu_news_stream():
    # 百度新闻爬虫流式接口
    global current_crawler
    
    kw = [k.strip() for k in request.args.get('kw', '').split(',') if k.strip()]
    max_results = int(request.args.get('max_results', '100'))
    clear_data = request.args.get('clear_data', 'false').lower() == 'true'

    if not kw:
        return jsonify({'error': '关键词不能为空'}), 400

    # 清除现有数据
    if clear_data:
        try:
            if os.path.exists('baidunews.csv'):
                os.remove('baidunews.csv')
        except:
            pass

    def event_stream():
        global current_crawler
        
        try:
            with crawler_lock:
                current_crawler = run_baidu_news_crawler_stream(kw, max_results)
            
            for item in current_crawler:
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                
        except GeneratorExit:
            # 正常的流结束，不需要特殊处理
            pass
        except Exception as e:
            yield f"event: error\n"
            yield f"data: {{'error': '{str(e)}'}}\n\n"
        finally:
            with crawler_lock:
                current_crawler = None
            # 结束事件
            yield "event: end\n"
            yield "data: {}\n\n"
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/stop_crawler')
def stop_crawler():
    # 停止当前运行的爬虫
    global current_crawler
    
    with crawler_lock:
        if current_crawler:
            try:
                # 对于生成器，我们需要调用其stop方法
                if hasattr(current_crawler, 'stop'):
                    current_crawler.stop()
                else:
                    # 如果是生成器，设置为None让流结束
                    current_crawler = None
                return jsonify({'success': True, 'message': '爬虫停止指令已发送'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        else:
            return jsonify({'success': False, 'error': '当前没有运行的爬虫'}), 404

@app.route('/check_captcha')
def check_captcha():
    # 检查是否需要验证码（由前端轮询调用）
    # 这里简化处理，实际应该在爬虫中实现验证码检测
    return jsonify({'captcha_required': False})

@app.route('/clear_data')
def clear_data():
    # 清除数据文件
    try:
        if os.path.exists('baidunews.csv'):
            os.remove('baidunews.csv')
        if os.path.exists('baidunewsall.csv'):
            os.remove('baidunewsall.csv')
        return jsonify({'success': True, 'message': '数据已清除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask on http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    app.run(debug=True, host='127.0.0.1', port=5000)
