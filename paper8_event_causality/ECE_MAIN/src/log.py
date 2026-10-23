# 论文八：将训练设置和每轮指标写入本任务日志，供监控与结果核查。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
import logging
from os.path import join, dirname
from pathlib import Path

home = dirname(__file__)

class Log():

    def __init__(self, filename):
        # 新提交目录不含历史 log 文件夹，首次运行时先创建它。
        (Path(home) / '../log').mkdir(parents=True, exist_ok=True)
        # 创建一个logger
        self.logger = logging.getLogger('statisticNew')
        self.logger.setLevel(logging.INFO)

        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(join(home, '../log', filename))
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getLog(self):
        return self.logger
