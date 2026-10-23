# 原附件采集与标注代码

`Spider-February-17/` 保留静态抓取与 CNN、Guardian、Times、Facebook、Instagram 的动态网页采集代码和关键词；`Label-March-27/` 保留数据库读写、三元组重建、实体/关系标注及相应词表。文件头补充了用途说明，原有算法和中文注释保留。

这部分是原项目的数据构建背景代码，不是本次报告实验的必需运行入口。数据库连接仍是原附件的占位参数，表结构和原数据库没有随附件提供；浏览器驱动、网站 DOM 和登录环境也需要另配。`Spider.py` 导入的 `utils` 未在原附件中找到，故保留该缺失事实，不伪造替代实现。`Label-March-27/main.py` 中的 `BioesLabel` 仅保留原作者的单位置赋值实现，没有补造完整 BIOES 标注算法。

原 `Spider.py` 中硬编码的会话 Cookie 已在提交副本移除；没有修改原工作目录。历史依赖见 `requirements.txt`，其中 Pattern 的旧 `pattern.text.en` 接口需要兼容环境。若研究网页采集，应先按实际数据库与浏览器版本完成适配；复跑报告则直接使用上级 `LLM/` 和 `AMR/` 的固定语料入口。
