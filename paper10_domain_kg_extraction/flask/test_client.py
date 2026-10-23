"""在进程内调用 Flask 的 /extract 接口，并保存返回 JSON。

用途：不启动 0.0.0.0:5000 服务，直接用 Flask 的 test_client 发请求，
方便在命令行/CI 里验证模型加载、语言检测和三元组解析是否正常。

运行前设置环境变量（或在 app.py 里改默认路径）：
  KG_MODEL_PATH   基础模型目录
  KG_LORA_PATH    LoRA 适配器目录
  KG_MAX_NEW_TOKENS  生成上限（快速测试可设 256）

输出：kg-extraction/output/smoke/flask_response.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 让 python flask/test_client.py 能 import 同目录的 app
sys.path.insert(0, HERE)

from app import app, load_model  # noqa: E402

# 两条固定用例：一条中文、一条英文，便于对比语言检测和输出格式
CASES = [
    ("zh", "2025年3月，OpenAI发布GPT-4.5模型，该模型具备多模态推理能力。"),
    ("en", "In 2025, NVIDIA released the Blackwell B200 GPU for AI training."),
]


def main():
    """加载模型，逐条请求 /extract，打印并保存结果。"""
    load_model()
    output = []
    with app.test_client() as client:
        for lang, text in CASES:
            response = client.post("/extract", json={"text": text, "lang": lang})
            payload = response.get_json()
            output.append({"status": response.status_code, "lang": lang, "text": text, "response": payload})
            print(lang, response.status_code, json.dumps(payload, ensure_ascii=False)[:400])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)), "output", "smoke")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "flask_response.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
