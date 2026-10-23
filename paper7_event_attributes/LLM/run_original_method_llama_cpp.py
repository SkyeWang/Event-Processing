# 论文七：将原始两次调用流程接到本地 Qwen/llama.cpp 服务。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Run the author's two-call extraction method with a local llama.cpp model.

The prompts, examples, schema, language decision, and response parser all come
from the unmodified llm_information_extraction.py. This file only replaces
ChatGLM's model.chat transport and adds JSONL input/output for the full corpus.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_information_extraction import (
    CLS_PATTERN,
    IE_PATTERN,
    clean_response,
    init_prompts,
    schema,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "original_sentences_full.jsonl"
DEFAULT_OUTPUT = HERE / "output" / "original_method_qwen35_9b_q8_full.jsonl"


# 适配层只转换聊天接口；不替换作者的提示词、schema 或解析函数。
class LlamaCppChat:
    """Expose the original model.chat(query, history=...) call shape."""

    def __init__(self, server_url, timeout, retries):
        self.endpoint = server_url.rstrip("/") + "/v1/chat/completions"
        self.timeout = timeout
        self.retries = retries

    # 将历史问答交替转换为 user/assistant 消息；网络故障按退避策略重试。
    def chat(self, _tokenizer, query, history):
        messages = []
        for old_query, old_answer in history:
            messages.append({"role": "user", "content": old_query})
            messages.append({"role": "assistant", "content": old_answer})
        messages.append({"role": "user", "content": query})

        # ChatGLM-6B's chat() defaults: do_sample=True, top_p=0.7,
        # temperature=0.95, max_length=2048. max_tokens is a generation cap,
        # not an exact equivalent of ChatGLM's total-sequence max_length.
        payload = {
            "messages": messages,
            "stream": False,
            "temperature": 0.95,
            "top_p": 0.7,
            "max_tokens": 2048,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                answer = data["choices"][0]["message"]["content"]
                if not isinstance(answer, str) or not answer:
                    raise ValueError(f"Empty model response: {data!r}")
                return answer, history + [(query, answer)]
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                # A malformed prompt/request will not improve on retry.
                if exc.code < 500 or attempt == self.retries:
                    raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"Cannot reach llama.cpp at {self.endpoint}: {exc}") from exc
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


# 严格检查模型分类为“中文”或“英文”，然后使用对应属性字段。
def extract_one(sentence, settings, model):
    """Follow the author's inference() control flow without altering prompts."""
    sentence_with_cls_prompt = CLS_PATTERN.format(sentence)
    cls_res, _ = model.chat(
        None, sentence_with_cls_prompt, history=settings["cls_pre_history"]
    )
    if cls_res not in schema:
        raise ValueError(
            f"The type model inferenced {cls_res!r} which is not in schema dict, exited."
        )

    properties_str = ", ".join(schema[cls_res])
    schema_str_list = f"“{cls_res}”({properties_str})"
    sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)
    ie_res, _ = model.chat(
        None, sentence_with_ie_prompt, history=settings["ie_pre_history"]
    )
    return cls_res, ie_res, clean_response(ie_res)


# 续跑依据已落盘记录的 ID；坏行报错，避免静默丢失实验记录。
def completed_ids(path):
    ids = set()
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"Bad resume file {path}, line {number}") from exc
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--first", type=int, default=None, help="Only consider the first N input records, including after --resume")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N new records in this invocation; for validation only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.first is not None and args.first < 1:
        parser.error("--first must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.retries < 0 or args.timeout < 1:
        parser.error("--retries must be nonnegative and --timeout positive")
    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.output.exists() and not args.resume:
        parser.error(f"Output exists: {args.output}. Pass --resume or choose a new path")

    # --first 限制总输入范围，--limit 限制本轮新增条数，续跑时含义不同。
    done = completed_ids(args.output) if args.resume and args.output.exists() else set()
    settings = init_prompts()
    model = LlamaCppChat(args.server_url, args.timeout, args.retries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    input_count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "a" if args.resume else "x", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            input_count += 1
            if args.first is not None and input_count > args.first:
                break
            try:
                record = json.loads(line)
                record_id = record["id"]
                sentence = record["text"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Bad input file {args.input}, line {line_number}") from exc
            if record_id in done:
                continue
            try:
                cls_res, raw_response, parsed = extract_one(sentence, settings, model)
            except Exception as exc:
                raise RuntimeError(f"Failed on {record_id} (input line {line_number})") from exc
            result = {
                "id": record_id,
                "source_line": record.get("source_line"),
                "text": sentence,
                "class_response": cls_res,
                "raw_response": raw_response,
                "parsed": parsed,
            }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            # 每句立即刷盘，正常中断后可按 ID 继续；原始回复也一并保留。
            destination.flush()
            count += 1
            print(f"[{count}] {record_id}: {cls_res}; JSON={isinstance(parsed, (dict, list))}", flush=True)
            if args.limit is not None and count >= args.limit:
                break
    print(f"Completed {count} new records; output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
