# 论文八：依次完成环境检查、训练、验证集选权重、测试推理和制图。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Run author ECE training, best-checkpoint test inference, and direct graph export."""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, cwd):
    print("\nRunning:", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="unique identifier for this run")
    parser.add_argument("--subset", default="sample700")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--allow-cpu", action="store_true", help="CPU smoke run only; formal run requires CUDA")
    parser.add_argument("--tfidf-threshold", type=float, default=None,
                        help="optional character TF-IDF approximate deduplication threshold")
    args = parser.parse_args()
    if not args.task_name.isidentifier() or not args.subset.isidentifier():
        parser.error("task-name and subset must be identifiers")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")
    if args.tfidf_threshold is not None and not 0 <= args.tfidf_threshold <= 1:
        parser.error("tfidf-threshold must be between 0 and 1")

    # 所有数据和输出相对于本代码目录定位；任务名必须唯一，以免覆盖已有实验。
    subset = ROOT / "data" / "subsets" / args.subset
    for path in (ROOT / "models" / args.task_name, ROOT / "res" / args.task_name,
                 ROOT / "log" / f"{args.task_name}.log"):
        if path.exists():
            parser.error(f"task output already exists; choose a new task-name: {path}")
    check = [sys.executable, str(ROOT / "check_environment.py"), "--subset", args.subset]
    if not args.allow_cpu:
        check.append("--require-cuda")
    run(check, ROOT)

    with (subset / "processed_data" / "train.pickle").open("rb") as file:
        train_size = len(pickle.load(file))
    if train_size < args.batch_size:
        parser.error("batch-size exceeds the training split size")
    remainder = train_size % args.batch_size
    if remainder:
        print(f"Author Data_data drops the final {remainder} training samples per epoch.", flush=True)

    # 这里传给作者 train.py 的路径以 src 为工作目录；公开参数通过命令行覆盖。
    base = [sys.executable, "train.py", "--task_name", args.task_name,
            "--debug", "0", "--data_path", f"../data/subsets/{args.subset}",
            "--train_data_path", f"../data/subsets/{args.subset}/processed_data/train.pickle",
            "--dev_data_path", f"../data/subsets/{args.subset}/processed_data/dev.pickle",
            "--test_data_path", f"../data/subsets/{args.subset}/processed_data/test.pickle",
            "--bert_path", "../data/bert-base-zh", "--batch_size", str(args.batch_size),
            "--lr", str(args.lr), "--hidden_size", "768"]
    # 训练时仅用 dev ECE F1 更新 best.pickle；test 不参与权重选择。
    run(base + ["--training", "1", "--epoches", str(args.epochs)], ROOT / "src")
    best = ROOT / "models" / args.task_name / "best.pickle"
    if not best.is_file():
        raise RuntimeError(f"Training ended without best checkpoint: {best}")
    # 训练结束后，用同一个最佳权重对 test 划分评估一次。
    run(base + ["--training", "0", "--model_name", "best"], ROOT / "src")

    output = ROOT / "res" / args.task_name
    predictions = output / "eval_best.txt"
    if not predictions.is_file():
        raise RuntimeError(f"Test inference did not produce: {predictions}")
    pairs_csv = output / "predictions.csv"
    run([sys.executable, str(ROOT / "postprocess_graph.py"), "prepare",
         "--predictions", str(predictions), "--raw-data", str(subset / "raw_data.txt"),
         "--output", str(pairs_csv)], ROOT)
    graph = output / "graph"
    # 图包含全部预测；精确重复项合并不等于人工审核或正确性判定。
    graph_command = [sys.executable, str(ROOT / "postprocess_graph.py"), "export",
                     "--reviewed", str(pairs_csv), "--output-dir", str(graph),
                     "--accept-all", "--plot"]
    if args.tfidf_threshold is not None:
        graph_command += ["--tfidf-threshold", str(args.tfidf_threshold)]
    run(graph_command, ROOT)
    print(json.dumps({"best_checkpoint": str(best), "test_predictions": str(predictions),
                      "event_pairs": str(pairs_csv), "graph_html": str(graph / "graph.html"),
                      "graph_png": str(graph / "graph.png")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
