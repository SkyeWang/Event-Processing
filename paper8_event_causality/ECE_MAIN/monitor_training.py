# 论文八：只读解析训练日志和 GPU 状态，提供本机监控页面。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Read-only local dashboard for a running ECE training task.

Usage: python monitor_training.py --task-name ece_full_run1 --port 8765
The monitor reads logs and nvidia-smi. It never loads or changes the model.
"""

import argparse
import csv
import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOSS_RE = re.compile(r"Epoch: (\d+), Loss: ([0-9.eE+-]+)")
METRIC_RE = re.compile(
    r"Epoch: (\d+), precision: ([0-9.eE+-]+), recall: ([0-9.eE+-]+), "
    r"f1: ([0-9.eE+-]+), tp: (\d+), fp: (\d+), fn:(\d+)"
)


def iso_time(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def process_state(task_name):
    try:
        import psutil
    except ImportError:
        return {"training_pid": None, "pipeline_pid": None, "started_at": None}
    training_pid = pipeline_pid = started_at = None
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            cmd = proc.info["cmdline"] or []
            if "--task_name" in cmd and cmd[cmd.index("--task_name") + 1] == task_name and any(
                part.endswith("train.py") for part in cmd
            ):
                training_pid = proc.info["pid"]
                started_at = iso_time(proc.info["create_time"])
            elif "--task-name" in cmd and cmd[cmd.index("--task-name") + 1] == task_name and any(
                part.endswith("run_pipeline.py") for part in cmd
            ):
                pipeline_pid = proc.info["pid"]
        except (IndexError, OSError, psutil.Error):
            continue
    return {"training_pid": training_pid, "pipeline_pid": pipeline_pid, "started_at": started_at}


# 从逐轮日志恢复 loss、P/R/F1 和最佳轮次；不参与训练或权重选择。
def parse_log(path):
    result = {"config": {}, "loss": [], "metrics": [], "events": [], "updated_at": None}
    if not path.is_file():
        return result
    result["updated_at"] = iso_time(path.stat().st_mtime)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " - INFO - " not in line:
            continue
        prefix, message = line.split(" - INFO - ", 1)
        try:
            timestamp = datetime.strptime(prefix.split(" - ", 1)[0], "%Y-%m-%d %H:%M:%S,%f").astimezone().isoformat()
        except ValueError:
            timestamp = result["updated_at"]
        if not result["loss"] and ": " in message and not message.startswith("Epoch:"):
            key, value = message.split(": ", 1)
            if key in {"task_name", "data_path", "epoches", "batch_size", "lr", "warmup_rate", "thresh"}:
                result["config"][key] = value.strip()
        match = LOSS_RE.search(message)
        if match:
            result["loss"].append({"epoch": int(match[1]), "value": float(match[2]), "time": timestamp})
            result["events"].append({"time": timestamp, "epoch": int(match[1]),
                                     "message": f"第 {match[1]} 轮完成 · Loss {float(match[2]):.3f}"})
            continue
        match = METRIC_RE.search(message)
        if match:
            epoch, f1 = int(match[1]), float(match[4])
            result["metrics"].append({"epoch": epoch, "precision": float(match[2]),
                                      "recall": float(match[3]), "f1": f1,
                                      "tp": int(match[5]), "fp": int(match[6]), "fn": int(match[7])})
            for event in reversed(result["events"]):
                if event.get("epoch") == epoch:
                    event["message"] += f" · 验证 ECE F1 {f1:.2f}%"
                    break
        if "Best checkpoint updated:" in message:
            result["events"].append({"time": timestamp, "message": message.strip()})
    result["events"] = result["events"][-9:][::-1]
    return result


class GpuSampler:
    def __init__(self):
        self.history = deque(maxlen=1200)
        self.error = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="gpu-sampler").start()

    def _loop(self):
        query = "index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while True:
            try:
                completed = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=8, check=True, creationflags=flags,
                )
                row = next(csv.reader(completed.stdout.splitlines()))
                values = [item.strip() for item in row]

                def number(index):
                    try:
                        return float(values[index])
                    except (IndexError, ValueError):
                        return None

                sample = {"time": iso_time(time.time()), "index": values[0], "name": values[1],
                          "utilization": number(2), "memory_used": number(3), "memory_total": number(4),
                          "power": number(5), "power_limit": number(6), "temperature": number(7)}
                with self.lock:
                    self.history.append(sample)
                    self.error = None
            except (OSError, subprocess.SubprocessError, StopIteration) as exc:
                with self.lock:
                    self.error = str(exc)
            time.sleep(3)

    def snapshot(self):
        with self.lock:
            points = list(self.history)
            return {"latest": points[-1] if points else None, "history": points[-180:], "error": self.error}


def collect_status(task_name, sampler):
    log = parse_log(ROOT / "log" / f"{task_name}.log")
    proc = process_state(task_name)
    output = ROOT / "res" / task_name
    latest_epoch = log["loss"][-1]["epoch"] if log["loss"] else 0
    try:
        total = int(log["config"].get("epoches", 0))
    except ValueError:
        total = 0
    best = None
    best_path = ROOT / "models" / task_name / "best.json"
    if best_path.is_file():
        try:
            best = json.loads(best_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if (output / "graph" / "manifest.json").is_file():
        stage = "complete"
    elif (output / "eval_best.txt").is_file():
        stage = "postprocessing"
    elif proc["training_pid"] is not None:
        stage = "training"
    elif proc["pipeline_pid"] is not None:
        stage = "pipeline"
    elif latest_epoch:
        stage = "stopped"
    else:
        stage = "waiting"
    durations = []
    losses = log["loss"]
    recent_losses = losses[-6:]
    for prev, curr in zip(recent_losses, recent_losses[1:]):
        try:
            t0 = datetime.fromisoformat(prev["time"].replace(" ", "T")).timestamp()
            t1 = datetime.fromisoformat(curr["time"].replace(" ", "T")).timestamp()
            if 0 < t1 - t0 < 86400:
                durations.append(t1 - t0)
        except ValueError:
            pass
    return {"task_name": task_name, "server_time": iso_time(time.time()), "stage": stage,
            "completed_epochs": latest_epoch, "total_epochs": total,
            "current_epoch": min(latest_epoch + 1, total) if stage == "training" and total else None,
            "average_epoch_seconds": round(sum(durations) / len(durations), 1) if durations else None,
            "config": log["config"], "loss": losses, "metrics": log["metrics"],
            "latest_metric": log["metrics"][-1] if log["metrics"] else None,
            "best": best, "events": log["events"], "log_updated_at": log["updated_at"],
            "process": proc, "gpu": sampler.snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.task_name.isidentifier() or not 1 <= args.port <= 65535:
        parser.error("invalid task name or port")
    sampler = GpuSampler()
    page = (ROOT / "monitor_training.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body, mime = page, "text/html; charset=utf-8"
            elif path == "/api/status":
                body = json.dumps(collect_status(args.task_name, sampler), ensure_ascii=False).encode("utf-8")
                mime = "application/json; charset=utf-8"
            elif path == "/api/health":
                body, mime = b'{"ok":true}', "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *values):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Monitoring {args.task_name} at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
