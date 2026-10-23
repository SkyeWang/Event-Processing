# 论文七：按固定仓库版本下载 GGUF，并校验完整文件的 SHA-256。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Download and verify the Qwen3.5-9B Q8_0 GGUF model.

The Hugging Face downloader resumes interrupted downloads automatically.  This
script only downloads the model; it does not install or invoke llama.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
REPO_ID = "unsloth/Qwen3.5-9B-GGUF"
FILENAME = "Qwen3.5-9B-Q8_0.gguf"
REVISION = "be196fb62de6135dc3f6728b6f5c2a11a181c97e"
EXPECTED_SHA256 = "63774c0100a07c7c572123d5ac62e64f26bfddf8cbb8fa6bfcc195bdc18a4369"
DEFAULT_DIR = ROOT / "models" / "Qwen3.5-9B-Q8_0"
MIN_FREE_GIB = 12


# 流式计算摘要，避免一次把约 9.5 GB 模型读入内存。
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="skip the final full-file checksum (download integrity checks still apply)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(args.output_dir).free / 1024**3
    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB is free; at least {MIN_FREE_GIB} GiB is required."
        )

    print(f"Repository : {REPO_ID}")
    print(f"Revision   : {REVISION}")
    print(f"File       : {FILENAME}")
    print(f"Destination: {args.output_dir.resolve()}")
    print(f"Free space : {free_gib:.2f} GiB")

    # 使用固定 revision，使下载模型与已有实验的量化版本一致。
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            revision=REVISION,
            local_dir=args.output_dir,
        )
    )
    print(f"Downloaded : {downloaded}")
    print(f"Size       : {downloaded.stat().st_size / 1024**3:.2f} GiB")

    if not args.skip_sha256:
        print("Verifying SHA-256 (this reads the complete 9.5 GB file)...")
        actual = sha256(downloaded)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
            )
        print(f"SHA-256    : {actual} (OK)")


if __name__ == "__main__":
    main()
