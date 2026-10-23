"""按清单准备外部资源；默认只检查，显式 --copy 才复制到本论文目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """分块校验大文件，避免一次占用模型大小的内存。"""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def inside(root: Path, relative: str) -> Path:
    """资源清单中的路径必须位于指定根目录内。"""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside root: {relative}')
    return path


def main() -> None:
    manifest = json.loads((ROOT / 'assets_manifest.json').read_text(encoding='utf-8'))
    choices = sorted({item['component'] for item in manifest['files']})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT.parents[1],
                        help='原论文工作目录（包含 ECE 和附录）；默认是 submission 的上级')
    parser.add_argument('--component', nargs='+', choices=choices, required=True)
    parser.add_argument('--copy', action='store_true', help='复制缺失资源；默认只检查')
    parser.add_argument('--verify', action='store_true', help='检查已有文件的 SHA-256，模型文件较大')
    args = parser.parse_args()
    selected = [item for item in manifest['files'] if item['component'] in args.component]
    # 先检查全体来源和已有目标，缺资源时在任何复制动作之前退出。
    pending = []
    problems = []
    for item in selected:
        target = inside(ROOT, item['target'])
        source = inside(args.source_root, item['source'])
        if target.exists():
            if not target.is_file() or (item['bytes'] is not None and target.stat().st_size != item['bytes']):
                problems.append(f'Existing target differs: {target}')
            elif args.verify and digest(target) != item['sha256']:
                problems.append(f'Existing target checksum differs: {target}')
            continue
        if not source.is_file() or (item['bytes'] is not None and source.stat().st_size != item['bytes']):
            problems.append(f'Missing or different source: {source}')
        else:
            pending.append((source, target, item))
    if problems:
        raise SystemExit('\n'.join(problems))
    total = sum(source.stat().st_size for source, _, _ in pending)
    print(f'Selected {len(selected)} assets; missing {len(pending)}; copy size {total / 1024**2:.2f} MiB')
    if not args.copy:
        print('Check only. Add --copy to prepare missing assets.')
        return
    for source, target, item in pending:
        # 不覆盖已有资源；部分复制使用独立后缀，校验成功后才发布为正式文件。
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        if temporary.exists():
            raise SystemExit(f'Partial file already exists; inspect it first: {temporary}')
        with source.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        if digest(temporary) != item['sha256']:
            raise SystemExit(f'Checksum mismatch; kept partial file: {temporary}')
        if target.exists():
            raise SystemExit(f'Target appeared during copy; kept partial file: {temporary}')
        temporary.rename(target)
        print(f'Prepared: {target.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
