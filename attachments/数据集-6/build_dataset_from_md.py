# -*- coding: utf-8 -*-
"""从《收集结果.md》生成本项目代码所需的数据集。

产出（全部为新建，不覆盖任何原有文件）：
1. 数据集/ch/*.csv、数据集/en/*.csv                    —— 任务书规定的交付格式
2. data/triplets/{ch,en}/all_triplets.jsonl            —— 合并三元组（对应 process_all_ch_data.py 的输出）
3. data/splits/ch/、data/splits/en_types/              —— 80/10/10 切分（对应 split_large_data.py 的输出）
4. data/entity_frequencies.json 等 4 个频率表           —— 对应 calculate_entity_influence.py 的输出

只依赖标准库；切分随机种子固定为 42，可重复运行、结果一致。
"""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / '收集结果.md'
CSV_DIR = ROOT / '数据集'
DATA_DIR = ROOT / 'data'

# 论文领域名（收集结果.md 中的"## 领域N：xxx"）→（中文文件名关键词, 英文文件名关键词）
DOMAIN_KEYS = {
    '脑机接口': ('BCI', 'BCI'),
    '6G通信': ('6G', '6G'),
    '新能源': ('新能源', 'new_energy'),
    '具身智能': ('具身智能', 'embodied_ai'),
    '集成电路': ('集成电路', 'IC'),
    '量子科技': ('量子科技', 'quantum'),
    '氢能与核聚变': ('氢能核聚变', 'hydrogen_fusion'),
    '生物制造': ('生物制造', 'biomanufacturing'),
}
# 任务书允许的 source_type 取值
VALID_SOURCE_TYPES = {
    'baidu_news', 'bing_academic', 'bilibili_title', 'bilibili_comment',
    'bbc', 'cnn', 'twitter', 'youtube_title', 'youtube_comment',
}
# 属性关系黑名单（任务书 4.1：关系必须是行为动词）
BAD_RELATIONS = {'是', '属于', '位于', '有', '为', '具有', '包括', '系', '在'}

LINE_RE = re.compile(
    r'^(?P<e1>.+?),\s*(?P<rel>.+?),\s*(?P<e2>.+?),\s*'
    r'"(?P<src>.*)"\s*,\s*(?P<st>[A-Za-z_]+)\s*,?\s*$'
)
DOMAIN_RE = re.compile(r'^##\s*领域\d+：(.+)$')


def parse_md():
    """解析收集结果.md，返回 (rows, warnings)。row 字段：domain/lang/e1/rel/e2/src/st。"""
    rows, warnings = [], []
    domain = lang = None
    in_fence = False

    for lineno, raw in enumerate(MD_PATH.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.strip()
        if line.startswith('```'):
            in_fence = not in_fence
            continue
        m = DOMAIN_RE.match(line)
        if m:
            domain = m.group(1).split('（')[0].strip()
            continue
        if line.startswith('### 中文'):
            lang = 'ch'
            continue
        if line.startswith('### 英文'):
            lang = 'en'
            continue
        if not (in_fence and domain and lang):
            continue
        m = LINE_RE.match(line)
        if not m:
            if line:
                warnings.append(f'[解析失败] 第{lineno}行: {line[:70]}')
            continue
        e1, rel, e2 = (m.group(k).strip() for k in ('e1', 'rel', 'e2'))
        if e1 == 'entity1':  # 表头行
            continue
        rows.append(dict(domain=domain, lang=lang, e1=e1, rel=rel, e2=e2,
                         src=m.group('src').strip(), st=m.group('st').strip()))
    return rows, warnings


def validate(rows):
    """质量自检：重复、属性关系、实体长度、source_type 合法性。返回告警列表。"""
    warnings = []
    seen = {}
    for r in rows:
        key = (r['lang'], r['domain'], r['e1'], r['rel'], r['e2'])
        tag = f"{r['lang']}/{r['domain']}: ({r['e1']}, {r['rel']}, {r['e2']})"
        if key in seen:
            warnings.append(f'[重复已去] {tag}')
        seen[key] = True
        if r['rel'] in BAD_RELATIONS:
            warnings.append(f'[属性关系] {tag}')
        if r['st'] not in VALID_SOURCE_TYPES:
            warnings.append(f"[来源类型] {tag} -> {r['st']}")
        if r['lang'] == 'ch':
            for e in (r['e1'], r['e2']):
                if len(e) > 15:
                    warnings.append(f'[实体过长] {tag}')
                    break
        else:
            for e in (r['e1'], r['e2']):
                if len(e.split()) > 5:
                    warnings.append(f'[实体过长] {tag}')
                    break
    return warnings


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = (r['lang'], r['domain'], r['e1'], r['rel'], r['e2'])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_csvs(rows):
    """按 任务书 格式写交付 CSV：数据集/{ch,en}/{领域}_{来源}_with_verbs.csv"""
    groups = {}
    for r in rows:
        groups.setdefault((r['lang'], r['domain']), []).append(r)

    for (lang, domain), items in sorted(groups.items()):
        zh_key, en_key = DOMAIN_KEYS[domain]
        name = f"{zh_key}_综合_with_verbs.csv" if lang == 'ch' else f"{en_key}_mixed_with_verbs.csv"
        out = CSV_DIR / ('ch' if lang == 'ch' else 'en') / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='') as f:  # utf-8 无 BOM
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['entity1', 'relation', 'entity2', 'source_text', 'source_type'])
            for it in items:
                w.writerow([it['e1'], it['rel'], it['e2'], it['src'], it['st']])
    return len(groups)


def write_data_files(rows):
    """生成代码直接消费的 data/ 文件：合并 JSONL → 80/10/10 切分 → 频率表。"""
    rng = random.Random(42)
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r['lang'], []).append(r)

    for lang, items in by_lang.items():
        # 合并 JSONL（字段风格跟随 process_all_ch_data.py）
        triplets_dir = DATA_DIR / 'triplets' / ('ch' if lang == 'ch' else 'en')
        triplets_dir.mkdir(parents=True, exist_ok=True)
        records = [dict(subject=it['e1'], subject_type='unknown', relation=it['rel'],
                        object=it['e2'], object_type='unknown', domain=it['domain'])
                   for it in items]
        with open(triplets_dir / 'all_triplets.jsonl', 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # 80/10/10 切分（跟随 split_large_data.py 的比例；种子固定）
        shuffled = records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        splits = {'train': shuffled[:train_end],
                  'val': shuffled[train_end:val_end],
                  'test': shuffled[val_end:]}
        splits_dir = DATA_DIR / 'splits' / ('ch' if lang == 'ch' else 'en_types')
        splits_dir.mkdir(parents=True, exist_ok=True)
        for name, data in splits.items():
            with open(splits_dir / f'{name}_triplets.jsonl', 'w', encoding='utf-8') as f:
                for rec in data:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # 频率表（跟随 calculate_entity_influence.py：按次数降序的扁平字典）
        ent_cnt, rel_cnt = Counter(), Counter()
        for rec in records:
            ent_cnt[rec['subject']] += 1
            ent_cnt[rec['object']] += 1
            rel_cnt[rec['relation']] += 1
        prefix = '' if lang == 'ch' else 'en_'
        for kind, cnt in (('entity', ent_cnt), ('relation', rel_cnt)):
            with open(DATA_DIR / f'{prefix}{kind}_frequencies.json', 'w', encoding='utf-8') as f:
                json.dump(dict(sorted(cnt.items(), key=lambda x: x[1], reverse=True)),
                          f, ensure_ascii=False, indent=2)


def write_readme(rows, n_groups):
    """交付目录 README：分文件行数统计。"""
    lines = ['# 数据集交付说明\n',
             f'由 `收集结果.md` 经 `build_dataset_from_md.py` 自动生成（切分种子 42，可复现）。\n',
             f'- 三元组总条数：{len(rows)}',
             f'- CSV 文件数：{n_groups}（8 领域 × 中/英）',
             '- 编码均为 UTF-8 无 BOM；表头 `entity1,relation,entity2,source_text,source_type`\n',
             '## 各文件条数\n',
             '| 文件 | 条数 |', '|---|---|']
    for csv_path in sorted(CSV_DIR.rglob('*_with_verbs.csv')):
        n = sum(1 for _ in open(csv_path, encoding='utf-8')) - 1
        lines.append(f'| {csv_path.relative_to(CSV_DIR).as_posix()} | {n} |')
    lines += ['\n## 同步生成的代码用数据文件\n',
              '- `data/triplets/{ch,en}/all_triplets.jsonl`',
              '- `data/splits/ch/{train,val,test}_triplets.jsonl`（80/10/10）',
              '- `data/splits/en_types/{train,val,test}_triplets.jsonl`（80/10/10）',
              '- `data/{en_}entity_frequencies.json`、`data/{en_}relation_frequencies.json`',
              '\n## 规模提示\n',
              '当前为种子数据（每领域几十条），远低于任务书最低验收线'
              '（中文 ≥5000 条/领域、英文 ≥2500 条/领域）。'
              '扩大规模后重跑 `build_dataset_from_md.py` 即可整体再生成。']
    (CSV_DIR / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    rows, warnings = parse_md()
    warnings += validate(rows)
    rows = dedupe(rows)

    if not rows:
        print('未解析到任何三元组，请检查 收集结果.md 格式')
        return
    for r in rows:
        if r['domain'] not in DOMAIN_KEYS:
            print(f"[未知领域] {r['domain']}")
            return

    n_groups = write_csvs(rows)
    write_data_files(rows)
    write_readme(rows, n_groups)

    # 汇总报告
    print(f'共解析 {len(rows)} 条去重后三元组')
    for lang in ('ch', 'en'):
        n = sum(1 for r in rows if r['lang'] == lang)
        doms = sorted({r['domain'] for r in rows if r['lang'] == lang})
        print(f'  {lang}: {n} 条，覆盖 {len(doms)}/8 领域')
    print(f'CSV 分组: {n_groups} 个；输出目录: 数据集/ 与 data/')
    print(f'\n自检告警 {len(warnings)} 条:')
    for w in warnings:
        print(f'  {w}')


if __name__ == '__main__':
    main()
