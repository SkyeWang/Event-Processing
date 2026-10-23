#!/usr/bin/env bash
set -euo pipefail

# 论文七 AMR 实验入口：只编排运行环境，不更改 CAMR/JAMR 解析和对齐算法。
# 提交包没有嵌套 .git。复制已随包附带的源码，再按 source_layout.json
# 恢复 Unix 可执行位及软链接（Windows 普通复制不会保留这些元信息）。
mode="${1:-full}"
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
    prepare|smoke|full) ;;
    *) echo "Usage: run_amr.sh {prepare|smoke|full} [--limit N]" >&2; exit 2 ;;
esac
limit=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit)
            if [ "$mode" != full ] || [ "$#" -lt 2 ] ||
               ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--limit requires a positive integer and full mode" >&2
                exit 2
            fi
            limit="$2"
            shift 2
            ;;
        *) echo "Usage: run_amr.sh {prepare|smoke|full} [--limit N]" >&2; exit 2 ;;
    esac
done

mkdir -p /work /results
for project in camr jamr; do
    if [ ! -e "/work/$project/.submission-source-ready" ]; then
        mkdir -p "/work/$project"
        cp -a "/source/$project/$project/." "/work/$project/"
        python - "/work/$project" <<'PY'
import json, os, sys
root = sys.argv[1]
with open(os.path.join(root, 'source_layout.json')) as stream:
    layout = json.load(stream)
for name, target in layout['symlinks'].items():
    path = os.path.join(root, name)
    if os.path.lexists(path):
        os.unlink(path)
    os.symlink(target, path)
for name in layout['executable']:
    path = os.path.join(root, name)
    os.chmod(path, os.stat(path).st_mode | 0o111)
PY
        touch "/work/$project/.submission-source-ready"
    fi
done

# 作者 CAMR 权重由 prepare_assets.py 另行准备，避免把模型当作实验源码提交。
if [ ! -s /work/model/amr-anno-1.0/amr-anno-1.0.train.basic-abt-brown-verb.m ]; then
    mkdir -p /work/model
    tar -xzf /source/camr/amr-anno-1.0.train.m.tar.gz -C /work/model
fi

# Stanford 2015 / BLLIP / WordNet 的版本与原实验一致；缓存存在时先校验摘要。
stanford_dir=/work/camr/stanfordnlp
stanford_zip="$stanford_dir/stanford-corenlp-full-2015-04-20.zip"
if [ ! -s "$stanford_dir/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar" ]; then
    if [ -s /source/reproduction/stanford-corenlp-full-2015-04-20.zip ]; then
        printf '%s  %s\n' \
            977d57bde427862dd4d980c2c671cdc51c27256eb7536c293a57c128c789437d \
            /source/reproduction/stanford-corenlp-full-2015-04-20.zip | sha256sum -c -
        cp /source/reproduction/stanford-corenlp-full-2015-04-20.zip "$stanford_zip"
    else
        wget --continue --progress=dot:giga \
            https://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip \
            -O "$stanford_zip"
    fi
    unzip -tq "$stanford_zip" >/dev/null
    unzip -q "$stanford_zip" -d "$stanford_dir"
fi

# CAMR itself still calls ModelFetcher('WSJ+Gigaword'). Preinstall exactly that
# model so Python 2's obsolete Dropbox downloader is not needed at runtime.
bllip_dir=/work/camr/bllip-parser/models/WSJ+Gigaword
if [ -s /source/reproduction/BLLIP-WSJ-Gigaword2000.tar.bz2 ] && \
   [ ! -s "$bllip_dir/parser/rm.g" ]; then
    printf '%s  %s\n' \
        78fbba6affe5c24ee9d6d657cd334ae6bab16a6db730b50c2dda067c98144c21 \
        /source/reproduction/BLLIP-WSJ-Gigaword2000.tar.bz2 | sha256sum -c -
    mkdir -p "$bllip_dir"
    tar -xjf /source/reproduction/BLLIP-WSJ-Gigaword2000.tar.bz2 -C "$bllip_dir"
    test -s "$bllip_dir/parser/rm.g"
fi

# The source JAMR aligner uses WordNet 3.0, not the Illinois NER/cdec tools
# that its general-purpose setup script also installs for other tasks.
if [ ! -s /work/jamr/tools/WordNet-3.0/dict/data.noun ]; then
    mkdir -p /work/jamr/tools
    if [ -s /source/reproduction/WordNet-3.0.tar.gz ]; then
        printf '%s  %s\n' \
            640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52 \
            /source/reproduction/WordNet-3.0.tar.gz | sha256sum -c -
        cp /source/reproduction/WordNet-3.0.tar.gz /work/jamr/tools/WordNet-3.0.tar.gz
    else
        wget --continue --progress=dot:giga \
            https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz \
            -O /work/jamr/tools/WordNet-3.0.tar.gz
    fi
    tar -xzf /work/jamr/tools/WordNet-3.0.tar.gz -C /work/jamr/tools
fi

if [ ! -s /work/jamr/target/scala-2.10/jamr-assembly-0.1-SNAPSHOT.jar ]; then
    if [ ! -e /work/jamr/project/plugins.sbt.author ]; then
        cp /work/jamr/project/plugins.sbt /work/jamr/project/plugins.sbt.author
        cp /source/reproduction/jamr-plugins.sbt /work/jamr/project/plugins.sbt
    fi
    (cd /work/jamr && \
        JAVA_TOOL_OPTIONS="-Dsbt.repository.config=/source/reproduction/sbt.repositories -Dsbt.override.build.repos=true" \
        ./compile)
fi

if [ "$mode" = prepare ]; then
    python -c 'import numpy, nltk, bllipparser; print("Python 2 dependencies OK")'
    java -version
    echo "Author CAMR/JAMR dependencies prepared."
    exit 0
fi

# smoke 只跑首行；full --limit 100 复跑报告的前 100 行，可能产生多于 100 个图。
input=/work/input/sentences.txt
expected_lines=15633
mkdir -p /work/input
if [ "$mode" = smoke ]; then
    input=/work/input/smoke.txt
    expected_lines=1
    if [ ! -s "$input" ]; then
        head -n 1 /source/camr/camr/sentences.txt > "$input"
    fi
else
    if [ -n "$limit" ]; then
        input="/work/input/sentences.first${limit}.txt"
        expected_lines="$limit"
        test "$(wc -l < /source/camr/camr/sentences.txt)" -ge "$limit"
        if [ ! -s "$input" ]; then
            head -n "$limit" /source/camr/camr/sentences.txt > "$input"
        fi
        if ! head -n "$limit" /source/camr/camr/sentences.txt | cmp -s - "$input"; then
            echo "Cached input differs from the source's first $limit lines: $input" >&2
            exit 1
        fi
    elif [ ! -s "$input" ]; then
        cp /source/camr/camr/sentences.txt "$input"
    fi
    test "$(wc -l < "$input")" -eq "$expected_lines"
fi

model=/work/model/amr-anno-1.0/amr-anno-1.0.train.basic-abt-brown-verb.m
parsed="$input.all.basic-abt-brown-verb.parsed"
stamp="$(date -u +%Y%m%dT%H%M%SZ).$$"
if [ -n "$limit" ]; then
    run_name="first${limit}"
else
    run_name="$mode"
fi
log_dir=/results/logs
mkdir -p "$log_dir"

count_lines() {
    if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi
}

count_pattern() {
    if [ -f "$2" ]; then grep -c -- "$1" "$2" || true; else echo 0; fi
}

show_progress() {
    local stage="$1" elapsed="$2" seen total parsed_seen
    case "$stage" in
        1)
            seen="$(count_pattern '----------------------------------------' "$input.prp")"
            total="$(count_lines "$input.tok")"
            parsed_seen="$(count_lines "$input.tok.charniak.parse")"
            echo "[1/4] Stanford $seen/$expected_lines input lines; Charniak $parsed_seen/$total tokenized sentences; ${elapsed}s"
            ;;
        2)
            echo "[2/4] CAMR parsing with author's model; ${elapsed}s (graph file is written when parsing ends)"
            ;;
        3)
            total="$(count_pattern '^# ::id ' "$parsed")"
            seen="$(count_pattern '----------------------------------------' "$parsed.sent.prp")"
            parsed_seen="$(count_lines "$parsed.sent.tok.charniak.parse")"
            echo "[3/4] AMR format: Stanford $seen/$total graphs; Charniak $parsed_seen sentences; ${elapsed}s"
            ;;
        4)
            total="$(count_pattern '^# ::id ' "$parsed")"
            seen="$(count_pattern '^# ::alignments ' "$output")"
            echo "[4/4] JAMR aligned $seen/$total graphs; ${elapsed}s"
            ;;
    esac
}

run_logged() {
    local stage="$1" label="$2" log_file pid elapsed status
    shift 2
    log_file="$log_dir/${run_name}_stage${stage}.log"
    echo "[$stage/4] $label; detailed log: $log_file"
    "$@" > "$log_file" 2>&1 &
    pid=$!
    elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        show_progress "$stage" "$elapsed"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    if wait "$pid"; then
        echo "[$stage/4] Completed in ${elapsed}s"
    else
        status=$?
        echo "[$stage/4] Failed (exit $status). Last 40 log lines:" >&2
        tail -n 40 "$log_file" >&2
        return "$status"
    fi
}

backup_unfinished() {
    local artifact
    for artifact in "$@"; do
        if [ -e "$artifact" ]; then
            mv -- "$artifact" "$artifact.incomplete.$stamp"
            echo "Preserved incomplete cache: $artifact.incomplete.$stamp"
        fi
    done
}

cd /work/camr
if [ "$mode" = full ] && [ ! -e "$input.preprocess.ok" ]; then
    backup_unfinished "$input.prp" "$input.tok" \
        "$input.tok.charniak.parse" "$input.tok.charniak.parse.dep"
fi
# 第一步：原始句子 → 分词、词性、命名实体和句法依赖。
run_logged 1 'CAMR preprocess (Stanford CoreNLP + Charniak)' \
    python -u amr_parsing.py -m preprocess "$input"
if [ "$mode" = full ]; then
    prp_count="$(grep -o -- '----------------------------------------' "$input.prp" | wc -l)"
    test "$prp_count" -eq "$expected_lines"
    test -s "$input.tok.charniak.parse.dep"
    touch "$input.preprocess.ok"
fi
echo "[1/4] Prepared $(count_pattern '----------------------------------------' "$input.prp")/$expected_lines input lines and $(count_lines "$input.tok") tokenized sentences"
if [ "$mode" = full ] && [ ! -e "$parsed.parse.ok" ]; then
    backup_unfinished "$parsed"
fi
# 第二步：使用作者模型将预处理句子解析为 AMR 图。
run_logged 2 "CAMR parse: $input" \
    python -u amr_parsing.py -m parse --model "$model" "$input"
test -s "$parsed"
echo "[2/4] CAMR produced $(count_pattern '^# ::id ' "$parsed") graphs"
if [ "$mode" = full ]; then touch "$parsed.parse.ok"; fi
if [ "$mode" = full ] && [ ! -e "$parsed.amrfmt.ok" ]; then
    backup_unfinished "$parsed.sent" "$parsed.sent.prp" "$parsed.sent.tok" \
        "$parsed.sent.tok.charniak.parse" \
        "$parsed.sent.tok.charniak.parse.dep" "$parsed.amr.tok"
fi
# 第三步：把 CAMR 输出转成 JAMR Aligner 所需的分词 AMR 格式。
run_logged 3 "CAMR tokenize parsed AMR: $parsed" \
    python -u amr_parsing.py -m preprocess --amrfmt amr "$parsed"
test -s "$parsed.amr.tok"
echo "[3/4] Prepared $(count_pattern '^# ::id ' "$parsed.amr.tok") AMR records for JAMR"
if [ "$mode" = full ]; then touch "$parsed.amrfmt.ok"; fi
if [ "$mode" = smoke ]; then
    output=/results/smoke_output_file
elif [ -n "$limit" ]; then
    output="/results/first${limit}_output_file"
else
    output=/results/full_output_file
fi
if [ "$mode" = full ] && [ -e "$output.complete" ] && [ -s "$output" ]; then
    echo "Output already complete: $output"
    exit 0
fi
if [ "$mode" = full ]; then backup_unfinished "$output"; fi
run_aligner() {
    /work/jamr/run Aligner -v 0 --print-nodes-and-edges \
        < "$parsed.amr.tok" > "$output"
}
# 第四步：对齐原句词语与图节点，同时输出节点表、边表、对齐区间。
run_logged 4 "JAMR Aligner: $parsed.amr.tok" run_aligner
test -s "$output"
# 图数与对齐数必须相等，才能把文件标记为完成，不能仅看文件是否存在。
parsed_count="$(grep -c '^# ::id ' "$parsed")"
output_count="$(grep -c '^# ::id ' "$output")"
alignment_count="$(grep -c '^# ::alignments ' "$output")"
echo "CAMR graphs: $parsed_count; JAMR graphs: $output_count; alignments: $alignment_count"
test "$parsed_count" -gt 0
test "$parsed_count" -eq "$output_count"
test "$output_count" -eq "$alignment_count"
if [ "$mode" = full ]; then touch "$output.complete"; fi
echo "Complete: $output"
