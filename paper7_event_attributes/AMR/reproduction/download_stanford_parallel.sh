#!/usr/bin/env bash
set -euo pipefail

# Download the exact Stanford CoreNLP release required by the bundled CAMR.
# Intended for WSL/Linux when the upstream single connection is very slow.
output="${1:?Pass the destination .zip path}"
url=https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-full-2015-04-20.zip
size=360824440
connections=8
chunk=$(( (size + connections - 1) / connections ))
parts_dir="${output}.parts"
mkdir -p "$parts_dir"

if [ -e "$output" ]; then
    echo "Destination already exists: $output" >&2
    exit 2
fi

pids=()
for ((i=0; i<connections; i++)); do
    start=$((i * chunk))
    end=$((start + chunk - 1))
    if ((end >= size)); then end=$((size - 1)); fi
    expected=$((end - start + 1))
    part="$parts_dir/part-$i"
    (
        if [ ! -e "$part" ] || [ "$(stat -c %s "$part")" -ne "$expected" ]; then
            curl --fail --location --retry 5 --retry-all-errors \
                --connect-timeout 20 --speed-limit 1024 --speed-time 45 \
                --silent --show-error \
                --range "$start-$end" --output "$part.tmp" "$url"
            test "$(stat -c %s "$part.tmp")" -eq "$expected"
            mv "$part.tmp" "$part"
        fi
        echo "Part $i complete: $expected bytes"
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
test "$failed" -eq 0

assembled="${output}.assembling"
if [ -e "$assembled" ]; then
    echo "Partial assembly exists; inspect it before retrying: $assembled" >&2
    exit 2
fi
for ((i=0; i<connections; i++)); do
    start=$((i * chunk))
    dd if="$parts_dir/part-$i" of="$assembled" bs=1M \
        seek="$start" oflag=seek_bytes conv=notrunc status=none
done
test "$(stat -c %s "$assembled")" -eq "$size"
if command -v unzip >/dev/null 2>&1; then
    unzip -tq "$assembled" >/dev/null
else
    python3 -m zipfile -t "$assembled" >/dev/null
fi
mv "$assembled" "$output"
sha256sum "$output"
echo "Verified Stanford archive: $output"
