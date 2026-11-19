"""Merge the output parts of an interrupted + resumed run into one results file, and check completeness.

    python tools/merge_parts.py --input finevideo/data --parts out/part1.jsonl out/part2.jsonl --out results.jsonl.gz

For every video keeps the lines of one part in which the video is complete (seconds 0..last without gaps, plus
an error line if any), preferring the latest part; partial copies (cut by an interruption) are dropped. Checks
that every row of every input parquet file has lines or an error line, and prints a summary.
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    chosen = {}  # vid -> list of lines (dicts)
    for p in a.parts:
        per = defaultdict(list)
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line
                per[d["video"]].append(d)
        done = None
        if os.path.exists(p + ".done"):  # written by run.py --done-file: exact
            done = {l.strip() for l in open(p + ".done", encoding="utf-8") if l.strip()}
        for vid, lines in per.items():
            if done is not None and vid not in done:
                continue
            secs = sorted(d["second"] for d in lines if "error" not in d)
            has_err = any("error" in d for d in lines)
            complete = (secs == list(range(len(secs))) and (secs or has_err)) and len(set(secs)) == len(secs)
            if complete:
                chosen[vid] = lines  # later parts win
    expected = []
    for f in sorted(os.listdir(a.input)):
        if not f.lower().endswith(".parquet"):
            continue
        try:
            n = pq.ParquetFile(os.path.join(a.input, f)).metadata.num_rows
            expected += [f"{f}:{r}" for r in range(n)]
        except Exception:
            expected.append(f)
    missing = [v for v in expected if v not in chosen]
    n_lines = n_err = n_sec = 0
    op = gzip.open if a.out.endswith(".gz") else open
    with op(a.out, "wt", encoding="utf-8", newline="\n") as out:
        for vid in expected:
            for d in chosen.get(vid, []):
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_lines += 1
                n_err += "error" in d
                n_sec += "error" not in d
    print(f"videos expected {len(expected)}, written {len(expected) - len(missing)}, missing {len(missing)}; "
          f"lines {n_lines} (seconds {n_sec}, error lines {n_err}) -> {a.out}")
    if missing:
        print("missing:", missing[:20], "..." if len(missing) > 20 else "")


if __name__ == "__main__":
    main()
