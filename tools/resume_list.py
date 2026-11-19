"""List videos of an input folder that a (partial) results file does not cover yet, for resuming a long run.

    python tools/resume_list.py --input finevideo/data --done part1.jsonl [part2.jsonl ...] --out todo.txt
    python run.py --input finevideo/data --output part2.jsonl --videos todo.txt

run.py writes all lines of a video at once when the video is finished, so a video is complete if its lines run
from second 0 to its last second without gaps, or it has an error line. A video cut short by an interrupted
write is listed again (its partial lines must then be dropped when merging: see --clean).
"""
import argparse
import json
import os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--done", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    secs, errs = {}, set()
    for p in a.done:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line
                if "error" in d:
                    errs.add(d["video"])
                else:
                    secs.setdefault(d["video"], set()).add(d["second"])
    complete = set(errs) | {v for v, s in secs.items() if s == set(range(max(s) + 1))}
    done_lists = [p + ".done" for p in a.done if os.path.exists(p + ".done")]
    if done_lists:  # written by run.py --done-file: exact, survives a torn last video
        complete = set()
        for p in done_lists:
            complete |= {l.strip() for l in open(p, encoding="utf-8") if l.strip()}
    todo = []
    for f in sorted(os.listdir(a.input)):
        if not f.endswith(".parquet"):
            continue
        try:
            n = pq.ParquetFile(os.path.join(a.input, f)).metadata.num_rows
        except Exception:
            if f not in complete:
                todo.append(f + ":0")
            continue
        todo += [f"{f}:{r}" for r in range(n) if f"{f}:{r}" not in complete]
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(todo) + ("\n" if todo else ""))
    print(f"{len(complete)} videos complete, {len(todo)} to do -> {a.out}")


if __name__ == "__main__":
    main()
