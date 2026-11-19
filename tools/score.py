"""Score a candidate results.jsonl against a reference results.jsonl with the brief's metrics.

    python tools/score.py --ref ref.jsonl --cand results.jsonl [--videos list.txt] [--per-video]

Metrics (section 5 of the brief):
  * objects F1 over (video, second, object) triples
  * people count exactly right, % of reference seconds
  * people count within 1, % of reference seconds
A reference second that the candidate does not have counts as wrong (all its objects are false negatives).
Videos with an error line in the reference are excluded from the metrics and listed separately.
Accepts .jsonl or .jsonl.gz files.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict

OBJECTS = ("person", "car", "dog", "laptop", "cell phone")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    secs, errors = {}, {}
    with op(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d["video"]
            if "error" in d:
                errors[v] = d["error"]
                continue
            key = (v, int(d["second"]))
            if key in secs:
                raise ValueError(f"{path}:{ln}: duplicate line for {key}")
            bad = set(d["objects"]) - set(OBJECTS)
            if bad:
                raise ValueError(f"{path}:{ln}: unknown object names {bad}")
            secs[key] = (frozenset(d["objects"]), int(d["people"]))
    return secs, errors


def score(ref, cand, videos=None):
    rs, rerr = ref
    cs, cerr = cand
    keys = [k for k in rs if (videos is None or k[0] in videos) and k[0] not in rerr]
    tp = fp = fn = exact = within1 = missing = 0
    per_obj = defaultdict(lambda: [0, 0, 0])
    per_video = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # tp fp fn exact within1 n
    for k in keys:
        robj, rpeople = rs[k]
        if k in cs:
            cobj, cpeople = cs[k]
        else:
            cobj, cpeople = frozenset(), None
            missing += 1
        t, p, n = len(robj & cobj), len(cobj - robj), len(robj - cobj)
        tp, fp, fn = tp + t, fp + p, fn + n
        for o in OBJECTS:
            a, b = o in robj, o in cobj
            per_obj[o][0] += a and b
            per_obj[o][1] += b and not a
            per_obj[o][2] += a and not b
        e = cpeople is not None and cpeople == rpeople
        w = cpeople is not None and abs(cpeople - rpeople) <= 1
        exact += e
        within1 += w
        pv = per_video[k[0]]
        pv[0] += t; pv[1] += p; pv[2] += n; pv[3] += e; pv[4] += w; pv[5] += 1
    # candidate seconds that the reference does not have (extra lines) -> false positives
    extra = [k for k in cs if (videos is None or k[0] in videos) and k[0] not in rerr and k not in rs]
    for k in extra:
        fp += len(cs[k][0])
    n = len(keys)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {
        "seconds": n,
        "videos": len({k[0] for k in keys}),
        "f1": f1,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "exact": exact / n if n else 1.0,
        "within1": within1 / n if n else 1.0,
        "missing_seconds": missing,
        "extra_seconds": len(extra),
        "ref_error_videos": sorted(v for v in rerr if videos is None or v in videos),
        "cand_error_videos": sorted(v for v in cerr if videos is None or v in videos),
        "per_object": {o: {"tp": a, "fp": b, "fn": c,
                           "f1": (2 * a / (2 * a + b + c)) if (2 * a + b + c) else 1.0}
                       for o, (a, b, c) in per_obj.items()},
        "per_video": {v: {"f1": (2 * a[0] / (2 * a[0] + a[1] + a[2])) if (2 * a[0] + a[1] + a[2]) else 1.0,
                          "exact": a[3] / a[5], "within1": a[4] / a[5], "seconds": a[5]}
                      for v, a in per_video.items()},
    }


def passes(s):
    return s["f1"] >= 0.95 and s["exact"] >= 0.85 and s["within1"] >= 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--videos", help="optional file with one video id per line to restrict scoring")
    ap.add_argument("--per-video", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()
    videos = None
    if a.videos:
        videos = {l.strip() for l in open(a.videos, encoding="utf-8") if l.strip()}
    s = score(load(a.ref), load(a.cand), videos)
    if a.json:
        print(json.dumps(s, indent=1))
        return
    print(f"videos={s['videos']} seconds={s['seconds']} missing={s['missing_seconds']} extra={s['extra_seconds']}")
    print(f"objects F1={s['f1']:.4f} (P={s['precision']:.4f} R={s['recall']:.4f})  "
          f"people exact={100 * s['exact']:.2f}%  within1={100 * s['within1']:.2f}%  "
          f"-> {'PASS' if passes(s) else 'FAIL'}")
    for o, d in s["per_object"].items():
        print(f"  {o:<10} F1={d['f1']:.4f} tp={d['tp']} fp={d['fp']} fn={d['fn']}")
    if s["ref_error_videos"] or s["cand_error_videos"]:
        print("ref errors:", s["ref_error_videos"], " cand errors:", s["cand_error_videos"])
    if a.per_video:
        for v, d in sorted(s["per_video"].items(), key=lambda x: x[1]["f1"]):
            print(f"  {v:<40} n={d['seconds']:<5} F1={d['f1']:.3f} exact={d['exact']:.3f} w1={d['within1']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
