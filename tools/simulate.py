"""Simulate frame-sampling strategies offline on every-frame reference dumps, and score them.

    python tools/simulate.py --dumps val/ref_frames_tune [--ks 1,2,3,5] [--adaptive]

The dumps (written by ``run.py -k 0 --dump-dets DIR``) hold every frame's detections, so the result of
sampling any subset of frames is known exactly without running the model again. The reference result
for each second is the grader's rule over all frames; each strategy is scored with tools/score.py.
"""
import argparse
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vindex.common import CONF, OBJECTS, Sampling, aggregate, pick_indices  # noqa: E402
from tools.score import score  # noqa: E402


def load_dumps(d):
    vids = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json.gz"))):
        with gzip.open(f, "rt") as g:
            x = json.load(g)
        if x.get("error") and not x["frames"]:
            continue
        per = {}
        for pts, s, dets in x["frames"]:
            per.setdefault(s, []).append(dets)
        vids[x["video"]] = (per, x["n_seconds"])
    return vids


def frame_result(dets, conf=CONF):
    present = tuple(any(c == i and p >= conf for c, p in dets) for i in range(len(OBJECTS)))
    people = sum(1 for c, p in dets if c == 0 and p >= conf)
    return present, people


def to_lines(vid, results, n_seconds):
    out = {}
    for s in range(n_seconds):
        objs, people = results.get(s, ([], 0))
        out[(vid, s)] = (frozenset(objs), people)
    return out


def reference(vids):
    ref = {}
    for vid, (per, n) in vids.items():
        res = {s: aggregate([frame_result(d) for d in fl]) for s, fl in per.items()}
        ref.update(to_lines(vid, res, n))
    return ref


def uniform(vids, k):
    cand, used = {}, 0
    for vid, (per, n) in vids.items():
        res = {}
        for s, fl in per.items():
            idx = pick_indices(len(fl), k)
            used += len(idx)
            res[s] = aggregate([frame_result(fl[i]) for i in idx])
        cand.update(to_lines(vid, res, n))
    return cand, used


def adaptive(vids, k1, k2, band=(0.25, 0.6), neighbours=True):
    """Exactly the rule run.py applies (vindex.common.Sampling): stage 1 = k1 frames per second; a second is
    unsure if a stage-1 frame has a detection with conf in `band`, or (neighbours) its stage-1 result differs
    from the previous/next second's; unsure seconds are aggregated over the union of the k2 and k1 frames."""
    sp = Sampling(k1, k2, band, neighbours)
    cand, used = {}, 0
    for vid, (per, n) in vids.items():
        first, unsure, idx1 = {}, {}, {}
        for s, fl in per.items():
            i1, _ = sp.stage_indices(len(fl))
            idx1[s] = i1
            used += len(i1)
            first[s] = aggregate([frame_result(fl[i]) for i in i1])
            unsure[s] = any(band[0] <= p < band[1] for i in i1 for c, p in fl[i])
        res = {}
        for s, fl in per.items():
            if sp.needs_more(first.get(s - 1), first[s], first.get(s + 1), unsure[s]):
                i1, i2 = sp.stage_indices(len(fl))
                used += len(i2)
                res[s] = aggregate([frame_result(fl[i]) for i in sorted(set(i1) | set(i2))])
            else:
                res[s] = first[s]
        cand.update(to_lines(vid, res, n))
    return cand, used


def report(name, ref, cand, used, total_frames, total_seconds):
    s = score((ref, {}), (cand, {}))
    print(f"{name:<34} frames/s={used / total_seconds:5.2f} ({100 * used / total_frames:5.1f}% of frames)  "
          f"F1={s['f1']:.4f}  exact={100 * s['exact']:.2f}%  within1={100 * s['within1']:.2f}%  "
          + " ".join(f"{o[:6]}={d['f1']:.3f}" for o, d in s["per_object"].items()), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--ks", default="1,2,3,4,5,7")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--write", nargs=2, metavar=("STRATEGY", "OUT"),
                    help="write one strategy's lines, STRATEGY = 'uniform:K' or 'adaptive:K1:K2:LO:HI:NB' "
                         "with NB 0 (off), 1 (objects and people) or o (objects only)")
    a = ap.parse_args()
    vids = load_dumps(a.dumps)
    if a.write:
        f = a.write[0].split(":")
        if f[0] == "uniform":
            cand, _ = uniform(vids, int(f[1]))
        else:
            nb = {"0": False, "1": True, "o": "objects"}[f[5]]
            cand, _ = adaptive(vids, int(f[1]), int(f[2]), (float(f[3]), float(f[4])), nb)
        with open(a.write[1], "w", encoding="utf-8") as out:
            for (v, s), (objs, people) in sorted(cand.items()):
                out.write(json.dumps({"video": v, "second": s, "objects": [o for o in OBJECTS if o in objs],
                                      "people": people}) + "\n")
        return
    ref = reference(vids)
    total_frames = sum(len(fl) for per, _ in vids.values() for fl in per.values())
    total_seconds = sum(n for _, n in vids.values())
    print(f"{len(vids)} videos, {total_seconds} seconds, {total_frames} frames "
          f"({total_frames / max(total_seconds, 1):.1f} frames/s)")
    for k in [int(x) for x in a.ks.split(",")]:
        cand, used = uniform(vids, k)
        report(f"uniform k={k}", ref, cand, used, total_frames, total_seconds)
    if a.adaptive:
        for k1, k2 in ((1, 3), (1, 5), (3, 5), (3, 7)):
            for band in ((0.3, 0.5), (0.3, 0.55), (0.25, 0.6), (0.2, 0.7)):
                for nb in (False, "objects", True):
                    cand, used = adaptive(vids, k1, k2, band, nb)
                    report(f"adaptive {k1}->{k2} band={band} nb={str(nb)[0]}", ref, cand, used, total_frames,
                           total_seconds)


if __name__ == "__main__":
    main()
