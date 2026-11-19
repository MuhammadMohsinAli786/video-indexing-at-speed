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

from vindex.common import CONF, OBJECTS, aggregate, pick_indices  # noqa: E402
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
    """Stage 1: k1 frames per second. Stage 2: k2 frames for 'uncertain' seconds, i.e. a stage-1 frame has a
    detection with confidence in `band`, or (neighbours) the stage-1 result differs from an adjacent second."""
    cand, used = {}, 0
    for vid, (per, n) in vids.items():
        first, unsure = {}, set()
        for s, fl in per.items():
            idx = pick_indices(len(fl), k1)
            used += len(idx)
            rs = [frame_result(fl[i]) for i in idx]
            first[s] = (aggregate(rs), idx)
            if any(band[0] <= p < band[1] for i in idx for c, p in fl[i]):
                unsure.add(s)
        if neighbours:
            for s in first:
                for t in (s - 1, s + 1):
                    if t in first and first[t][0] != first[s][0]:
                        unsure.add(s)
        res = {}
        for s, fl in per.items():
            if s in unsure:
                idx = sorted(set(pick_indices(len(fl), k2)) | set(first[s][1]))
                used += len(idx) - len(first[s][1])
                res[s] = aggregate([frame_result(fl[i]) for i in idx])
            else:
                res[s] = first[s][0]
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
    a = ap.parse_args()
    vids = load_dumps(a.dumps)
    ref = reference(vids)
    total_frames = sum(len(fl) for per, _ in vids.values() for fl in per.values())
    total_seconds = sum(n for _, n in vids.values())
    print(f"{len(vids)} videos, {total_seconds} seconds, {total_frames} frames "
          f"({total_frames / max(total_seconds, 1):.1f} frames/s)")
    for k in [int(x) for x in a.ks.split(",")]:
        cand, used = uniform(vids, k)
        report(f"uniform k={k}", ref, cand, used, total_frames, total_seconds)
    if a.adaptive:
        for k1, k2 in ((1, 3), (1, 5), (2, 5), (3, 7)):
            for band in ((0.3, 0.55), (0.25, 0.6), (0.2, 0.7)):
                for nb in (False, True):
                    cand, used = adaptive(vids, k1, k2, band, nb)
                    report(f"adaptive {k1}->{k2} band={band} nb={int(nb)}", ref, cand, used, total_frames, total_seconds)


if __name__ == "__main__":
    main()
