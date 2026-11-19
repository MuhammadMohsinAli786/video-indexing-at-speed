"""Check that vindex.detector (our letterbox + batched inference + NMS) matches Ultralytics predict per frame.

    python tools/check_equivalence.py --input finevideo/data --videos ids.txt --frames 200 [--half]

For each listed video, takes every n-th frame, runs Ultralytics ``YOLO.predict`` (one frame at a time,
imgsz=640, conf=0.4, iou=0.7) and our Detector (batched), and compares the five-class presence and the
people count. Also checks our letterbox output equals Ultralytics LetterBox pixel for pixel.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vindex.common import CLASS_IDS, CONF, IMGSZ, IOU  # noqa: E402
from vindex.detector import Detector, letterbox_into  # noqa: E402
from vindex.video import open_container, to_bgr  # noqa: E402
from tools.reference import read_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--frames", type=int, default=100, help="frames per video")
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    ap.add_argument("--half", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    yolo = YOLO(a.weights)
    det = Detector(a.weights, "cuda:0", half=a.half)
    lb = LetterBox(IMGSZ, auto=True, stride=32)
    tot = same = lb_bad = 0
    diffs = []
    for line in open(a.videos, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        fname, row = line.rsplit(":", 1)
        data = next(d for r, d in read_rows(os.path.join(a.input, fname), {int(row)}))
        c = open_container(data)
        vs = c.streams.video[0]
        n = vs.frames or 3000
        step = max(1, n // a.frames)
        imgs = []
        for i, fr in enumerate(c.decode(vs)):
            if i % step == 0:
                imgs.append(to_bgr(fr))
            if len(imgs) >= a.frames:
                break
        c.close()
        # letterbox equality
        for im in imgs[:5]:
            if not np.array_equal(lb(image=im), letterbox_into(im)):
                lb_bad += 1
        # ours, batched
        ours = []
        for i in range(0, len(imgs), 16):
            x = torch.from_numpy(np.stack([letterbox_into(im) for im in imgs[i:i + 16]]))
            pr, pp, _, _ = det.detect(x)
            ours += list(zip(pr.cpu().numpy().tolist(), pp.cpu().numpy().tolist()))
        # reference, one frame at a time
        for im, (opr, opp) in zip(imgs, ours):
            r = yolo.predict(im, imgsz=IMGSZ, conf=CONF, iou=IOU, half=a.half, verbose=False)[0]
            cls = r.boxes.cls.int().tolist()
            rpr = [cid in cls for cid in CLASS_IDS]
            rpp = cls.count(0)
            tot += 1
            if rpr == opr and rpp == opp:
                same += 1
            else:
                diffs.append((line, rpr, opr, rpp, opp, [round(float(x), 3) for x in r.boxes.conf.tolist()]))
        print(f"{line}: {len(imgs)} frames, cumulative agreement {same}/{tot}", flush=True)
    print(f"letterbox mismatches: {lb_bad}")
    print(f"frames identical (presence of 5 classes and people count): {same}/{tot} = {same / max(tot, 1):.4f}")
    for d in diffs[:20]:
        print("  diff", d)


if __name__ == "__main__":
    main()
