"""Ground-truth reference: run YOLO11l on EVERY frame, the way the brief's grader does.

    python tools/reference.py --input finevideo/data --videos ids.txt --output ref.jsonl --frames-dir ref_frames

* decoding: PyAV, every frame, BGR, rotated per the display matrix (vindex.video)
* model: Ultralytics public API ``YOLO("yolo11l.pt").predict(frames, imgsz=640, iou=0.7, half=False)``,
  default letterbox; frames of one batch always share a shape (so letterboxing equals one-frame predict)
* per second: object present if found in >= half of the second's frames; people = median, rounded down
* --frames-dir stores every frame's detections (5 classes, conf >= 0.25) so sampling strategies can
  be simulated offline (tools/simulate.py) without running the model again.

Independent of run.py's own letterbox / inference / NMS code on purpose.
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq  # noqa: E402

from vindex.common import CLASS_IDS, CONF, IMGSZ, IOU, OBJECTS, aggregate  # noqa: E402
from vindex.video import FrameClock, VideoError, open_container, sec, to_bgr  # noqa: E402

STORE_CONF = 0.25  # detections stored per frame (>= 0.4 are the ones that count)


def read_rows(path, rows):
    """Yield (row, bytes) for the wanted rows of one parquet file (None = all rows)."""
    pf = pq.ParquetFile(path)
    i = 0
    for b in pf.iter_batches(batch_size=1, columns=["mp4"]):
        for v in b.column(0):
            if rows is None or i in rows:
                yield i, v.as_py()
            i += 1


def run_video(model, data, batch, device):
    """Return list of frames: (t_float, second, [[obj_index, conf], ...])."""
    c = open_container(data)
    vs = c.streams.video[0]
    vs.thread_type = "AUTO"
    clock = FrameClock(vs)
    frames, pending, meta = [], [], []

    def flush():
        if not pending:
            return
        res = model.predict(pending, imgsz=IMGSZ, conf=STORE_CONF, iou=IOU, device=device,
                            verbose=False, classes=None)
        for (t, s), r in zip(meta, res):
            cls = r.boxes.cls.int().tolist()
            conf = r.boxes.conf.tolist()
            dets = [[CLASS_IDS.index(k), round(p, 4)] for k, p in zip(cls, conf) if k in CLASS_IDS]
            frames.append((float(t), s, dets))
        pending.clear()
        meta.clear()

    err = None
    try:
        for fr in c.decode(vs):
            t = clock.time(fr)
            img = to_bgr(fr)
            if pending and img.shape != pending[0].shape:
                flush()
            pending.append(img)
            meta.append((t, sec(t)))
            if len(pending) >= batch:
                flush()
    except Exception as e:  # decoding failed part way
        err = f"decode failed: {type(e).__name__}: {e}"
    flush()
    c.close()
    return frames, err


def lines_from_frames(video, frames):
    per = {}
    for _, s, dets in frames:
        present = tuple(any(ci == i and p >= CONF for ci, p in dets) for i in range(len(OBJECTS)))
        people = sum(1 for ci, p in dets if ci == 0 and p >= CONF)
        per.setdefault(s, []).append((present, people))
    out = []
    if frames:
        for s in range(max(per) + 1):
            objs, people = aggregate(per.get(s, []))
            out.append({"video": video, "second": s, "objects": objs, "people": people})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with .parquet files")
    ap.add_argument("--videos", required=True, help="file with one 'name.parquet:row' per line")
    ap.add_argument("--output", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--weights", default="weights/yolo11l.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    os.makedirs(a.frames_dir, exist_ok=True)
    want = {}
    for line in open(a.videos, encoding="utf-8"):
        line = line.strip()
        if line:
            f, r = line.rsplit(":", 1)
            want.setdefault(f, set()).add(int(r))
    done = set()
    if os.path.exists(a.output):
        for line in open(a.output, encoding="utf-8"):
            done.add(json.loads(line)["video"])
    with open(a.output, "a", encoding="utf-8") as out:
        for fname in sorted(want):
            rows = want[fname] - {int(v.rsplit(":", 1)[1]) for v in done if v.startswith(fname + ":")}
            if not rows:
                continue
            for row, data in read_rows(os.path.join(a.input, fname), rows):
                vid = f"{fname}:{row}"
                t0 = time.time()
                try:
                    frames, err = run_video(model, data, a.batch, a.device)
                except VideoError as e:
                    frames, err = [], str(e)
                if not frames and err is None:
                    err = "could not decode video"
                lines = lines_from_frames(vid, frames)
                with gzip.open(os.path.join(a.frames_dir, vid.replace(":", "__") + ".json.gz"), "wt") as g:
                    json.dump({"video": vid, "error": err, "frames": frames}, g)
                for l in lines:
                    out.write(json.dumps(l) + "\n")
                if err:
                    out.write(json.dumps({"video": vid, "error": err}) + "\n")
                out.flush()
                dt = time.time() - t0
                print(f"{vid}: {len(frames)} frames, {len(lines)} s, {dt:.1f}s ({len(frames) / max(dt, 1e-9):.1f} fps)"
                      + (f" ERROR {err}" if err else ""), flush=True)


if __name__ == "__main__":
    main()
