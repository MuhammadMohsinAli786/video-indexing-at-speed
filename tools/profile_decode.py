"""Profile the CPU side for one process: demux, decode, BGR conversion and letterbox, per frame.

    python tools/profile_decode.py --input finevideo/data --videos val/tune_videos.txt --n 6
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vindex.detector import letterbox_into
from vindex.video import open_container, packet_pts, to_bgr
from tools.reference import read_rows

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--videos", required=True); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
ids = [l.strip() for l in open(a.videos) if l.strip()][: a.n]
tot = {"frames": 0, "demux": 0.0, "decode": 0.0, "bgr": 0.0, "letterbox": 0.0, "read": 0.0}
for vid in ids:
    f, r = vid.rsplit(":", 1)
    t = time.perf_counter(); data = next(d for _, d in read_rows(os.path.join(a.input, f), {int(r)})); tot["read"] += time.perf_counter() - t
    t = time.perf_counter(); packet_pts(data); tot["demux"] += time.perf_counter() - t
    c = open_container(data); vs = c.streams.video[0]; vs.thread_type = "AUTO"; vs.thread_count = a.threads
    frames = []
    t = time.perf_counter()
    for fr in c.decode(vs):
        frames.append(fr) if len(frames) < 300 else None
        tot["frames"] += 1
    tot["decode"] += time.perf_counter() - t
    t = time.perf_counter(); imgs = [to_bgr(fr) for fr in frames]; tot["bgr"] += (time.perf_counter() - t) / len(frames)
    t = time.perf_counter(); [letterbox_into(im) for im in imgs]; tot["letterbox"] += (time.perf_counter() - t) / len(frames)
    c.close()
n = len(ids)
print(f"{n} videos, {tot['frames']} frames, threads/decoder={a.threads}")
print(f"decode: {tot['frames'] / tot['decode']:.0f} frames/s per process ({1e3 * tot['decode'] / tot['frames']:.2f} ms/frame)")
print(f"demux pass: {1e3 * tot['demux'] / tot['frames']:.3f} ms/frame; parquet read: {tot['read'] / n:.3f} s/video")
print(f"to BGR+rotate: {1e3 * tot['bgr'] / n:.2f} ms/frame; letterbox: {1e3 * tot['letterbox'] / n:.2f} ms/frame")
