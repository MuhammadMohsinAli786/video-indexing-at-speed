"""Survey the dataset without decoding: per video codec, size, fps, duration, frame count, rotation.

    python tools/survey.py --input finevideo/data --out val/survey.jsonl --procs 4
"""
import argparse
import collections
import glob
import io
import json
import os
from multiprocessing import Pool


def one_file(path):
    import av
    import pyarrow.parquet as pq

    av.logging.set_level(av.logging.PANIC)
    out = []
    name = os.path.basename(path)
    for i, b in enumerate(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["mp4"])):
        data = b.column(0)[0].as_py()
        rec = {"video": f"{name}:{i}", "bytes": len(data or b"")}
        try:
            c = av.open(io.BytesIO(data))
            vs = c.streams.video[0]
            pts = sorted(p.pts for p in c.demux(vs) if p.pts is not None and p.size)
            rot = 0
            try:
                for sd in getattr(vs, "side_data", {}) or {}:
                    pass
            except Exception:
                pass
            rec.update(codec=vs.codec_context.name, w=vs.width, h=vs.height,
                       fps=float(vs.average_rate or 0), frames=len(pts),
                       duration=float((pts[-1] - pts[0]) * vs.time_base) if pts else 0.0,
                       audio=len(c.streams.audio))
            c.close()
            # rotation is exposed on decoded frames: decode the first frame only
            c = av.open(io.BytesIO(data))
            for fr in c.decode(c.streams.video[0]):
                rot = fr.rotation
                break
            c.close()
            rec["rotation"] = rot
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))
    recs = []
    with Pool(a.procs) as pool:
        for r in pool.imap(one_file, files):
            recs += r
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in recs if "error" not in r]
    dur = sum(r["duration"] for r in ok)
    print(f"videos {len(recs)}, errors {len(recs) - len(ok)}, total duration {dur / 3600:.2f} h, "
          f"frames {sum(r['frames'] for r in ok)}, mean fps {sum(r['frames'] for r in ok) / max(dur, 1):.2f}")
    for key in ("codec", "rotation"):
        print(key, collections.Counter(r.get(key) for r in ok).most_common(8))
    print("size", collections.Counter(f"{r['w']}x{r['h']}" for r in ok).most_common(10))
    print("fps", collections.Counter(round(r["fps"], 2) for r in ok).most_common(8))
    print("longest (s)", sorted((round(r["duration"]) for r in ok), reverse=True)[:5])


if __name__ == "__main__":
    main()
