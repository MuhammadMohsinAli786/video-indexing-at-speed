"""Pick a reproducible, varied validation subset: python tools/pick_videos.py --input DIR --files GLOB --budget SECONDS --seed N"""
import argparse, glob, io, os, random, sys
import av, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--files", required=True)
ap.add_argument("--budget", type=float, default=5400); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-dur", type=float, default=420)
a = ap.parse_args()
cands = []
for fp in sorted(glob.glob(os.path.join(a.input, a.files))):
    for i, b in enumerate(pq.ParquetFile(fp).iter_batches(batch_size=1, columns=["mp4"])):
        try:
            c = av.open(io.BytesIO(b.column(0)[0].as_py())); vs = c.streams.video[0]
            dur = float(vs.duration * vs.time_base) if vs.duration else float(c.duration or 0) / 1e6
            cands.append((os.path.basename(fp), i, dur, vs.width, vs.height, float(vs.average_rate or 0)))
            c.close()
        except Exception as e:
            print("skip", fp, i, e, file=sys.stderr)
rnd = random.Random(a.seed); rnd.shuffle(cands)
# make sure unusual shapes are represented first
odd = [c for c in cands if c[3] < c[4] or c[3] < 400][:8]
cands = odd + [c for c in cands if c not in odd]
picked, tot = [], 0.0
for c in cands:
    if c[2] <= a.max_dur and tot + c[2] <= a.budget:
        picked.append(c); tot += c[2]
for c in picked:
    print(f"{c[0]}:{c[1]}")
print(f"# {len(picked)} videos, {tot:.0f} s; " + ", ".join(f"{c[3]}x{c[4]}@{c[5]:.2f}" for c in picked), file=sys.stderr)
