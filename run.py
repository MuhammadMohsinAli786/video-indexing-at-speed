"""Video indexing at speed: per-second YOLO11l object presence and people counts for FineVideo parquet files.

    python run.py --input <folder of .parquet files> --output results.jsonl

See README.md. Runs fully offline: the weights must already be in weights/yolo11l.pt (python prepare.py).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("YOLO_OFFLINE", "1")          # Ultralytics: no network checks
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with .parquet files (mp4 column)")
    ap.add_argument("--output", required=True, help="output JSON Lines file")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "yolo11l.pt"))
    ap.add_argument("--k1", type=int, default=1, help="stage-1 frames analysed per second (0 = every frame)")
    ap.add_argument("--k2", type=int, default=3, help="frames per second for seconds flagged as unsure "
                                                       "(k2 == k1: plain uniform sampling)")
    ap.add_argument("--band", type=float, nargs=2, default=(0.25, 0.6), metavar=("LO", "HI"),
                    help="a stage-1 detection with LO <= conf < HI marks its second as unsure")
    ap.add_argument("--no-neighbours", action="store_true",
                    help="do not mark a second unsure when its stage-1 result differs from a neighbour's")
    ap.add_argument("--workers", type=int, default=None, help="decode processes (default: CPU threads - 1)")
    ap.add_argument("--batch", type=int, default=None, help="GPU batch size (default: auto)")
    ap.add_argument("--fp32", action="store_true", help="force FP32 inference")
    ap.add_argument("--fp16", action="store_true", help="force FP16 inference")
    ap.add_argument("--videos", help=argparse.SUPPRESS)  # dev: file with 'name.parquet:row' ids to process
    ap.add_argument("--dump-dets", help=argparse.SUPPRESS)  # dev: per-frame detections (conf>=0.25) dir
    ap.add_argument("--done-file", help="append each finished video id here (for resuming an interrupted run)")
    a = ap.parse_args()
    only = None
    if a.videos:
        only = {l.strip() for l in open(a.videos, encoding="utf-8") if l.strip() and not l.startswith("#")}

    from vindex.common import Sampling
    from vindex.pipeline import run

    sampling = Sampling(a.k1, a.k2, tuple(a.band), not a.no_neighbours)

    half = True if a.fp16 else (False if a.fp32 else None)
    t0 = time.time()
    res = run(a.input, a.output, a.weights, sampling=sampling, workers=a.workers, batch=a.batch,
              half=half, log=lambda s: print(s, file=sys.stderr, flush=True), only=only, dump_dir=a.dump_dets,
              done_file=a.done_file)
    print(f"wrote {res['lines']} lines for {res['videos']} videos in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
