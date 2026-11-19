# Hardware of the full 50-file run

The results in `results.jsonl.gz` (all 50 files, `train-00000…00049-of-01357.parquet`) were produced on a laptop:

| | |
|---|---|
| Machine | HP ZBook 15 G5 (laptop), local run |
| GPU | NVIDIA Quadro P2000 (mobile), 4 GB GDDR5, Pascal (compute capability 6.1), driver 573.71 (CUDA 12.8) |
| CPU | Intel Core i7-8850H @ 2.60 GHz, 6 cores / 12 threads |
| RAM | 16 GB DDR4-2667 |
| Disk | Samsung MZVLB512HAJQ NVMe SSD (input Parquet files and output on it) |
| OS | Windows 11 Pro (10.0.26300), Python 3.11.9, PyTorch 2.14.1+cu126 |
| Inference | FP32 (Pascal has slow FP16), batch 16 |

## Full run time (start to finish)

_To be filled in when the run finishes._

Note: this GPU is far slower than the grader's RTX 4090 (about 22 YOLO11l images/s at 640×384 in FP32), so the
run time here is dominated by GPU inference. See `docs/measurements.md` for the breakdown.
