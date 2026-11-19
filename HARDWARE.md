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
| Inference | FP32 (Pascal has slow FP16), batch 8 (chosen automatically for a 4 GB GPU), 11 decode processes |
| Settings | `run.py` defaults (adaptive sampling 1→3 frames per second), as submitted |

## Full run time (start to finish)

_To be filled in when the run finishes._ The run was interrupted once (the agent session that had started it
restarted at ~18:39 PKT and its child processes died) and resumed for the remaining videos with the done-file
mechanism (`tools/resume_list.py`, `tools/merge_parts.py`); the time below is the sum of the two parts' wall
times, not counting the gap between them. An earlier attempt at 18:19 was stopped after ~5 minutes because of a
GPU out-of-memory bug (fixed; see `docs/measurements.md`) and is not counted; its output was discarded.

| part | start (PKT) | end (PKT) | wall | videos |
|---|---|---|---|---|
| part 1 | 2026-11-19 18:32:39 | ~18:39 (interrupted) | _tbd_ | 9 |
| part 2 | 2026-11-19 18:41:55 | _tbd_ | _tbd_ | 1641 |
| **total** | | | _tbd_ | 1650 |

Note: this GPU is far slower than the grader's RTX 4090 (about 22 YOLO11l images/s at 640×384 in FP32), so the
run time here is dominated by GPU inference. See `docs/measurements.md` for the breakdown.
