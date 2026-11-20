# Hardware of the full 50-file run

The results in `results.jsonl.gz` (all 50 files, `train-00000…00049-of-01357.parquet`: 1650 videos, 128.03 h of
video, 12.98 M frames) were produced on one machine, a laptop:

| | |
|---|---|
| Machine | HP ZBook 15 G5 (laptop), local run |
| GPU | NVIDIA Quadro P2000 (mobile), 4 GB GDDR5, Pascal (compute capability 6.1), driver 573.71 (CUDA 12.8) |
| CPU | Intel Core i7-8850H @ 2.60 GHz, 6 cores / 12 threads |
| RAM | 16 GB DDR4-2667 |
| Disk | Samsung MZVLB512HAJQ NVMe SSD (input Parquet files and output on it) |
| OS | Windows 11 Pro (10.0.26300), Python 3.11.9, PyTorch 2.14.1+cu126 |
| Inference | FP32 (Pascal has slow FP16), batch 8 (chosen automatically for a 4 GB GPU) |
| Settings | `run.py` defaults (adaptive sampling 1→3 frames per second), as submitted |
| Decode processes | 11 (parts 1–2), 8 (part 3, to keep the laptop cooler) |

## Full run time (start to finish)

Frames analysed by the model over the three parts: 788,482 (6,302 + 261,163 + 521,017, including the frames of
videos that were redone), of 12.98 M frames in the 50 files (~6%). GPU busy 34,267 s of 34,697 s (99%).
Result: 1650 videos, 461,757 per-second lines, 0 error lines.

**Total: 34,697 s = 9 h 38 min** (sum of the three parts' run times below; gaps between parts not counted).

The run did not go through in one piece, so it was done in three parts with the resume mechanism
(`run.py --done-file`, `tools/resume_list.py`, `tools/merge_parts.py`). Each part only processed videos that
earlier parts had not finished; the merge keeps one complete result per video.

| part | start (PKT) | end (PKT) | wall | videos finished | why it ended |
|---|---|---|---|---|---|
| part 1 | 2026-11-19 18:32:39 | ~18:39 | 371 s (last progress line) | 9 | the agent session that had started it restarted, which killed it |
| part 2 | 2026-11-19 18:41:55 | ~21:57 | ~11,711 s (last progress line) | 524 | the laptop shut down (probably heat; GPU at 84 °C) |
| part 3 | 2026-11-20 12:09:00 | 2026-11-20 18:26:11 | 22,615 s (run.py; 22,631 s launcher wall) | 1129 | finished (rc 0) |
| **total** | | | **34,697 s (9 h 38 min)** | **1650** | |

Notes:
- Part 3 also re-ran 12 videos that part 2 had finished with the error "timed out waiting for sampling decisions"
  (a pipeline bug fixed between the parts, see `docs/measurements.md`), so part 3's time includes those 12 videos
  a second time. Videos that were in progress when parts 1 and 2 were killed were redone from the start in the
  next part.
- Not counted: an attempt started at 18:19 on 19 Nov and stopped after ~5 minutes because of a GPU out-of-memory
  bug (fixed); its output was discarded.
- This GPU is far slower than the grader's RTX 4090 (about 22–24 YOLO11l images/s at 640×384 in FP32), so the run
  time here is dominated by GPU inference (GPU busy ~98% of the wall time). See `docs/measurements.md`.
