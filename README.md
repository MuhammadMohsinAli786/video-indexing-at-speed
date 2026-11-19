# Video indexing at speed

For every second of every video in a folder of FineVideo Parquet files, `run.py` records which of
**person, car, dog, laptop, cell phone** appear (YOLO11l, conf ≥ 0.4) and how many people are visible,
and writes one JSON line per second.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver for CUDA 12)

```bash
sudo apt-get install -y libgl1 libglib2.0-0      # needed by opencv-python (pulled in by ultralytics)
python3.11 -m pip install -r requirements.txt
```

`requirements.txt` pins exact versions; PyTorch comes from the official CUDA 12.6 wheel index
(`--extra-index-url` inside the file), which runs on any CUDA 12.x driver.

## Preparation (once, untimed, needs internet)

```bash
python3.11 prepare.py
```

Downloads the official `yolo11l.pt` to `weights/` (SHA-256 checked) and runs a GPU self-test that checks our
batched inference against Ultralytics' own `predict` on two sample images. Nothing else is downloaded later.

## Run (timed, no internet needed)

```bash
python3.11 run.py --input <folder of .parquet files> --output results.jsonl
```

Options (defaults are chosen automatically and are what was validated):

| option | default | meaning |
|---|---|---|
| `-k / --samples-per-second` | `3` | frames analysed per second of video (`0` = every frame) |
| `--workers` | CPU threads − 1 | decode processes |
| `--batch` | auto (largest of 64/32/… that fits) | GPU batch size |
| `--fp16 / --fp32` | FP16 on Volta or newer, FP32 on older GPUs | inference precision |

## Output

`results.jsonl.gz` in this repo holds the results for the first 50 FineVideo files
(`train-00000…00049-of-01357.parquet`). Format, one line per second of each video:

```json
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```

Second *s* holds the frames with timestamp *s* ≤ t < *s*+1, measured from the first decoded frame
(pts × time_base, exact fractions, so variable frame rate is handled). Every second up to the one holding the
last frame gets a line; seconds without frames get `[]` / `0`.

## How it works

```
parquet reader thread ──► N decode processes ──► shared-memory slots ──► main: batch by shape ──► GPU thread
 (mp4 column, row by row)   PyAV/FFmpeg            (letterboxed uint8)     pinned buffer           YOLO11l + NMS
                                                                                               ──► per-second
                                                                                                   aggregation ──► results.jsonl
```

1. **Read**: a thread streams the `mp4` column of each Parquet file row by row (only that column is read).
2. **Decode** (one process per CPU thread): demux once to learn every frame's timestamp, then decode all frames
   (H.264 needs that) but convert to BGR, rotate (display matrix, like `ffmpeg`/OpenCV autorotate) and
   letterbox **only the k sampled frames per second**: k evenly spaced frames out of that second's frames
   (all of them if the second has ≤ k frames).
3. **Infer**: frames are grouped by letterbox shape and run in batches through YOLO11l (Ultralytics
   AutoBackend, fused, FP16 on Volta+) on a dedicated GPU thread while the main thread prepares the next batch.
   Letterbox (minimum rectangle, stride 32, pad 114, `INTER_LINEAR`), BGR→RGB, /255 and NMS (best class of 80,
   conf ≥ 0.4, class-aware IoU 0.7, max 300) reproduce Ultralytics' `predict` exactly; this was checked
   frame by frame (`tools/check_equivalence.py`, `prepare.py`).
4. **Aggregate** each second exactly like the grader, over the sampled frames: an object is present if found in at
   least half of the samples; people = median of the per-sample counts, rounded down at .5.

## Robustness

Every video ends with either its lines or an error line; the run never stops early.
Tested on purpose (`tests/make_cases.py`): truncated mp4 bytes, `None`/empty `mp4` value, random bytes,
audio-only file, < 1 s and single-frame videos, VFR with a 2.5 s gap, rotated 90/−90/180, 3840×2160 and
7680×4320 HEVC, VP9, AV1, MPEG-4, 10-bit HEVC, MKV, odd sizes, corrupt middle, corrupt / truncated parquet files,
a parquet without an `mp4` column, a non-parquet file and an empty input folder. A decode error part way keeps
the lines for the seconds decoded so far and adds one error line. A decode process that crashes or hangs is
restarted and its video gets an error line (tested with injected crash/hang). The timed run makes no network
access (tested with sockets blocked in every process: `tests/offline_guard`).

## Validation

`tools/reference.py` / `run.py -k 0` produce the every-frame reference; `tools/score.py` computes the
grader's three metrics; `tools/simulate.py` replays sampling strategies offline. Numbers are in
`docs/measurements.md`.

## What was tested where

* Everything above was developed and run on Windows 11 with a Quadro P2000 (4 GB, Pascal) — see `HARDWARE.md`.
  Pascal has slow FP16, so the 50-file results were produced in FP32.
* The grader's RTX 4090 (Ada) will use FP16 automatically. The FP16 code path was run and validated for
  accuracy on the P2000, but its speed was not measured on a 4090.
* The Linux install was checked for wheel availability (all pins exist for cp311 manylinux x86_64) but not
  run on Ubuntu.

## Repository

* `run.py` — entry point; `prepare.py` — one-time preparation
* `vindex/` — pipeline (`pipeline.py`), decode workers (`worker.py`), video helpers (`video.py`),
  detector (`detector.py`), per-second rules (`common.py`)
* `tools/` — reference, scoring, simulation, equivalence check, validation-set picker
* `tests/` — edge-case generator, offline guard
* `val/` — validation video lists and reference results
* `docs/measurements.md` — measurement log; `HARDWARE.md`; `WRITEUP.md`; `ai_chats/`
