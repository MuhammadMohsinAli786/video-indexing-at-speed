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
python run.py --input <folder of .parquet files> --output results.jsonl     # python = Python 3.11
```

No options are needed; progress goes to stderr (one line per minute) and the output file is written as videos
finish. Options (defaults are chosen automatically and are what was validated):

| option | default | meaning |
|---|---|---|
| `--k1` | `1` | stage-1 frames analysed per second of video (`0` = every frame) |
| `--k2` | `3` | frames per second used for seconds flagged as unsure (`k2 = k1`: plain uniform sampling) |
| `--band LO HI` | `0.25 0.6` | a stage-1 detection with confidence in [LO, HI) flags its second as unsure |
| `--neighbours` | `objects` | flag a second whose stage-1 objects (`objects`) or objects and people (`all`) differ from the previous/next second's (`off` to disable) |
| `--workers` | CPU threads − 1 | decode processes |
| `--batch` | auto from GPU memory: 8 (< 6 GB), 16, 32, 64 (≥ 20 GB), halved if a 640×640 test batch does not fit | GPU batch size |
| `--done-file` | none | append each finished video id (for resuming an interrupted run, see `tools/resume_list.py`, `tools/merge_parts.py`) |
| `--fp16 / --fp32` | FP16 on Volta or newer, FP32 on older GPUs | inference precision |

## Output

`results.jsonl.gz` in this repo holds the results for the first 50 FineVideo files
(`train-00000…00049-of-01357.parquet`): 1650 videos, 461,757 per-second lines, 0 error lines, produced with the
default settings in 9 h 38 min on a Quadro P2000 laptop GPU (`HARDWARE.md`). The 23 held-out validation videos
inside it score F1 0.9905 / people exact 96.15% / within-1 99.86% against the every-frame reference.
Format, one line per second of each video:

```json
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```

Second *s* holds the frames with timestamp *s* ≤ t < *s*+1, measured from the first decoded frame
(pts × time_base, exact fractions, so variable frame rate is handled). Every second up to the one holding the
last frame gets a line; seconds without frames get `[]` / `0`. Object names are exactly `person`, `car`, `dog`,
`laptop`, `cell phone`, in that order. Line order across videos is the order in which videos finish.

Error lines: a video that cannot be processed gets one `{"video": ..., "error": ...}` line (e.g. empty `mp4`
value, not a video, no video stream, no decodable frames); a video that fails part way keeps the lines of the
seconds decoded so far plus one error line. A Parquet file that cannot be read at all gets one error line with
the file name as `video` (no row number, since its rows are unknown); a file without an `mp4` column gets one
error line per row.

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
   letterbox **only the sampled frames** (see *Sampling* below): the middle frame of each second, and for
   "unsure" seconds the frames at 1/6 and 5/6 of the second, which the worker holds until the main process
   decides from the stage-1 results.
3. **Infer**: frames are grouped by letterbox shape and run in batches through YOLO11l (Ultralytics
   AutoBackend, fused, FP16 on Volta+) on one or two inference threads (two on GPUs with ≥ 8 GB, each with its
   own CUDA stream and model copy) while the main thread prepares the next batch. A partial batch waits at most
   0.5 s. On CUDA out-of-memory a batch is split and retried (one image falls back to the CPU as a last resort).
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

## Sampling: why it is fast and still correct

The model is the cost (on the grader's machine as here), so the main lever is how many frames go through it.
`run.py` uses **adaptive two-stage sampling**:

* stage 1: the middle frame of every second;
* a second is *unsure* if a stage-1 detection of the five classes has confidence in [0.25, 0.6) (close to the
  0.4 threshold), or if its stage-1 objects differ from the previous or next second's;
* unsure seconds also get the frames at 1/6 and 5/6 of the second (3 evenly spaced frames in total), and are
  aggregated over them with the grader's rules (≥ half → present; median people, rounded down).

On average 1.8 frames per second of video are analysed (≈ 6.7% of all frames) instead of ~28.

## Validation

`tools/reference.py` (Ultralytics predict on every frame) and `run.py --k1 0` (every frame through our
pipeline; identical to the former) give the every-frame reference; `tools/score.py` computes the grader's three
metrics; `tools/simulate.py` replays sampling strategies offline on the reference's per-frame detections.

Settings were tuned on 35 videos from files 02–09 (1.5 h) and then **confirmed on 23 held-out videos from files
45–49 (1 h) that were not used for tuning**:

| set | objects F1 | people exact | people within 1 |
|---|---|---|---|
| tuning (real `run.py` output, FP32, 35 videos) | 0.9879 | 95.44% | 99.85% |
| held-out (real `run.py` output, FP32, 23 videos) | 0.9905 | 96.15% | 99.86% |
| held-out (real `run.py` output, FP16, 23 videos) | 0.9905 | 96.12% | 99.86% |
| pass thresholds | ≥ 0.95 | ≥ 85% | ≥ 95% |

Separate test on another recording, outside FineVideo (a 296 s mix of a crowded street-market walk and a dog park, built
with `tools/make_unseen_video.py`, not used for tuning): objects F1 0.9798, people within 1 97.98%, but people
exact only **66.7%**: in dense crowds (≥ 6 people in 70% of its seconds) an exact per-second median needs nearly
every frame (uniform 9 frames/s still gives only 81%). FineVideo has few such seconds (≈ 3%), which is why the
validation above holds; see `docs/measurements.md`.

Details, the full strategy grid, profiling and dropped approaches: `docs/measurements.md`.

Reproduce (needs the FineVideo files in `finevideo/data` and a GPU):

```bash
python run.py --input finevideo/data --videos val/heldout_videos.txt --k1 0 --output ref.jsonl     # every frame
python run.py --input finevideo/data --videos val/heldout_videos.txt --output cand.jsonl            # final settings
python tools/score.py --ref ref.jsonl --cand cand.jsonl                                             # the 3 metrics
python tools/simulate.py --dumps val/ref_frames_tune --ks 1,3,5 --adaptive                          # strategy grid (CPU only)
python tests/test_rules.py && python tests/test_oom_recovery.py                                     # unit tests (CPU only)
```

`--videos` (a list of `file.parquet:row` ids) and `--dump-dets` are development options, hidden from `--help`.

## What was tested where

* Everything above was developed and run on Windows 11 with a Quadro P2000 (4 GB, Pascal) — see `HARDWARE.md`.
  Pascal has slow FP16, so the 50-file results were produced in FP32.
* The grader's RTX 4090 (Ada) will use FP16 automatically (and batch 64 with two inference threads, chosen from
  the GPU's memory). FP16 accuracy was validated on the held-out set (same metrics as FP32, table above), but
  FP16 speed was not measured on a 4090.
* Linux: `pip install -r requirements.txt` and `python prepare.py` were run on a Google Colab VM (Ubuntu, Python
  3.13, Tesla T4): install OK and the FP16 self-test on the T4 matched Ultralytics. The full run itself was not
  done there. All pins also have cp311 manylinux x86_64 wheels.
* No TensorRT/ONNX engine is used: it could not be validated on the Pascal dev GPU, so the timed run uses
  PyTorch (Ultralytics AutoBackend) only.

## Repository

* `run.py` — entry point; `prepare.py` — one-time preparation
* `vindex/` — pipeline (`pipeline.py`), decode workers (`worker.py`), video helpers (`video.py`),
  detector (`detector.py`), per-second rules (`common.py`)
* `tools/` — reference, scoring, simulation, equivalence check, validation-set picker, dataset survey, profiling, resume/merge helpers, Colab notebook
* `tests/` — edge-case generator, offline guard, unit tests (`test_rules.py`, `test_oom_recovery.py`)
* `val/` — validation video lists, every-frame reference results, per-frame reference detections (`ref_frames_*`, to replay sampling strategies), validation outputs
* `docs/measurements.md` — measurement log; `HARDWARE.md`; `WRITEUP.md`; `ai_chats/`
