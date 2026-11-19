# Measurements log

Facts and numbers only. Each entry has a date (PKT). Used by the candidate to write `WRITEUP.md`.

## Dev machine

- CPU: Intel Core i7-8850H @ 2.60 GHz, 6 cores / 12 threads
- RAM: 16 GB
- GPU: NVIDIA Quadro P2000, 4 GB (Pascal, sm_61), driver 573.71 (CUDA 12.8)
- Disk: Samsung MZVLB512HAJQ NVMe SSD
- OS: Windows 11 Pro (10.0.26300), Python 3.11.9

## Log

- 2026-11-19: environment set up. PyPI throughput on the dev network was 4-250 KB/s (Fastly edge dependent); wheels fetched with parallel resumable curl pinned to the fastest edge IP. HF download ~6 MB/s per connection.
- 2026-11-19: data survey (demux only, files 00-01, 66 videos): all H.264, mostly 640x360 (also 360x640 portrait, 270x480, 202x360, 480x360), 15-30 fps, up to ~10 min, keyframe every ~2-5 s, B-frames present (Main profile). The model input is therefore 640x384 (rect letterbox) for most videos and decoding is cheap.
- 2026-11-19: all 50 files downloaded, sizes equal the HF listing, 24.82 GB total.

## Equivalence checks (our batched detector vs Ultralytics)

- 2026-11-19, `tools/check_equivalence.py`, 3 videos x 60 frames: our letterbox output equals Ultralytics `LetterBox(640, auto=True, stride=32)` pixel for pixel (0 mismatches); five-class presence and people count identical to `YOLO("yolo11l.pt").predict(frame, imgsz=640, conf=0.4, iou=0.7)` on 180/180 frames.
- 2026-11-19, `prepare.py` self-test: bus.jpg (4 people) and zidane.jpg (2 people) identical to Ultralytics predict.
- Ultralytics 8.4.215 note: `model.predict` sets `rect=True`, so the default PyTorch letterbox is the minimum rectangle (640x360 -> 640x384); the predict default IoU in this version is 0.5, so IoU 0.7 is passed explicitly as the brief requires.
- Rotation (display matrix) handling vs OpenCV `cv2.VideoCapture` autorotate on remuxed copies of a real video with rotation 0/90/-90/180: same output shape, mean absolute pixel difference 0.0.
- Colour conversion: PyAV `to_ndarray("bgr24")` and `cv2.VideoCapture` give bit-identical frames on 3 BT.709-tagged videos (max diff 0); a naive BT.601 I420->BGR conversion (cv2.cvtColor) differs by up to 22 levels (mean 1.5-6), so it was not used.

## Profiling (dev machine)

- GPU, `vindex.detector.Detector`, random 640x384 input, P2000 (other jobs idle): FP32 batch 1/8/16/32 = 19.1/21.8/22.1/22.5 img/s; FP16 = 18.3/23.2/24.3/24.4 img/s (Pascal has no fast FP16). Measured with `torch.cuda.synchronize()` around 64 images after 3 warm-up batches.
- CPU, one process (`tools/profile_decode.py`, 8 tuning videos, 9325 frames, while a GPU job was running): H.264 decode 1777 frames/s (0.56 ms/frame); demux-only pass 0.015 ms/frame; letterbox 0.69 ms/frame.
- BGR conversion: `frame.to_ndarray("bgr24")` 6.9-7.5 ms/frame on first conversion of each frame (a new SwsContext per frame), 0.43-0.54 ms with one shared `VideoReformatter` (identical pixels). Switched to the shared reformatter.

## Robustness tests (all pass)

- `tests/make_cases.py` edge suite (24 rows) + corrupt/truncated parquet, parquet without mp4 column, non-parquet file, empty folder: every row ends with lines or an error line; no crash.
- Injected worker crash (`os._exit(3)`) and hang (sleep) via `VINDEX_TEST_CRASH` / `VINDEX_TEST_HANG` with `VINDEX_HANG_SECONDS=20`: both videos get one error line, the worker is restarted, all other videos' lines identical to a clean run. First version lost the crashed video entirely (its "start" message died with the process); fixed by recording the current video in shared memory.
- Offline: `PYTHONPATH=tests/offline_guard` blocks socket connect/getaddrinfo in every process: 0 blocked attempts, output identical to the online run (also with ultralytics-platform and nvidia-ml-py installed).

## Approaches tried / dropped

- Reference with the Ultralytics predict API (`tools/reference.py`, one process): GPU busy only ~35-60% (CPU-bound: per-frame Results objects, single-process decode), ~10 frames/s. Replaced for the bulk reference by `run.py -k 0` (every frame, same detector shown identical to Ultralytics above) at ~17-22 frames/s.
- 2026-11-19, cuDNN autotuning (`torch.backends.cudnn.benchmark`), P2000 FP32, 640x384, batch sizes 3/5/7/9/11 (reference job sharing the GPU): with benchmark=True the first call for each new batch size took 1935 ms/image vs 116 ms/image on repeat; with benchmark=False 119 vs 114 ms/image. Steady-state batch 16: 8.9 vs 9.0 img/s. Partial batches create many shapes, so benchmark is off.
- TensorRT not used: dev GPU is Pascal (TensorRT 10 targets newer GPUs) and the TensorRT wheel index downloaded at ~32 KB/s here (2.2 GB wheel), so an engine path could not be validated; not shipped untested.
- Two inference threads on one shared model failed (Ultralytics Detect head caches anchors per input shape: "size of tensor a (5040) must match ... (6300)"); fixed with one model instance per thread. Output identical to the single-thread version on the edge suite.
- 2026-11-19, dataset survey of all 50 files (`tools/survey.py`, demux only): 1650 videos (33 per file), 0 unreadable, total 128.03 h, 12,981,615 frames (mean 28.16 fps). Codec: h264 for all. Rotation tag: none (0 for all 1650). Sizes: 640x360 1284, 360x640 141, 480x360 49, 270x480 31, 202x360 28, others < 10 each. fps: 29.97 575, 30 505, 25 265, 23.98 177, 24 71. Longest video 658 s.
- Every-frame cost estimate on the dev GPU: 12.98 M frames / ~22 frames/s = ~590,000 s (~164 h) for the 50 files.
- 2026-11-19, end-to-end reference cross-check: `tools/reference.py` (decode + Ultralytics `YOLO.predict` on every frame + per-second rule) vs `run.py --k1 0` (every frame through our pipeline) on 3 short tuning videos (`val/crosscheck_videos.txt`, 2108 frames, 72 seconds): F1 1.0000, people exact 100%, within-1 100% (identical lines). The bulk references below therefore use `run.py --k1 0`.
- 2026-11-19, interruption test: edge suite run killed after 75 s (9 of 24 videos in the done-file), resumed with `tools/resume_list.py` + `run.py --videos`, merged with `tools/merge_parts.py`: merged output identical to an uninterrupted run. First version of the resume check used "seconds 0..last contiguous" to call a video complete, which cannot detect a video torn in the middle of writing; replaced by `--done-file` (a video id is appended only after its lines are flushed and fsynced).
- 2026-11-19, Google Colab attempt (free tier, `tools/colab_full_run.ipynb` opened from GitHub): runtime was a Tesla T4 15 GB, driver 580.82.07, Intel Xeon @ 2.00 GHz with 2 vCPUs (1 core, 2 threads), 12 GB RAM, Python 3.13.16. `pip install -r requirements.txt` succeeded on this Linux VM (first real Linux install test) and `prepare.py`'s self-test passed in FP16 on the T4 (bus.jpg 4 people, zidane.jpg 2 people, identical to Ultralytics predict). The run stopped at the Hugging Face token prompt: entering the token into the Colab page by automation was not permitted in this session, and the user was away, so the full run stays on the local machine (one machine for the whole run).

## Sampling study (tuning set: 35 videos from files 02-09, 5395 s, 146,763 frames)

Every-frame reference = `run.py --k1 0` (identical to Ultralytics predict, see above), per-frame detections dumped.
Strategies replayed offline with `tools/simulate.py` (the pipeline's own adaptive output was checked to be identical
to the simulator's: 539 s, F1 1.0000, exact 100%). Metrics vs the every-frame reference:

| strategy | frames analysed per second of video | % of all frames | objects F1 | people exact | people within 1 |
|---|---|---|---|---|---|
| uniform k=1 (middle frame) | 1.00 | 3.7 | 0.9781 | 92.90% | 99.41% |
| uniform k=2 | 2.00 | 7.3 | 0.9673 | 90.88% | 99.26% |
| uniform k=3 | 3.00 | 11.0 | 0.9880 | 95.59% | 99.87% |
| uniform k=5 | 5.00 | 18.4 | 0.9918 | 96.77% | 99.91% |
| adaptive 1->3, band [0.3,0.5), neighbours off | 1.39 | 5.1 | 0.9829 | 94.81% | 99.78% |
| adaptive 1->3, band [0.3,0.5), neighbours objects | 1.73 | 6.4 | 0.9878 | 95.40% | 99.83% |
| **adaptive 1->3, band [0.25,0.6), neighbours objects (chosen)** | **1.82** | **6.7** | **0.9879** | **95.44%** | **99.85%** |
| adaptive 1->3, band [0.25,0.6), neighbours objects+people | 1.88 | 6.9 | 0.9879 | 95.55% | 99.87% |
| adaptive 1->5, band [0.3,0.55), neighbours objects | 2.51 | 9.2 | 0.9909 | 96.52% | 99.87% |

Full grid: 4 (k1,k2) x 4 bands x 3 neighbour modes, output kept in the run log. Observations (facts):
- k=2 is worse than k=1 (the ">= half" rule makes an object present when seen in 1 of 2 samples).
- F1 levels off near 0.99 even at k=5 (objects present in about half of a second's frames).
- Chosen setting: same accuracy as uniform k=3 with 39% fewer frames; the wider band was kept over [0.3,0.5) (5% fewer frames, same accuracy) as the more conservative choice for unseen data.
- Per object (22-video interim check, chosen setting): person 0.989, car 0.973, dog 1.000, laptop 0.965, cell phone 0.972; worst single video F1 0.962, worst single-video people exact 85.2%.
