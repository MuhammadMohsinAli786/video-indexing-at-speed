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
