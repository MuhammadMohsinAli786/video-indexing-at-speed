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

## Held-out confirmation (23 videos from files 45-49, 3606 s, 102,347 frames; not used for tuning)

Every-frame reference (`run.py --k1 0`, FP32, P2000): 4255.6 s wall.
Real `run.py` output with the final default settings (adaptive 1->3, band [0.25,0.6), neighbours objects), scored with `tools/score.py`:

| run (2026-11-19, P2000) | frames analysed | wall | GPU busy | objects F1 (P / R) | people exact | within 1 |
|---|---|---|---|---|---|---|
| final, FP32 | 5982 (1.66/s of video, 5.8%) | 287.3 s | 251.7 s | 0.9905 (0.9907 / 0.9903) | 96.15% | 99.86% |
| final, FP16 | 5976 | 243.6 s | 222.7 s | 0.9905 (0.9910 / 0.9899) | 96.12% | 99.86% |

- Per object (FP32): person 0.9947 (2440 positive seconds), laptop 0.9799 (176), dog 1.0000 (8), cell phone 0.8736 (40), car 0.7879 (15). The low car/cell-phone F1 comes from few positive seconds.
- The real pipeline output equals the simulator's prediction for this setting (simulated: F1 0.9905, exact 96.15%, within-1 99.86%).
- Same videos, same GPU: every frame 4255.6 s vs final 287.3 s = 14.8x faster.
- Held-out simulation of other strategies (`val/sim_heldout_grid.txt`): uniform k=1 F1 0.9770 / exact 93.54% / within-1 99.25%; uniform k=3 0.9907 / 96.48% / 99.86%; uniform k=5 0.9929 / 96.59% / 99.92%.
- FP16 vs FP32 changed 6 of ~5980 sampled frames' stage-2 decisions and no metric beyond the 4th digit (on Pascal, where FP16 is not faster: 243.6 s vs 287.3 s wall here).

## Tuning set, real run.py output (final settings, FP32, P2000, 2026-11-19)

- 35 videos, 5395 s of video: 9799 frames analysed (1.82/s), 432.3 s wall, GPU busy 407.0 s. Score vs the every-frame reference: F1 0.9879 (P 0.9906 / R 0.9853), people exact 95.44%, within-1 99.85% (= the simulator's prediction).
- Note: the first tuning reference process died with CUDA out-of-memory after 22 videos (I was running test jobs on the same 4 GB GPU at the same time); its last buffered output lines were lost, while its per-frame dumps were complete. `val/ref_tune.jsonl` was therefore rebuilt from the dumps (`tools/simulate.py --write uniform:0`, 5395 lines); the lines that had been written agreed with the rebuilt file 100%. Since then the GPU threads catch out-of-memory and retry the batch in halves, and no other GPU job runs next to a long run.

## Full 50-file run (local, P2000)

- Started 2026-11-19 18:19:33 PKT by `D:\vi-tools\after_refs.sh` after the held-out gate passed; command:
  `python run.py --input finevideo/data --output part1.jsonl --done-file part1.jsonl.done --fp32` (defaults otherwise: adaptive 1->3, batch 16, 11 decode workers).
- 2026-11-19 18:19 first full-run attempt ABORTED after ~5 min: every finished video got "inference failed: AcceleratorError: CUDA error: out of memory" (19 error lines, 31 videos counted). Cause: without `--batch`, `auto_batch` probed a worst-case 640x640 batch of 16 and PyTorch kept that memory cached; on the 4 GB WDDM GPU (with other load on the PC: ~2.7 GB free RAM, CPU at 100%) this oversubscribed GPU memory (startup 126.6 s instead of ~6 s) and real batches then failed; the split-and-retry gave up at batch size 1. Fixes: batch by GPU size (8 below 6 GB, 16 / 32 / 64 above 6 / 10 / 20 GB), probe memory released after the probe, out-of-memory at batch 1 retried 3x and finally run on the CPU (never an error line for a transient shortage; counters `oom` and `cpu_fallback` in the log). Unit test `tests/test_oom_recovery.py`; held-out rerun with the fix: identical output to the validated run, 286.8 s, 0 OOM events. The aborted part was discarded (not resumed).
- 2026-11-19 18:32:39 PKT full run restarted from scratch (batch 8 chosen automatically, startup 9.5 s).
- 2026-11-19 ~18:39 PKT part1 was killed by a Claude Code restart (all child processes of the agent shell died) after 9 complete videos, 0 errors. The run was resumed as part2 at 18:41:55 (detached from the agent via WMI so a restart cannot kill it): `tools/resume_list.py` listed 1641 videos to do; `run.py --videos todo2.txt --done-file part2.jsonl.done`. Part2 startup took 215.6 s (GPU shared with a short edge-suite test at that moment); steady state ~24 frames/s analysed, GPU 97% busy, 0 errors, 0 OOM.
- Edge suite re-run with the final defaults (adaptive sampling, done-file, OOM handling): all 24 rows answered, the same 7 videos with error lines as the first version; corrupt-folder and empty-folder results unchanged.

## Versions and timings (dev machine, P2000)

Measured on the same 23 held-out videos (3606 s of video, 102,347 frames) unless marked "estimate":

| version (date) | what it does | measured | full 50 files (128.03 h video, 12.98 M frames) |
|---|---|---|---|
| v0 (2026-11-19) `tools/reference.py` | every frame, single process, Ultralytics `predict` API | ~10 frames/s (GPU 35-60% busy) on the first tuning videos | estimate 12.98 M / 10 = ~1.3 M s (~360 h) |
| v1 (2026-11-19) `run.py --k1 0` | every frame, decode processes + batched GPU + own NMS | 4255.6 s for the held-out set = 24.1 frames/s, GPU 97-99% busy | estimate 12.98 M / 24 = ~540,000 s (~150 h) |
| v2 (2026-11-19, simulated) | uniform 3 frames per second | 3 frames/s of video | estimate 128.03 h x 3600 x 3 / 23 = ~60,000 s (~17 h) |
| final (2026-11-19) `run.py` defaults | adaptive 1->3 frames per second | 287.3 s for the held-out set (5982 frames, 1.66 per s of video), GPU busy 251.7 s (88%) | measured: see HARDWARE.md (expected ~10 h: 1.8 x 460,900 s / ~24 frames/s) |

Where the time goes in the final version (held-out run, FP32): wall 287.3 s; GPU busy (model + NMS, timed around
each batch in the inference thread) 251.7 s = 88%; the rest is start-up (model load ~5-10 s, worker spawn) and the
tail where the last videos finish. CPU work per video is small next to the GPU: decode 0.56 ms/frame for all 102k
frames = ~57 s of CPU spread over 11 decode processes, BGR + letterbox ~1.2 ms per analysed frame (~7 s).
Measurement method: `time.perf_counter()` around each batch in the GPU thread (`gpu_busy_seconds`), run wall time
from `run.py`, per-stage CPU costs from `tools/profile_decode.py`.
- 2026-11-19 ~19:25 PKT: part2 produced 2 error lines "timed out waiting for sampling decisions" (train-00001:20, 480x360 -> letterbox 480x640; train-00002:5, 202x360 -> 640x384 portrait). Cause (a bug in the pipeline, not bad input): frames wait in per-shape groups until a full batch; a partial group was only flushed when the message queue was idle, and then only the largest group. With 11 busy decode workers the queue is never idle, so the last few frames of a video with a rare letterbox shape stayed queued, its stage-2 decisions could not be made, and the worker gave up after 600 s. Fixes: (1) any partial batch older than 0.5 s is submitted, checked on every loop iteration; (2) the worker no longer fails the video: it sends a heartbeat every 30 s while waiting and, after a long cap (3600 s), analyses its held candidate frames instead (same rules, more frames); (3) the hang watchdog now also runs under constant message traffic. Moving the watchdog exposed a race (a worker that exits normally while its "done" message is still queued was treated as crashed, seen in a CPU test as "worker 0 died (exit 0)"): a normal exit is now never treated as a crash, and the crashed worker's video is taken from the shared-memory record.
- Re-validation of these fixes on the CPU (`CUDA_VISIBLE_DEVICES=-1`, so the GPU stays with the full run): the 3 cross-check videos (shapes 480x360, 270x480, 640x360) give lines identical to the simulator's prediction (0 differing lines), 0 errors.
- Plan: part3 (`D:\vi-tools\fullrun_part3.cmd`, detached) waits for part2's END, then reruns with the fixed code the videos part1+2 did not finish plus those with the timeout error (`tools/resume_list.py --redo-errors`); `merge_parts.py` lets later parts win, so part3's result replaces the part2 error line.
- A CPU fault test with a 20 s / 60 s hang threshold also showed false "hung" verdicts for workers that were legitimately waiting (for decisions, or for a free frame slot while inference is slow). Workers now send a heartbeat every 30 s while waiting for a slot or for decisions, and the watchdog only looks at silence (default 600 s). Fault test after the fix (CPU, threshold 60 s, injected crash on edge:0 and hang on edge:7): exactly those two videos get "worker crashed or hung"; all other lines identical to the clean run.
- 2026-11-19 ~21:57 PKT the laptop shut down during part2 (probably heat; GPU reached 84 °C). Part2's last progress line: 11,711 s, 524 videos in its done-file (12 of them with the timeout error), 261,163 frames analysed (22.3/s), GPU busy 11,478 s (98%), 0 OOM.
- 2026-11-20 12:09:00 PKT part3 started (detached), with 8 decode workers instead of 11 to keep the laptop cooler (decode is not the bottleneck: GPU-bound). `tools/resume_list.py --redo-errors "timed out waiting for sampling decisions"`: 521 videos complete, 1129 to do (incl. the 12 re-runs). At 16,561 s: 819 videos, 378,542 frames (22.9/s), GPU busy 16,379 s (99%), 0 errors, 0 OOM.
- Expected merge result (from the dataset survey: lines per video = floor(last frame time) + 1, which matches all 23 held-out reference videos exactly): 1650 videos, 461,757 second-lines, and ~0 error lines (the survey found no unreadable video).
- 2026-11-20 requirements.txt: also pinned Ultralytics' own dependencies to the tested versions (matplotlib 3.11.2, polars 2.1.0 + polars-runtime-32 2.1.0, psutil 7.2.2, ultralytics-thop 2.2.5, nvidia-ml-py 13.615.71, ultralytics-platform 0.1.117, httpx 0.28.1); all have Linux cp311/py3 wheels and their declared requirements are consistent with the pins (checked from the wheels' metadata).

## Full 50-file run: final (2026-11-20)

- part3 finished 2026-11-20 18:26:11 PKT, rc 0: 1129 videos, 521,017 frames analysed (210,706 stage-2), 310,311 lines, 0 errors, 22,614.9 s, GPU busy 22,431.2 s (99%), 0 OOM, 0 CPU fallbacks.
- Merge (`tools/merge_parts.py`, parts 1-3): 1650 of 1650 videos, 0 missing, 461,757 per-second lines (= expected from the survey), 0 error lines -> `results.jsonl.gz`.
- Run time: part1 371 s + part2 11,711 s + part3 22,614.9 s = **34,697 s (9 h 38 min)** on the P2000; GPU busy 34,267 s (99%). Frames analysed: 788,482 (incl. redone videos) of 12,981,615 (~6.1%).
- Consistency check inside the full results: the 23 held-out videos score F1 0.9905 / exact 96.15% / within-1 99.86% vs the every-frame reference and their lines are identical to the held-out validation run; the 35 tuning videos score 0.9879 / 95.44% / 99.85%.
- Versions on the full 50 files (P2000): v0 every frame with the Ultralytics API ~360 h (estimate), v1 every frame through the pipeline ~150 h (estimate: 12.98 M frames / 24 per s), final 9 h 38 min (measured).

## requirements.txt resolution check (2026-11-20)

- `pip install --dry-run --ignore-installed --report --only-binary=:all: --platform manylinux_2_35..manylinux1_x86_64 --python-version 3.11 -r requirements.txt` (metadata only, no wheels downloaded): resolves, 41 packages, including torch 2.14.1+cu126, torchvision 0.29.1+cu126, ultralytics 8.4.215, opencv-python 5.1.0.95, av 18.1.0, pyarrow 26.0.1, numpy 2.4.6. First attempt with only `--platform manylinux_2_28_x86_64` failed for matplotlib 3.11.2 (its wheel is tagged manylinux2014/2_17; pip does not widen an explicit platform tag); with all compatible tags it resolves. Limitation: pip evaluates markers for the machine it runs on (Windows), so torch's Linux-only CUDA dependencies (nvidia-*-cu12, triton) are not in this list; on Linux torch's own metadata pins them (the Colab install covered that path).

---

## Separate test: another recording, outside FineVideo (2026-11-20)

This test is kept apart from the FineVideo results above. It uses a different recording that is not in the 50 files and was not used for any tuning, to see how the final settings behave on new footage.

- Video: built with `tools/make_unseen_video.py` from two archive.org clips (not in FineVideo; not committed because of their licences):
  "Shenzhen Electronics Markets - Back Alley Walking Tour" (identifier youtube-HOZUBk4ETO0, CC BY-SA 4.0, 1080p, seconds 60-300) +
  "Valmont Dog Park" (identifier Valmont_Dog_Park, CC BY-NC-ND 3.0, 720p, whole 56.2 s); re-encoded to one H.264 1280x720 video, 8830 frames, 296.1 s, wrapped as `unseen.parquet` row 0.
- Reference: `tools/reference.py` (Ultralytics predict on every frame): 297 seconds, 460.4 s processing (19.2 frames/s), 581 s wall.
- Final `run.py --fp32` (defaults): 843 frames analysed (2.84 per second of video; 546 stage-2), 100.7 s, 103 s wall, 0 errors.
- Score (`tools/score.py --per-video`): objects F1 0.9798 (P 0.9764 / R 0.9833), people exact **66.67%** (below the brief's 85%), within-1 97.98%.
- Why exact is low: the market walk is a dense crowd: the every-frame median is >= 6 people in 209 of 297 seconds (10+ in 61). On those seconds the default got the exact count in 124/209, vs 39/39 for 0 people. Sampling the same video more densely (simulated on its every-frame detections): uniform k=1 / 3 / 5 / 9 -> exact 50.5% / 67.0% / 74.8% / 81.1%, within-1 89.2% / 98.0% / 99.0% / 99.3%. Exact median counts of crowds need nearly every frame.
- How common crowds are in FineVideo: seconds with a reference median >= 6 people are 1.5% of the held-out set and 5.6% of the tuning set (default exact on them 81% / 80%); in the 50-file results 3.1% of seconds have >= 6 people and 79.5% have 0-1 people (exact ~98-99% at 0-1 in validation).
- Real runs on the same video with more frames (`run.py --fp32`): uniform 9 frames/s (`--k1 9 --k2 9`): 2669 frames, 128 s wall, F1 0.9917, exact 81.14%, within-1 99.33%. Adaptive 1->9 (`--k2 9`): 2481 frames, 120 s wall, F1 0.9917, exact 80.47%, within-1 99.33%. On the 50 files this would cost about 2x (adaptive 1->9, ~21 h) to 5x (uniform 9, ~50 h) the run time here for a gain only on crowded seconds, so the default stays 1->3.
- No setting was changed after this test.
