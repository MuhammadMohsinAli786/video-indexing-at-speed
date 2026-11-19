"""Orchestrator: parquet reader -> decode worker processes -> batched GPU inference -> per-second lines.

Main process:
  * a reader thread streams the `mp4` column of every .parquet file (row by row) into a bounded task queue;
  * N decode worker processes (vindex.worker) write letterboxed sample frames into shared-memory slots;
  * the main thread groups ready frames by letterbox shape, gathers a batch into a pinned buffer and hands
    it to the inference thread (model + NMS on the GPU), so CPU work overlaps GPU work;
  * when all frames of a video are through, its lines (seconds 0..last) are written to the output file;
  * a watchdog restarts a worker that dies or hangs and writes an error line for its video.
"""
import gzip
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np

from .common import Sampling, aggregate
from .worker import CUR_BYTES, SLOT_BYTES, get_current, set_current, worker_main

HANG_SECONDS = float(os.environ.get("VINDEX_HANG_SECONDS", "600"))  # a worker silent this long is hung
MAX_PENDING_WAIT = 0.5  # seconds a partial batch may wait for more frames of its shape


def list_parquets(folder):
    if not os.path.isdir(folder):
        raise SystemExit(f"input folder not found: {folder}")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(".parquet") and os.path.isfile(os.path.join(folder, f)))
    if not files:  # tolerate a folder that only holds the files in sub-folders
        for root, _, names in os.walk(folder):
            files += [os.path.join(root, f) for f in names if f.lower().endswith(".parquet")]
        files.sort()
    return files


def reader_thread(files, task_q, err_lines, n_workers, stop, only=None):
    """Stream (video_id, mp4 bytes) tasks; unreadable files/rows become error lines."""
    import pyarrow.parquet as pq

    for path in files:
        name = os.path.basename(path)
        if stop.is_set():
            break
        if only is not None and not any(v.startswith(name + ":") for v in only):
            continue
        try:
            pf = pq.ParquetFile(path)
            nrows = pf.metadata.num_rows
        except Exception as e:
            err_lines.put({"video": name, "error": f"could not read parquet file: {type(e).__name__}: {e}"[:500]})
            continue
        if "mp4" not in pf.schema_arrow.names:
            for r in range(nrows):
                err_lines.put({"video": f"{name}:{r}", "error": "no mp4 column in parquet file"})
            continue
        row = 0
        try:
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"], use_threads=False):
                col = batch.column(0)
                for i in range(len(col)):
                    vid = f"{name}:{row}"
                    row += 1
                    if only is not None and vid not in only:
                        continue
                    v = col[i].as_py()
                    if not v:
                        err_lines.put({"video": vid, "error": "empty mp4 value"})
                        continue
                    if isinstance(v, str):
                        v = v.encode("latin-1", "ignore")
                    while not stop.is_set():
                        try:
                            task_q.put((vid, v), timeout=1.0)
                            break
                        except queue.Full:
                            continue
        except Exception as e:
            msg = f"could not read row from parquet file: {type(e).__name__}: {e}"[:500]
            for r in range(row, max(nrows, row + 1)):
                if only is None or f"{name}:{r}" in only:
                    err_lines.put({"video": f"{name}:{r}", "error": msg})
    for _ in range(n_workers):
        task_q.put(None)


class Writer:
    """Output lines. With a done-file, a video id is appended there only after its lines are flushed to the
    output, so after an interruption every video listed in the done-file is complete (used for resuming)."""

    def __init__(self, path, done_path=None):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.done_f = open(done_path, "a", encoding="utf-8", newline="\n") if done_path else None
        self.finished = []
        self.n = 0
        self.last_flush = time.monotonic()

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.n += 1

    def write_error(self, obj):
        self.write(obj)
        self.video_done(obj["video"])

    def video_done(self, vid):
        self.finished.append(vid)
        if time.monotonic() - self.last_flush > 2.0:
            self.flush()

    def flush(self):
        self.f.flush()
        if self.done_f:
            os.fsync(self.f.fileno())
            if self.finished:
                self.done_f.write("".join(v + "\n" for v in self.finished))
                self.done_f.flush()
        self.finished = []
        self.last_flush = time.monotonic()

    def close(self):
        self.flush()
        self.f.close()
        if self.done_f:
            self.done_f.close()


def video_lines(vid, samples, n_seconds):
    lines = []
    for s in range(n_seconds):
        objs, people = aggregate(samples.get(s, []))
        lines.append({"video": vid, "second": s, "objects": objs, "people": people})
    return lines


class GPURunner:
    """Inference threads: each takes a batch from `jobs`, runs model + NMS on its own CUDA stream and puts the
    result on `results`. Two threads keep the GPU busy while the other one waits on NMS / copies."""

    def __init__(self, dets, batch, dump, band):
        """dets: one Detector per thread (the Ultralytics head caches anchors per input shape: not thread-safe)."""
        import torch

        self.torch = torch
        det = dets[0]
        n_threads = len(dets)
        self.dump, self.band = dump, band
        cuda = det.device.type == "cuda"
        n_bufs = 2 * n_threads + 1
        self.bufs = [torch.empty(batch * SLOT_BYTES, dtype=torch.uint8, pin_memory=cuda) for _ in range(n_bufs)]
        self.np_bufs = [b.numpy() for b in self.bufs]
        self.free = queue.Queue()
        for i in range(n_bufs):
            self.free.put(i)
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.busy_seconds = 0.0
        self.oom_events = 0
        self.cpu_fallbacks = 0
        self.cpu_det = None
        self.cpu_lock = threading.Lock()
        self.threads = [threading.Thread(target=self._loop, args=(d, torch.cuda.Stream(det.device) if cuda else None),
                                         daemon=True) for d in dets]
        for t in self.threads:
            t.start()

    def stop(self):
        for _ in self.threads:
            self.jobs.put(None)

    def _free_gpu(self):
        try:
            self.torch.cuda.synchronize()
        except Exception:
            pass
        self.torch.cuda.empty_cache()

    def _detect_safe(self, det, x, band):
        """det.detect with out-of-memory recovery, results on the CPU.

        On CUDA out-of-memory: free the cache and run the batch in two halves (recursively); a single image is
        retried 3 times (2 s apart) and finally run on the CPU, so a transient memory shortage never costs a video.
        """
        torch = self.torch
        tries = 3 if x.shape[0] == 1 else 1
        for attempt in range(tries):
            try:
                r = det.detect(x, band=band, return_dets=self.dump)
                return tuple(None if t is None else t.cpu() for t in r[:3]) + (r[3],)
            except RuntimeError as e:  # torch.OutOfMemoryError and torch.AcceleratorError are RuntimeErrors
                if "out of memory" not in str(e).lower():
                    raise
                self.oom_events += 1
                self._free_gpu()
                if x.shape[0] == 1 and attempt < tries - 1:
                    time.sleep(2)
        if x.shape[0] > 1:
            h = x.shape[0] // 2
            a = self._detect_safe(det, x[:h], band)
            b = self._detect_safe(det, x[h:], band)
            cat = lambda u, v: None if u is None else torch.cat((u, v))  # noqa: E731
            return (cat(a[0], b[0]), cat(a[1], b[1]), cat(a[2], b[2]),
                    None if a[3] is None else a[3] + b[3])
        with self.cpu_lock:  # last resort for this one image: the CPU
            if self.cpu_det is None:
                from .detector import Detector
                self.cpu_det = Detector(det.weights, "cpu", half=False)
            self.cpu_fallbacks += 1
            r = self.cpu_det.detect(x.cpu(), band=band, return_dets=self.dump)
            return tuple(None if t is None else t.cpu() for t in r[:3]) + (r[3],)

    def _loop(self, det, stream):
        import contextlib

        ctx = self.torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with ctx:
            while True:
                job = self.jobs.get()
                if job is None:
                    break
                items, H, W, bi = job
                n = len(items)
                t0 = time.perf_counter()
                try:
                    x = self.bufs[bi][: n * H * W * 3].view(n, H, W, 3)
                    band = (0.25, 0.25) if self.dump and not self.band else self.band
                    present, people, unsure, dets = self._detect_safe(det, x, band)
                    out = (items, present.cpu().numpy(), people.cpu().numpy(),
                           None if unsure is None else unsure.cpu().numpy(), dets, None)
                except Exception as e:  # e.g. CUDA OOM on an odd batch: report, keep going
                    out = (items, None, None, None, None, f"{type(e).__name__}: {e}")
                self.busy_seconds += time.perf_counter() - t0
                self.free.put(bi)
                self.results.put(out)


def run(input_dir, output, weights, sampling=None, workers=None, batch=None, n_slots=None, half=None,
        threads_per_worker=1, log=print, only=None, dump_dir=None, gpu_threads=None, done_file=None):
    sampling = sampling or Sampling()
    t_start = time.time()
    files = list_parquets(input_dir)
    writer = Writer(output, done_file)
    if not files:
        writer.close()
        log("no .parquet files found; wrote an empty output file")
        return {"videos": 0, "frames": 0, "lines": 0, "errors": 0, "seconds": time.time() - t_start}

    import torch
    from .detector import Detector

    det = Detector(weights, "cuda:0" if torch.cuda.is_available() else "cpu", half=half)
    gpu_mem = torch.cuda.get_device_properties(det.device).total_memory if det.device.type == "cuda" else 0
    if gpu_threads is None:  # a second inference thread hides NMS/copy gaps; needs memory for two batches
        gpu_threads = 2 if gpu_mem >= 8 * 2 ** 30 else 1
    if batch is None:
        batch = auto_batch(det, 64 if gpu_mem >= 20 * 2 ** 30 else 32 if gpu_mem >= 10 * 2 ** 30
                           else 16 if gpu_mem >= 6 * 2 ** 30 else 8)
    cpu = os.cpu_count() or 4
    if workers is None:
        workers = max(1, min(cpu - 1, 48))
    if n_slots is None:
        n_slots = max(6 * batch, 4 * workers, 64)
    log(f"device={det.device} ({torch.cuda.get_device_name(0) if det.device.type == 'cuda' else 'cpu'}) "
        f"half={det.half} batch={batch} gpu_threads={gpu_threads} workers={workers} slots={n_slots} "
        f"sampling: {sampling.describe()} "
        f"startup={time.time() - t_start:.1f}s")

    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=n_slots * SLOT_BYTES)
    slots = np.ndarray((n_slots, SLOT_BYTES), dtype=np.uint8, buffer=shm.buf)
    task_q = ctx.Queue(maxsize=max(2, workers))
    msg_q = ctx.Queue()
    free_q = ctx.Queue()
    for i in range(n_slots):
        free_q.put(i)
    err_lines = queue.Queue()
    stop = threading.Event()
    cur = ctx.RawArray("c", CUR_BYTES * workers)  # what each worker is decoding (written synchronously)
    ctrl = {}  # wid -> control queue (stage-2 decisions)

    def start_worker(wid):
        set_current(cur, wid, None)
        ctrl[wid] = ctx.Queue()
        p = ctx.Process(target=worker_main, args=(wid, task_q, msg_q, free_q, ctrl[wid], shm.name, n_slots, sampling,
                                                  threads_per_worker, cur), daemon=True)
        p.start()
        return p

    procs = {wid: start_worker(wid) for wid in range(workers)}
    rt = threading.Thread(target=reader_thread, args=(files, task_q, err_lines, workers, stop, only), daemon=True)
    rt.start()
    dets = [det] + [Detector(weights, det.device, half=det.half) for _ in range(max(1, gpu_threads) - 1)]
    gpu = GPURunner(dets, batch, bool(dump_dir), sampling.band if sampling.adaptive else None)

    samples = defaultdict(lambda: defaultdict(list))  # vid -> second -> [(present, people)]
    dumps = defaultdict(list)  # vid -> [(pts, second, dets)] when dump_dir is set
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    inflight = defaultdict(int)  # vid -> frames not yet through the GPU
    gpu_errors = {}  # vid -> error from the GPU side
    done_info = {}  # vid -> (n_seconds, err) once the worker finished it
    current = {}  # wid -> vid being processed
    last_msg = {wid: time.monotonic() for wid in procs}
    pending = defaultdict(list)  # (H, W) -> [(slot, vid, second, pts)]
    n_videos = n_frames = n_err = n_stage2 = 0
    jobs_out = 0
    # adaptive sampling state, per video
    owner = {}  # vid -> wid
    s1 = defaultdict(lambda: defaultdict(list))  # vid -> second -> stage-1 [(present, people)]
    s1_unsure = defaultdict(dict)  # vid -> second -> bool
    s1_expected = defaultdict(dict)  # vid -> second -> number of stage-1 frames sent
    cands = defaultdict(set)  # vid -> seconds whose stage-2 decision is pending
    last_sec = defaultdict(lambda: -1)  # vid -> highest second announced
    s1_finished = set()  # vids whose decoding ended

    def s1_ready(vid, t):
        return t in s1_expected[vid] and len(s1[vid].get(t, ())) >= s1_expected[vid][t]

    def s1_result(vid, t):
        return aggregate(s1[vid][t]) if t in s1_expected[vid] else None

    def try_decide(vid):
        if not cands.get(vid):
            return
        for s in sorted(cands[vid]):
            if not s1_ready(vid, s):
                continue
            ok = True
            for t in (s - 1, s + 1):
                known = t < 0 or t <= last_sec[vid] or vid in s1_finished
                if not known or (t in s1_expected[vid] and not s1_ready(vid, t)):
                    ok = False
                    break
            if not ok:
                continue
            more = sampling.needs_more(s1_result(vid, s - 1), s1_result(vid, s), s1_result(vid, s + 1),
                                       s1_unsure[vid].get(s, False))
            cands[vid].discard(s)
            q = ctrl.get(owner.get(vid))
            if q is not None:
                q.put(("more" if more else "drop", vid, s))

    def finish(vid):
        nonlocal n_videos, n_err
        n_sec, err = done_info.pop(vid)
        err = err or gpu_errors.pop(vid, None)
        smp = samples.pop(vid, {})
        if n_sec == 0 and smp:
            n_sec = max(smp) + 1
        for line in video_lines(vid, smp, n_sec):
            writer.write(line)
        if err:
            writer.write({"video": vid, "error": err})
            n_err += 1
        writer.video_done(vid)
        inflight.pop(vid, None)
        for d in (owner, s1, s1_unsure, s1_expected, cands, last_sec):
            d.pop(vid, None)
        s1_finished.discard(vid)
        n_videos += 1
        if dump_dir:
            fr = sorted(dumps.pop(vid, []))
            with gzip.open(os.path.join(dump_dir, vid.replace(":", "__") + ".json.gz"), "wt") as g:
                json.dump({"video": vid, "error": err, "n_seconds": n_sec, "frames": fr}, g)

    def submit(shape):
        nonlocal jobs_out
        items = pending.pop(shape)
        pending_since.pop(shape, None)
        H, W = shape
        n, fb = len(items), H * W * 3
        bi = gpu.free.get()  # blocks only while all pinned buffers are queued on the GPU
        dst = gpu.np_bufs[bi][: n * fb].reshape(n, fb)
        for i, it in enumerate(items):
            dst[i] = slots[it[0], :fb]
            free_q.put(it[0])
        gpu.jobs.put((items, H, W, bi))
        jobs_out += 1

    def handle_results(block=False):
        nonlocal n_frames, jobs_out
        while True:
            try:
                items, present, people, unsure, dets, gerr = gpu.results.get(block=block,
                                                                             timeout=0.5 if block else None)
            except queue.Empty:
                return
            block = False
            jobs_out -= 1
            touched = set()
            for i, (_, vid, s, pts, stage) in enumerate(items):
                if gerr is not None:
                    gpu_errors.setdefault(vid, f"inference failed: {gerr}"[:500])
                    r = ((False,) * 5, 0)
                else:
                    r = (tuple(bool(b) for b in present[i]), int(people[i]))
                    samples[vid][s].append(r)
                    if dets is not None:
                        dumps[vid].append((pts if pts is not None else -1, s, dets[i]))
                if stage == 1 and sampling.adaptive:
                    s1[vid][s].append(r)
                    if unsure is not None and unsure[i]:
                        s1_unsure[vid][s] = True
                    touched.add(vid)
                inflight[vid] -= 1
                if inflight[vid] == 0 and vid in done_info:
                    finish(vid)
            for vid in touched:
                if vid not in done_info:
                    try_decide(vid)
            n_frames += len(items)

    def flush_largest():
        if pending:
            submit(max(pending, key=lambda s: len(pending[s])))

    pending_since = {}  # shape -> time its oldest pending frame arrived
    last_watch = 0.0
    last_report = time.monotonic()
    while True:
        while not err_lines.empty():
            writer.write_error(err_lines.get())
            n_err += 1
        handle_results()
        try:
            m = msg_q.get(timeout=0.01 if pending else 0.2)
        except queue.Empty:
            m = None
        if m is not None:
            kind, wid = m[0], m[1]
            last_msg[wid] = time.monotonic()
            if kind == "f":
                _, _, vid, s, slot, H, W, pts, stage = m
                if vid in done_info and current.get(wid) != vid:
                    free_q.put(slot)  # late frame of a video the watchdog already gave up on
                    continue
                inflight[vid] += 1
                n_stage2 += stage == 2
                if (H, W) not in pending:
                    pending_since[(H, W)] = time.monotonic()
                pending[(H, W)].append((slot, vid, s, pts, stage))
                if len(pending[(H, W)]) >= batch:
                    submit((H, W))
            elif kind == "start":
                current[wid] = m[2]
                owner[m[2]] = wid
            elif kind == "sec":
                _, _, vid, s, n1, has_c = m
                s1_expected[vid][s] = n1
                last_sec[vid] = max(last_sec[vid], s)
                if has_c:
                    cands[vid].add(s)
                try_decide(vid)
            elif kind == "s1done":
                s1_finished.add(m[2])
                try_decide(m[2])
            elif kind == "done":
                _, _, vid, n_sec, err = m
                current.pop(wid, None)
                if vid not in done_info:  # else already handled by the watchdog
                    done_info[vid] = (n_sec, err)
                    if inflight[vid] == 0:
                        finish(vid)
        now = time.monotonic()
        if m is None and pending and gpu.jobs.qsize() == 0 and gpu.free.qsize() > 0:
            flush_largest()  # GPU idle and no new frames: run what we have
        # A partial batch never waits long, even while other shapes keep the queue busy: a rare letterbox shape
        # (e.g. 480x360 or 202x360 video) would otherwise hold its last frames, and with them the video's
        # sampling decisions, indefinitely.
        for shape in [sh for sh, t0 in list(pending_since.items()) if now - t0 > MAX_PENDING_WAIT]:
            if shape in pending:
                submit(shape)
        if now - last_watch > 0.5:  # watchdog, also under constant message traffic
            last_watch = now
            for wid, p in list(procs.items()):
                if p.exitcode == 0:
                    procs.pop(wid)  # normal exit after its None; its last messages may still be queued
                    continue
                dead = p.exitcode is not None
                hung = wid in current and now - last_msg[wid] > HANG_SECONDS  # waiting workers send heartbeats
                if not (dead or hung):
                    continue
                if hung:
                    p.kill()
                p.join(timeout=5)
                # the shared-memory record is authoritative (queued messages can make `current` stale)
                vid = get_current(cur, wid)
                if current.get(wid) == vid:
                    current.pop(wid, None)
                if vid is not None and vid not in done_info:
                    done_info[vid] = (0, "worker crashed or hung while decoding this video")
                    if inflight[vid] == 0:
                        finish(vid)
                log(f"worker {wid} {'died (exit %s)' % p.exitcode if dead else 'hung'} on {vid}; restarting")
                procs[wid] = start_worker(wid)
                if not rt.is_alive():
                    task_q.put(None)  # make sure the replacement can always exit
                last_msg[wid] = time.monotonic()
        if m is None and not procs and not pending and not rt.is_alive():
            break
        if time.monotonic() - last_report > 60:
            last_report = time.monotonic()
            el = time.time() - t_start
            log(f"[{el:7.0f}s] videos={n_videos} frames={n_frames} ({n_frames / el:.1f}/s, stage2={n_stage2}) "
                f"gpu_busy={gpu.busy_seconds:.0f}s lines={writer.n} errors={n_err} oom={gpu.oom_events} "
                f"cpu_fallback={gpu.cpu_fallbacks}")

    while pending:
        flush_largest()
    while jobs_out > 0:
        handle_results(block=True)
    for vid in list(done_info):
        finish(vid)
    while not err_lines.empty():
        writer.write_error(err_lines.get())
        n_err += 1
    writer.close()
    stop.set()
    gpu.stop()
    for p in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    del slots
    shm.close()
    shm.unlink()
    el = time.time() - t_start
    res = {"videos": n_videos, "frames": n_frames, "stage2_frames": n_stage2, "lines": writer.n, "errors": n_err,
           "seconds": round(el, 1),
           "gpu_busy_seconds": round(gpu.busy_seconds, 1), "oom_events": gpu.oom_events,
           "cpu_fallback_frames": gpu.cpu_fallbacks}
    log(f"done: {res}")
    return res


def auto_batch(det, start=64):
    """Largest batch (<= start, halving) whose worst-case 640x640 forward fits; the probe's memory is released
    afterwards (a cached probe on a 4 GB Windows GPU caused out-of-memory errors later, see measurements)."""
    import torch

    if det.device.type != "cuda":
        return 4
    b = start
    try:
        while b > 1:
            try:
                det.detect(torch.zeros((b, 640, 640, 3), dtype=torch.uint8))
                torch.cuda.synchronize()
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                b //= 2
    finally:
        torch.cuda.empty_cache()
    return max(b, 1)
