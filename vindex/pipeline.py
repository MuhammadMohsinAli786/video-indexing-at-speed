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

from .common import aggregate
from .worker import CUR_BYTES, SLOT_BYTES, get_current, set_current, worker_main

HANG_SECONDS = float(os.environ.get("VINDEX_HANG_SECONDS", "600"))  # silent worker (with free slots) = hung


def list_parquets(folder):
    if not os.path.isdir(folder):
        raise SystemExit(f"input folder not found: {folder}")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(".parquet") and os.path.isfile(os.path.join(folder, f)))


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
    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.n = 0
        self.last_flush = time.monotonic()

    def write(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.n += 1
        if time.monotonic() - self.last_flush > 2.0:
            self.f.flush()
            self.last_flush = time.monotonic()

    def close(self):
        self.f.flush()
        self.f.close()


def video_lines(vid, samples, n_seconds):
    lines = []
    for s in range(n_seconds):
        objs, people = aggregate(samples.get(s, []))
        lines.append({"video": vid, "second": s, "objects": objs, "people": people})
    return lines


class GPURunner(threading.Thread):
    """Runs detector batches from `jobs` and puts results on `results` (one thread owns the GPU)."""

    def __init__(self, det, batch, n_bufs, dump):
        super().__init__(daemon=True)
        import torch

        self.det, self.dump = det, dump
        pin = det.device.type == "cuda"
        self.bufs = [torch.empty(batch * SLOT_BYTES, dtype=torch.uint8, pin_memory=pin) for _ in range(n_bufs)]
        self.np_bufs = [b.numpy() for b in self.bufs]
        self.free = queue.Queue()
        for i in range(n_bufs):
            self.free.put(i)
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.busy_seconds = 0.0

    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                break
            items, H, W, bi = job
            n = len(items)
            t0 = time.perf_counter()
            try:
                x = self.bufs[bi][: n * H * W * 3].view(n, H, W, 3)
                if self.dump:
                    present, people, dets = self.det.detect(x, conf_thres=0.25, return_dets=True)
                else:
                    (present, people), dets = self.det.detect(x), None
                out = (items, present.cpu().numpy(), people.cpu().numpy(), dets, None)
            except Exception as e:  # e.g. CUDA OOM on an odd batch: report, keep going
                out = (items, None, None, None, f"{type(e).__name__}: {e}")
            self.busy_seconds += time.perf_counter() - t0
            self.free.put(bi)
            self.results.put(out)


def run(input_dir, output, weights, k=3, workers=None, batch=None, n_slots=None, half=None,
        threads_per_worker=1, log=print, only=None, dump_dir=None):
    t_start = time.time()
    files = list_parquets(input_dir)
    writer = Writer(output)
    if not files:
        writer.close()
        log("no .parquet files found; wrote an empty output file")
        return {"videos": 0, "frames": 0, "lines": 0, "errors": 0, "seconds": time.time() - t_start}

    import torch
    from .detector import Detector

    det = Detector(weights, "cuda:0" if torch.cuda.is_available() else "cpu", half=half)
    if batch is None:
        batch = auto_batch(det)
    cpu = os.cpu_count() or 4
    if workers is None:
        workers = max(1, min(cpu - 1, 48))
    if n_slots is None:
        n_slots = max(6 * batch, 4 * workers, 64)
    log(f"device={det.device} ({torch.cuda.get_device_name(0) if det.device.type == 'cuda' else 'cpu'}) "
        f"half={det.half} batch={batch} workers={workers} slots={n_slots} k={k} "
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

    def start_worker(wid):
        set_current(cur, wid, None)
        p = ctx.Process(target=worker_main, args=(wid, task_q, msg_q, free_q, shm.name, n_slots, k,
                                                  threads_per_worker, None, cur), daemon=True)
        p.start()
        return p

    procs = {wid: start_worker(wid) for wid in range(workers)}
    rt = threading.Thread(target=reader_thread, args=(files, task_q, err_lines, workers, stop, only), daemon=True)
    rt.start()
    gpu = GPURunner(det, batch, 3, bool(dump_dir))
    gpu.start()

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
    n_videos = n_frames = n_err = 0
    jobs_out = 0

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
        inflight.pop(vid, None)
        n_videos += 1
        if dump_dir:
            fr = sorted(dumps.pop(vid, []))
            with gzip.open(os.path.join(dump_dir, vid.replace(":", "__") + ".json.gz"), "wt") as g:
                json.dump({"video": vid, "error": err, "n_seconds": n_sec, "frames": fr}, g)

    def submit(shape):
        nonlocal jobs_out
        items = pending.pop(shape)
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
                items, present, people, dets, gerr = gpu.results.get(block=block, timeout=0.5 if block else None)
            except queue.Empty:
                return
            block = False
            jobs_out -= 1
            for i, (_, vid, s, pts) in enumerate(items):
                if gerr is not None:
                    gpu_errors.setdefault(vid, f"inference failed: {gerr}"[:500])
                else:
                    samples[vid][s].append((tuple(bool(b) for b in present[i]), int(people[i])))
                    if dets is not None:
                        dumps[vid].append((pts if pts is not None else -1, s, dets[i]))
                inflight[vid] -= 1
                if inflight[vid] == 0 and vid in done_info:
                    finish(vid)
            n_frames += len(items)

    def flush_largest():
        if pending:
            submit(max(pending, key=lambda s: len(pending[s])))

    last_report = time.monotonic()
    while True:
        while not err_lines.empty():
            writer.write(err_lines.get())
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
                _, _, vid, s, slot, H, W, pts = m
                if vid in done_info and current.get(wid) != vid:
                    free_q.put(slot)  # late frame of a video the watchdog already gave up on
                    continue
                inflight[vid] += 1
                pending[(H, W)].append((slot, vid, s, pts))
                if len(pending[(H, W)]) >= batch:
                    submit((H, W))
            elif kind == "start":
                current[wid] = m[2]
            elif kind == "done":
                _, _, vid, n_sec, err = m
                current.pop(wid, None)
                if vid not in done_info:  # else already handled by the watchdog
                    done_info[vid] = (n_sec, err)
                    if inflight[vid] == 0:
                        finish(vid)
        else:
            if pending and gpu.jobs.qsize() == 0:
                flush_largest()  # GPU idle and no new frames: run what we have
            now = time.monotonic()
            for wid, p in list(procs.items()):
                if p.exitcode == 0 and wid not in current and get_current(cur, wid) is None:
                    procs.pop(wid)  # finished normally (got its None)
                    continue
                dead = p.exitcode is not None
                hung = wid in current and now - last_msg[wid] > HANG_SECONDS and free_q.qsize() > 0
                if not (dead or hung):
                    continue
                if hung:
                    p.kill()
                p.join(timeout=5)
                vid = current.pop(wid, None) or get_current(cur, wid)
                if vid is not None and vid not in done_info:
                    done_info[vid] = (0, "worker crashed or hung while decoding this video")
                    if inflight[vid] == 0:
                        finish(vid)
                log(f"worker {wid} {'died (exit %s)' % p.exitcode if dead else 'hung'} on {vid}; restarting")
                procs[wid] = start_worker(wid)
                if not rt.is_alive():
                    task_q.put(None)  # make sure the replacement can always exit
                last_msg[wid] = time.monotonic()
            if not procs and not pending and not rt.is_alive():
                break
        if time.monotonic() - last_report > 60:
            last_report = time.monotonic()
            el = time.time() - t_start
            log(f"[{el:7.0f}s] videos={n_videos} frames={n_frames} ({n_frames / el:.1f}/s) "
                f"gpu_busy={gpu.busy_seconds:.0f}s lines={writer.n} errors={n_err}")

    while pending:
        flush_largest()
    while jobs_out > 0:
        handle_results(block=True)
    for vid in list(done_info):
        finish(vid)
    while not err_lines.empty():
        writer.write(err_lines.get())
        n_err += 1
    writer.close()
    stop.set()
    gpu.jobs.put(None)
    for p in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    del slots
    shm.close()
    shm.unlink()
    el = time.time() - t_start
    res = {"videos": n_videos, "frames": n_frames, "lines": writer.n, "errors": n_err, "seconds": round(el, 1),
           "gpu_busy_seconds": round(gpu.busy_seconds, 1)}
    log(f"done: {res}")
    return res


def auto_batch(det, start=64):
    """Largest batch (<= start, power of two) whose 640x640 forward fits in free GPU memory."""
    import torch

    if det.device.type != "cuda":
        return 4
    b = start
    while b > 1:
        try:
            det.detect(torch.zeros((b, 640, 640, 3), dtype=torch.uint8))
            torch.cuda.synchronize()
            return b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b //= 2
    return 1
