"""Orchestrator: parquet reader -> decode worker processes -> batched GPU inference -> per-second lines.

Main process:
  * a reader thread streams the `mp4` column of every .parquet file (row by row) into a bounded task queue;
  * N decode worker processes (vindex.worker) put letterboxed sample frames into shared-memory slots;
  * the main loop groups ready frames by letterbox shape, runs YOLO11l in batches on the GPU, frees the
    slots, and when a video is complete writes its lines (seconds 0..last) to the output file;
  * a watchdog restarts a worker that dies or hangs and writes an error line for its video.
"""
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np

from .common import OBJECTS, aggregate
from .worker import worker_main

HANG_SECONDS = 600  # a worker silent this long while slots are free is considered hung


def list_parquets(folder):
    if not os.path.isdir(folder):
        raise SystemExit(f"input folder not found: {folder}")
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(".parquet") and os.path.isfile(os.path.join(folder, f)))


def reader_thread(files, task_q, err_lines, n_workers, stop):
    """Stream (video_id, mp4 bytes) tasks; unreadable files/rows become error lines."""
    import pyarrow.parquet as pq

    for path in files:
        name = os.path.basename(path)
        if stop.is_set():
            break
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
                    v = col[i].as_py()
                    vid = f"{name}:{row}"
                    row += 1
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


def run(input_dir, output, weights, k=3, workers=None, batch=None, n_slots=None, half=None,
        threads_per_worker=1, log=print, stats=None):
    t_start = time.time()
    files = list_parquets(input_dir)
    writer = Writer(output)
    if not files:
        writer.close()
        log("no .parquet files found; wrote an empty output file")
        return {"videos": 0, "lines": 0, "seconds": time.time() - t_start}

    import torch
    from .detector import Detector

    det = Detector(weights, "cuda:0" if torch.cuda.is_available() else "cpu", half=half)
    if batch is None:
        batch = auto_batch(det)
    cpu = os.cpu_count() or 4
    if workers is None:
        workers = max(1, min(cpu - 1, 32))
    if n_slots is None:
        n_slots = max(4 * batch, 2 * workers + batch, 64)
    log(f"device={det.device} half={det.half} batch={batch} workers={workers} slots={n_slots} k={k}")

    ctx = mp.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=n_slots * 640 * 640 * 3)
    slots = np.ndarray((n_slots, 640, 640, 3), dtype=np.uint8, buffer=shm.buf)
    task_q = ctx.Queue(maxsize=max(2, workers))
    msg_q = ctx.Queue()
    free_q = ctx.Queue()
    for i in range(n_slots):
        free_q.put(i)
    err_lines = queue.Queue()
    stop = threading.Event()

    def start_worker(wid):
        p = ctx.Process(target=worker_main, args=(wid, task_q, msg_q, free_q, shm.name, n_slots, k,
                                                  threads_per_worker, None), daemon=True)
        p.start()
        return p

    procs = {wid: start_worker(wid) for wid in range(workers)}
    rt = threading.Thread(target=reader_thread, args=(files, task_q, err_lines, workers, stop), daemon=True)
    rt.start()

    samples = defaultdict(lambda: defaultdict(list))  # vid -> second -> [(present, people)]
    inflight = defaultdict(int)  # vid -> frames not yet through the GPU
    done_info = {}  # vid -> (n_seconds, err) once the worker finished it
    current = {}  # wid -> vid being processed
    last_msg = {wid: time.monotonic() for wid in procs}
    pending = defaultdict(list)  # (H, W) -> [(slot, vid, second)]
    n_videos = n_frames = n_err = 0
    t_gpu = 0.0

    def finish(vid):
        nonlocal n_videos, n_err
        n_sec, err = done_info.pop(vid)
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

    def run_batch(shape):
        nonlocal n_frames, t_gpu
        items = pending.pop(shape)
        H, W = shape
        ids = [it[0] for it in items]
        x = torch.from_numpy(slots[ids, :H, :W])  # copy out of shared memory
        for sid in ids:
            free_q.put(sid)
        t0 = time.perf_counter()
        present, people = det.detect(x)
        present, people = present.cpu().numpy(), people.cpu().numpy()
        t_gpu += time.perf_counter() - t0
        for (sid, vid, s), pr, pp in zip(items, present, people):
            samples[vid][s].append((tuple(bool(b) for b in pr), int(pp)))
            inflight[vid] -= 1
            if inflight[vid] == 0 and vid in done_info:
                finish(vid)
        n_frames += len(items)

    def flush_largest():
        if pending:
            run_batch(max(pending, key=lambda s: len(pending[s])))

    last_report = time.monotonic()
    while True:
        while not err_lines.empty():
            writer.write(err_lines.get())
            n_err += 1
        try:
            m = msg_q.get(timeout=0.02 if pending else 0.5)
        except queue.Empty:
            m = None
        if m is not None:
            kind, wid = m[0], m[1]
            last_msg[wid] = time.monotonic()
            if kind == "f":
                _, _, vid, s, slot, H, W = m
                if vid in done_info and done_info[vid][1] and current.get(wid) != vid:
                    free_q.put(slot)  # late frame of a video already declared failed by the watchdog
                    continue
                inflight[vid] += 1
                pending[(H, W)].append((slot, vid, s))
                if len(pending[(H, W)]) >= batch:
                    run_batch((H, W))
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
            if pending:
                flush_largest()
            now = time.monotonic()
            for wid, p in list(procs.items()):
                if p.exitcode == 0 and wid not in current:
                    procs.pop(wid)  # finished normally (got its None)
                    continue
                dead = p.exitcode is not None
                hung = wid in current and now - last_msg[wid] > HANG_SECONDS and free_q.qsize() > 0
                if not (dead or hung):
                    continue
                if hung:
                    p.kill()
                p.join(timeout=5)
                vid = current.pop(wid, None)
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
            log(f"[{el:7.0f}s] videos={n_videos} frames={n_frames} ({n_frames / el:.1f}/s) gpu={t_gpu:.0f}s "
                f"lines={writer.n} errors={n_err}")

    while pending:
        flush_largest()
    for vid in list(done_info):
        finish(vid)
    while not err_lines.empty():
        writer.write(err_lines.get())
        n_err += 1
    writer.close()
    stop.set()
    for p in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    del slots
    shm.close()
    shm.unlink()
    el = time.time() - t_start
    res = {"videos": n_videos, "frames": n_frames, "lines": writer.n, "errors": n_err, "seconds": el,
           "gpu_seconds": t_gpu}
    log(f"done: {res}")
    return res


def auto_batch(det, start=64):
    """Largest batch (<= start) whose 640x640 forward fits in free GPU memory."""
    import torch

    if det.device.type != "cuda":
        return 4
    b = start
    while b >= 1:
        try:
            x = torch.zeros((b, 640, 640, 3), dtype=torch.uint8)
            det.detect(x)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            return b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b //= 2
    return 1
