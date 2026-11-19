"""Decode worker process: video bytes -> letterboxed sample frames in shared memory slots.

For each video:
 1. demux only (no decoding) to get every frame's pts, and plan per second (vindex.common.Sampling):
    stage-1 frames (k1 evenly spaced) and stage-2 candidates (the extra frames of k2 evenly spaced);
 2. decode every frame (H.264 needs it); a stage-1 frame is converted to BGR, rotated, letterboxed into a free
    shared-memory slot and announced; stage-2 candidates are kept (as decoded frames) until the main process,
    which has the stage-1 results, says "more" (send them) or "drop" for that second;
 3. if a second ends with decoded frames but no stage-1 sample (e.g. demux and decoder disagree), its last
    frame is used as its stage-1 sample, so every second that holds a frame gets at least one sample.

Messages to the main process (one queue, FIFO per worker):
  ("start", wid, vid)
  ("f", wid, vid, second, slot, H, W, pts, stage)   a letterboxed frame is in `slot`
  ("sec", wid, vid, second, n_stage1, has_candidates)  all stage-1 frames of `second` have been sent
  ("s1done", wid, vid, n_seconds)                    decoding finished (stage-2 decisions may be pending)
  ("done", wid, vid, n_seconds, error_or_None)
Control messages from the main process (per-worker queue): ("more"|"drop", vid, second).
"""
import math
import os
import queue
import time
import traceback
from fractions import Fraction

import numpy as np

SLOT_BYTES = 640 * 640 * 3
CUR_BYTES = 512  # per-worker record of the video being processed (survives a hard crash)
DECISION_WAIT_CAP = float(os.environ.get("VINDEX_DECISION_WAIT_CAP", "3600"))  # seconds, then analyse all candidates


def _plan(pts_sorted, pts0, tb, sampling):
    """pts -> stage (1 or 2) for the frames to sample."""
    per = {}
    for p in pts_sorted:
        if p < pts0:
            continue
        s = math.floor(Fraction(p - pts0) * tb)
        per.setdefault(s, []).append(p)
    plan = {}
    for s, lst in per.items():
        i1, i2 = sampling.stage_indices(len(lst))
        for i in i1:
            plan[lst[i]] = 1
        for i in i2:
            plan.setdefault(lst[i], 2)
    return plan


def process_video(vid, data, sampling, get_slot, slots, send, wid, threads, ctrl):
    """Decode one video and send its sample frames. Returns (n_seconds, error)."""
    from .detector import letterbox_into, letterbox_shape
    from .video import FrameClock, VideoError, open_container, packet_pts, sec, to_bgr

    try:
        pts_sorted, _ = packet_pts(data)
    except VideoError as e:
        return 0, str(e)
    except Exception:
        pts_sorted = []  # demux failed; decoding will tell us more

    try:
        c = open_container(data)
    except VideoError as e:
        return 0, str(e)
    every = sampling.k1 <= 0
    n_seconds = 0
    err = None
    hold = {}  # second -> [decoded frames] (stage-2 candidates awaiting a decision)
    state = {"s": None, "n1": 0, "last": None}

    def emit(fr, s, stage):
        img = to_bgr(fr)
        H, W = letterbox_shape(img.shape[0], img.shape[1])
        slot = get_slot()
        letterbox_into(img, slots[slot][: H * W * 3].reshape(H, W, 3))
        send(("f", wid, vid, s, slot, H, W, fr.pts, stage))

    def handle(msg):
        kind, mvid, s = msg
        if mvid != vid:
            return  # stale decision for an earlier video
        frames = hold.pop(s, None)
        if kind == "more" and frames:
            for fr in frames:
                emit(fr, s, 2)

    def poll():
        while hold:
            try:
                handle(ctrl.get_nowait())
            except queue.Empty:
                return

    def end_second():
        s = state["s"]
        if s is None:
            return
        if state["n1"] == 0 and state["last"] is not None:
            fr = state["last"]
            if s in hold and fr in hold[s]:
                hold[s].remove(fr)
                if not hold[s]:
                    del hold[s]
            emit(fr, s, 1)
            state["n1"] = 1
        send(("sec", wid, vid, s, state["n1"], s in hold))
        state["s"] = None

    try:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        vs.thread_count = threads
        clock = FrameClock(vs)
        plan = None
        for fr in c.decode(vs):
            t = clock.time(fr)
            if t < 0:
                continue
            s = sec(t)
            if plan is None:
                plan = _plan(pts_sorted, clock.pts0, clock.tb, sampling) if (pts_sorted and clock.pts0 is not None) else {}
            if s != state["s"]:
                end_second()
                state.update(s=s, n1=0, last=None)
            stage = 1 if every else plan.get(fr.pts)
            if stage == 1:
                emit(fr, s, 1)
                state["n1"] += 1
                state["last"] = None
            else:
                if stage == 2:
                    hold.setdefault(s, []).append(fr)
                state["last"] = fr
            n_seconds = max(n_seconds, s + 1)
            if hold:
                poll()
    except Exception as e:
        err = f"decoding failed part way: {type(e).__name__}: {e}"
    try:
        end_second()
    except Exception as e:
        err = err or f"decoding failed part way: {type(e).__name__}: {e}"
    finally:
        try:
            c.close()
        except Exception:
            pass
    send(("s1done", wid, vid, n_seconds))
    # Wait for the remaining stage-2 decisions. They need this video's last stage-1 results from the GPU, which
    # can take long on a slow or busy GPU: keep the main process' watchdog informed, and never fail the video
    # for it. After a very long wait (main process stuck) the candidates are simply analysed too (still exact
    # sampling rules, just more frames).
    waited = 0.0
    while hold:
        try:
            handle(ctrl.get(timeout=30))
        except queue.Empty:
            waited += 30
            send(("alive", wid, vid))
            if waited >= DECISION_WAIT_CAP:
                for s in sorted(hold):
                    handle(("more", vid, s))
    if n_seconds == 0 and err is None:
        err = "could not decode video (no frames)"
    return n_seconds, err


def set_current(cur, wid, vid):
    b = (vid or "").encode("utf-8")[: CUR_BYTES - 1]
    cur[wid * CUR_BYTES: wid * CUR_BYTES + len(b) + 1] = b + bytes(1)


def get_current(cur, wid):
    raw = bytes(cur[wid * CUR_BYTES: (wid + 1) * CUR_BYTES])
    return raw.split(bytes(1), 1)[0].decode("utf-8", "ignore") or None


def worker_main(wid, task_q, msg_q, free_q, ctrl_q, shm_name, n_slots, sampling, threads, cur):
    """Entry point of a decode worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    from multiprocessing import shared_memory

    cv2.setNumThreads(1)
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = np.ndarray((n_slots, SLOT_BYTES), dtype=np.uint8, buffer=shm.buf)  # frame (H, W, 3) stored contiguously

    def get_slot():
        """A free shared-memory slot; while all are in use (slow GPU) keep the watchdog informed."""
        while True:
            try:
                return free_q.get(timeout=30)
            except queue.Empty:
                msg_q.put(("alive", wid, None))

    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            vid, data = task
            set_current(cur, wid, vid)
            msg_q.put(("start", wid, vid))
            if vid == os.environ.get("VINDEX_TEST_CRASH"):  # test hook: simulate a hard crash in FFmpeg
                os._exit(3)
            if vid == os.environ.get("VINDEX_TEST_HANG"):  # test hook: simulate a hang
                time.sleep(10 ** 6)
            try:
                n, err = process_video(vid, data, sampling, get_slot, slots, msg_q.put, wid, threads, ctrl_q)
            except Exception as e:  # never let one video kill the worker
                n, err = 0, f"internal error: {type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-300:]}"
            del data, task
            msg_q.put(("done", wid, vid, n, err))
            set_current(cur, wid, None)
    finally:
        del slots
        shm.close()
