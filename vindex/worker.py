"""Decode worker process: video bytes -> letterboxed sample frames in shared memory slots.

For each video:
 1. demux only (no decoding) to get every frame's pts;
 2. decode every frame (H.264 needs it), and for each second pick k evenly spaced frames out of that
    second's frames (all frames if the second has <= k); only those are converted to BGR, rotated and
    letterboxed straight into a free shared-memory slot;
 3. if a second ends up with decoded frames but no sample (e.g. demux and decoder disagree), its last
    frame is used, so every second that holds a frame gets at least one sample.

Messages to the main process (one queue):
  ("start", wid, vid)
  ("f", wid, vid, second, slot, H, W)          a letterboxed frame is in `slot`
  ("done", wid, vid, n_seconds, error_or_None)
"""
import math
import os
import queue
import time
import traceback
from fractions import Fraction

import numpy as np

SLOT_BYTES = 640 * 640 * 3


def _plan(pts_sorted, pts0, tb, k):
    """Set of pts to sample: per second, pick_indices(n, k) among the frames of that second."""
    from .common import pick_indices

    per = {}
    for p in pts_sorted:
        if p < pts0:
            continue
        s = math.floor(Fraction(p - pts0) * tb)
        per.setdefault(s, []).append(p)
    chosen = set()
    for s, lst in per.items():
        for i in pick_indices(len(lst), k):
            chosen.add(lst[i])
    return chosen


def process_video(vid, data, k, get_slot, slots, send, wid, threads=1, deadline=None):
    """Decode one video and send its sample frames. Returns (n_seconds, error)."""
    import cv2  # noqa: F401
    from .detector import letterbox_into, letterbox_shape
    from .video import FrameClock, VideoError, open_container, packet_pts, sec, to_bgr

    try:
        pts_sorted, _ = packet_pts(data)
    except VideoError as e:
        return 0, str(e)
    except Exception as e:
        pts_sorted = []  # demux failed; decode will tell us more

    try:
        c = open_container(data)
    except VideoError as e:
        return 0, str(e)
    n_seconds = 0
    err = None
    try:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        vs.thread_count = threads
        clock = FrameClock(vs)
        chosen = None
        cur_s, cur_sampled, cur_last = None, False, None

        def emit(fr, s):
            img = to_bgr(fr)
            H, W = letterbox_shape(img.shape[0], img.shape[1])
            slot = get_slot()
            letterbox_into(img, slots[slot])
            send(("f", wid, vid, s, slot, H, W))

        for fr in c.decode(vs):
            t = clock.time(fr)
            if t < 0:
                continue
            s = sec(t)
            if chosen is None:
                chosen = _plan(pts_sorted, clock.pts0, clock.tb, k) if (pts_sorted and clock.pts0 is not None) else set()
                use_all_fallback = not chosen
            if s != cur_s:
                if cur_s is not None and not cur_sampled and cur_last is not None:
                    emit(cur_last, cur_s)
                cur_s, cur_sampled, cur_last = s, False, None
            if fr.pts in chosen:
                emit(fr, s)
                cur_sampled = True
                cur_last = None
            else:
                cur_last = fr
            n_seconds = max(n_seconds, s + 1)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("video took too long to decode")
        if cur_s is not None and not cur_sampled and cur_last is not None:
            emit(cur_last, cur_s)
    except Exception as e:
        err = f"decoding failed part way: {type(e).__name__}: {e}"
    finally:
        try:
            c.close()
        except Exception:
            pass
    if n_seconds == 0 and err is None:
        err = "could not decode video (no frames)"
    return n_seconds, err


def worker_main(wid, task_q, msg_q, free_q, shm_name, n_slots, k, threads, max_seconds_per_video):
    """Entry point of a decode worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    from multiprocessing import shared_memory

    cv2.setNumThreads(1)
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = np.ndarray((n_slots, 640, 640, 3), dtype=np.uint8, buffer=shm.buf)

    def get_slot():
        return free_q.get()

    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            vid, data = task
            msg_q.put(("start", wid, vid))
            try:
                deadline = time.monotonic() + max_seconds_per_video if max_seconds_per_video else None
                n, err = process_video(vid, data, k, get_slot, slots, msg_q.put, wid, threads, deadline)
            except Exception as e:  # never let one video kill the worker
                n, err = 0, f"internal error: {type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-300:]}"
            del data, task
            msg_q.put(("done", wid, vid, n, err))
    finally:
        del slots
        shm.close()
