"""Constants and the per-second rules shared by run.py, the reference tool and the offline simulator."""
from fractions import Fraction
import math

# The five objects, in output order, with their COCO class ids in YOLO11.
OBJECTS = ("person", "car", "dog", "laptop", "cell phone")
CLASS_IDS = (0, 2, 16, 63, 67)
PERSON = 0  # index into OBJECTS

CONF = 0.4      # a detection counts only if confidence >= 0.4
IOU = 0.7       # Ultralytics default NMS IoU
IMGSZ = 640     # model input size
MAX_DET = 300   # Ultralytics default max detections per image


def second_of(pts, pts0, time_base):
    """Second index of a frame: floor((pts - pts0) * time_base), exact (no float rounding)."""
    return math.floor(Fraction(pts - pts0) * time_base)


def pick_indices(n, k):
    """Evenly spaced indices of k samples out of n frames of one second (all frames if n <= k).

    Sample j is the frame at the centre of the j-th of k equal slices of the second's frame list.
    """
    if k <= 0 or n <= k:
        return list(range(n))
    return sorted({min(n - 1, int((j + 0.5) * n / k)) for j in range(k)})


def aggregate(frames):
    """Combine per-frame results of one second the way the grader does.

    frames: list of (present, people) where present is a 5-tuple/bitmask-like of bools in OBJECTS order.
    An object is present if it is found in at least half of the frames; people is the median of the
    per-frame counts, rounded down when it falls halfway.
    """
    m = len(frames)
    if m == 0:
        return [], 0
    objs = [name for i, name in enumerate(OBJECTS) if 2 * sum(1 for p, _ in frames if p[i]) >= m]
    c = sorted(n for _, n in frames)
    people = c[m // 2] if m % 2 else (c[m // 2 - 1] + c[m // 2]) // 2
    return objs, people


def video_lines(video_id, per_second, n_seconds):
    """Lines for seconds 0..n_seconds-1; seconds without frames get [] / 0."""
    out = []
    for s in range(n_seconds):
        objs, people = aggregate(per_second.get(s, []))
        out.append({"video": video_id, "second": s, "objects": objs, "people": people})
    return out
