"""Video decoding helpers (PyAV / FFmpeg) shared by run.py and the reference tool.

Timestamps: a frame's time is (pts - pts_of_first_decoded_frame) * time_base, computed exactly with
fractions, so second s holds the frames with s <= t < s + 1 (variable frame rate is handled naturally).

Rotation: FFmpeg stores a display matrix; ``frame.rotation`` is its counter-clockwise angle. Like the
ffmpeg CLI (autorotate) and OpenCV (CAP_PROP_ORIENTATION_AUTO), the frame is turned clockwise by
-rotation degrees before it is used.
"""
import io
import math
from fractions import Fraction

import av
import cv2
import numpy as np

av.logging.set_level(av.logging.PANIC)  # damaged inputs would otherwise flood stderr


class VideoError(Exception):
    """A video that cannot be processed (the message goes into the error line)."""


def open_container(data):
    if data is None or len(data) == 0:
        raise VideoError("empty mp4 value")
    try:
        c = av.open(io.BytesIO(data), mode="r", metadata_errors="ignore")
    except Exception as e:  # av.error.* (InvalidDataError, ...)
        raise VideoError(f"could not open video: {type(e).__name__}: {e}") from None
    if not c.streams.video:
        c.close()
        raise VideoError("no video stream")
    return c


def clockwise_degrees(rotation):
    """Clockwise rotation (0/90/180/270) needed for display, from the display-matrix CCW angle."""
    if not rotation:
        return 0
    return int(round(-float(rotation) / 90.0)) % 4 * 90


def rotate(img, cw):
    if cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def frame_rotation(frame):
    try:
        return clockwise_degrees(frame.rotation)
    except Exception:
        return 0


_REFORMATTER = None


def to_bgr(frame):
    """Decoded frame -> HxWx3 uint8 BGR array, rotated for display.

    Same pixels as ``frame.to_ndarray(format="bgr24")`` and as OpenCV's VideoCapture (both honour the
    stream's colour matrix, e.g. BT.709), but one shared reformatter keeps its SwsContext between frames:
    ~0.5 ms instead of ~7 ms per 640x360 frame.
    """
    global _REFORMATTER
    if _REFORMATTER is None:
        from av.video.reformatter import VideoReformatter
        _REFORMATTER = VideoReformatter()
    img = _REFORMATTER.reformat(frame, format="bgr24").to_ndarray()
    cw = frame_rotation(frame)
    if cw:
        img = rotate(img, cw)
    return np.ascontiguousarray(img)


def packet_pts(data):
    """Demux only (no decoding): sorted presentation timestamps of the video packets.

    Returns (pts_list, time_base). Cheap; used to plan which frames to sample.
    """
    c = open_container(data)
    try:
        vs = c.streams.video[0]
        pts = []
        for p in c.demux(vs):
            if p.pts is None or p.size == 0:
                continue
            if getattr(p, "is_discard", False):
                continue
            pts.append(p.pts)
        pts.sort()
        return pts, vs.time_base
    finally:
        c.close()


class FrameClock:
    """Assigns each decoded frame its exact time (Fraction seconds) relative to the first decoded frame.

    Frames without a pts get the previous time plus one nominal frame duration.
    """

    def __init__(self, stream):
        self.tb = stream.time_base or Fraction(1, 90000)
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        self.step = Fraction(1) / Fraction(rate) if rate else Fraction(1, 30)
        self.pts0 = None
        self.last_t = None

    def time(self, frame):
        pts = frame.pts
        if pts is None:
            t = Fraction(0) if self.last_t is None else self.last_t + self.step
        else:
            if self.pts0 is None:
                self.pts0 = pts
            t = Fraction(pts - self.pts0) * self.tb
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        return t


def sec(t):
    return math.floor(t)
