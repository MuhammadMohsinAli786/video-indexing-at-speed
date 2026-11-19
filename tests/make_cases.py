"""Generate unusual / broken inputs to test run.py's robustness.

    python tests/make_cases.py --source finevideo/data/train-00000-of-01357.parquet --out tests/cases

Writes several folders under --out:
  edge/      edge.parquet with one row per case (see CASES below), plus a non-parquet file
  corrupt/   a parquet file that is random bytes, a truncated real parquet, a parquet without an mp4 column
  empty/     an empty folder
"""
import argparse
import io
import json
import os
import random
from fractions import Fraction

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def synth(codec="libx264", w=320, h=240, n=60, rate=30, pix="yuv420p", pts_list=None, container="mp4",
          rotation=None, options=None, audio=False, video=True):
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format=container)
    vs = None
    if video:
        vs = out.add_stream(codec, rate=rate)
        vs.width, vs.height, vs.pix_fmt = w, h, pix
        if options:
            vs.options = options
        if pts_list is not None:
            vs.time_base = Fraction(1, 1000)
            vs.codec_context.time_base = Fraction(1, 1000)
        if rotation is not None:
            vs.set_display_rotation(rotation)
    if audio:
        a = out.add_stream("aac", rate=44100)
    if video:
        for i in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            img[..., 0] = (np.arange(w)[None, :] + 4 * i) % 256
            img[..., 1] = (np.arange(h)[:, None] * 2) % 256
            img[..., 2] = (i * 7) % 256
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            if pts_list is not None:
                fr.pts = pts_list[i]
                fr.time_base = Fraction(1, 1000)
            for p in vs.encode(fr):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
    if audio:
        for i in range(40):
            af = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
            af.sample_rate = 44100
            af.pts = i * 1024
            for p in a.encode(af):
                out.mux(p)
        for p in a.encode():
            out.mux(p)
    out.close()
    return buf.getvalue()


def remux_rotated(data, rotation):
    """Copy a real video's packets into a new mp4 with a display-matrix rotation (no re-encode)."""
    inp = av.open(io.BytesIO(data))
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    ivs = inp.streams.video[0]
    ovs = out.add_stream_from_template(ivs)
    ovs.set_display_rotation(rotation)
    for p in inp.demux(ivs):
        if p.dts is None:
            continue
        p.stream = ovs
        out.mux(p)
    out.close()
    inp.close()
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="a FineVideo parquet file to take real videos from")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rnd = random.Random(0)

    real = []
    for i, b in enumerate(pq.ParquetFile(a.source).iter_batches(batch_size=1, columns=["mp4"])):
        d = b.column(0)[0].as_py()
        real.append(d)
    real.sort(key=len)
    short = real[1]  # a small real video
    short2 = real[2]

    vfr_pts = []
    t = 0
    for i in range(80):
        vfr_pts.append(t)
        t += rnd.choice([10, 33, 33, 50, 120, 200])
        if i == 40:
            t += 2500  # a 2.5 s gap: seconds with no frames must still get lines
    cases = [
        ("real_short", short),
        ("none_value", None),
        ("empty_bytes", b""),
        ("garbage_bytes", bytes(rnd.getrandbits(8) for _ in range(5000))),
        ("truncated_half", short2[: len(short2) // 2]),
        ("truncated_2000_bytes", short2[:2000]),
        ("audio_only", synth(video=False, audio=True)),
        ("under_1s_10_frames", synth(n=10)),
        ("single_frame", synth(n=1)),
        ("vfr_with_gap", synth(n=80, pts_list=vfr_pts)),
        ("rotated_90", remux_rotated(short, 90)),
        ("rotated_minus90", remux_rotated(short, -90)),
        ("rotated_180", remux_rotated(short, 180)),
        ("uhd_3840x2160", synth(w=3840, h=2160, n=20, options={"preset": "ultrafast"})),
        ("huge_7680x4320_hevc", synth(codec="libx265", w=7680, h=4320, n=4, options={"preset": "ultrafast"})),
        ("vp9", synth(codec="libvpx-vp9", n=30, options={"deadline": "realtime", "cpu-used": "8"})),
        ("av1", synth(codec="libsvtav1", w=320, h=240, n=30, options={"preset": "12"})),
        ("mpeg4_part2", synth(codec="mpeg4", n=30)),
        ("hevc_10bit", synth(codec="libx265", n=30, pix="yuv420p10le", options={"preset": "ultrafast"})),
        ("odd_size_101x75", synth(w=102, h=76, n=30)),
        ("mkv_container", synth(n=30, container="matroska")),
        ("corrupt_middle", None),  # filled below
        ("video_plus_audio", synth(n=45, audio=True)),
        ("bframes_long_gop", synth(n=90, options={"bf": "3", "g": "250"})),
    ]
    cm = bytearray(short2)
    lo, hi = int(len(cm) * 0.40), int(len(cm) * 0.45)
    cm[lo:hi] = bytes(rnd.getrandbits(8) for _ in range(hi - lo))
    cases[[c[0] for c in cases].index("corrupt_middle")] = ("corrupt_middle", bytes(cm))

    edge = os.path.join(a.out, "edge")
    os.makedirs(edge, exist_ok=True)
    tbl = pa.table({"mp4": pa.array([c[1] for c in cases], type=pa.binary()),
                    "json": pa.array([json.dumps({"case": c[0]}) for c in cases])})
    pq.write_table(tbl, os.path.join(edge, "edge.parquet"))
    with open(os.path.join(edge, "cases.txt"), "w") as f:
        for i, c in enumerate(cases):
            f.write(f"edge.parquet:{i}\t{c[0]}\t{0 if c[1] is None else len(c[1])} bytes\n")
    with open(os.path.join(edge, "not_a_parquet.txt"), "w") as f:
        f.write("ignore me\n")

    bad = os.path.join(a.out, "corrupt")
    os.makedirs(bad, exist_ok=True)
    with open(os.path.join(bad, "random_bytes.parquet"), "wb") as f:
        f.write(bytes(rnd.getrandbits(8) for _ in range(100000)))
    raw = open(a.source, "rb").read()
    with open(os.path.join(bad, "truncated.parquet"), "wb") as f:
        f.write(raw[: len(raw) // 2])
    pq.write_table(pa.table({"json": pa.array(["{}", "{}"])}), os.path.join(bad, "no_mp4_column.parquet"))
    pq.write_table(pa.table({"mp4": pa.array([short], type=pa.binary())}), os.path.join(bad, "ok_one_row.parquet"))
    os.makedirs(os.path.join(a.out, "empty"), exist_ok=True)
    for i, c in enumerate(cases):
        print(i, c[0], 0 if c[1] is None else len(c[1]))


if __name__ == "__main__":
    main()
