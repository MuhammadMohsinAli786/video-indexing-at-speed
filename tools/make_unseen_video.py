"""Build one ~5-minute test video that is not in FineVideo and wrap it in a parquet file (`mp4` column).

Sources (archive.org; used only for this local test, the video itself is NOT in the repo because of the licences):
  * "Shenzhen Electronics Markets - Back Alley Walking Tour", identifier youtube-HOZUBk4ETO0, CC BY-SA 4.0,
    1920x1080 H.264 29.97 fps, 499.6 s: seconds 60-300 are used
  * "Valmont Dog Park", identifier Valmont_Dog_Park, CC BY-NC-ND 3.0, 1280x720 H.264 29.97 fps, 56.2 s: whole clip
Both are 16:9 without rotation, so scaling to 1280x720 keeps the aspect ratio. Output: 1280x720, 30 fps clock,
H.264 (libx264, crf 23), no audio; source frames keep their own times (29.97 -> 30 fps leaves a rare 1-frame gap).

    python -I tools/make_unseen_video.py <src_dir> <out_dir>     # writes unseen.mp4 and unseen.parquet (row 0)
"""
import os
import sys
from fractions import Fraction

import av
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
W, H = 1280, 720
CLIPS = [("shenzhen.mkv", 60.0, 300.0), ("dogpark.mov", 0.0, None)]


def main(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, "unseen.mp4")
    out = av.open(mp4_path, "w", format="mp4")
    st = out.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "23", "preset": "veryfast"}
    st.codec_context.time_base = Fraction(1, FPS)  # frames carry pts in 1/FPS units
    offset = 0.0
    last_pts = -1
    for name, start, end in CLIPS:
        inp = av.open(os.path.join(src_dir, name))
        vs = inp.streams.video[0]
        vs.thread_type = "AUTO"
        if start > 0:
            inp.seek(int(start / vs.time_base), stream=vs, backward=True)
        clip_end = 0.0
        for fr in inp.decode(vs):
            t = fr.time
            if t is None or t < start:
                continue
            if end is not None and t >= end:
                break
            rel = t - start
            pts = int(round((offset + rel) * FPS))
            if pts <= last_pts:
                continue  # source faster than 30 fps: drop extra frames
            img = fr.reformat(width=W, height=H, format="yuv420p")
            img.pts = pts
            img.time_base = Fraction(1, FPS)
            for p in st.encode(img):
                out.mux(p)
            last_pts = pts
            clip_end = rel
        inp.close()
        print(f"{name}: {clip_end:.1f} s used, total frames so far {last_pts + 1}")
        offset += clip_end + 1.0 / FPS
    for p in st.encode():
        out.mux(p)
    out.close()
    data = open(mp4_path, "rb").read()
    pq.write_table(pa.table({"mp4": pa.array([data], type=pa.binary())}),
                   os.path.join(out_dir, "unseen.parquet"))
    print(f"wrote {mp4_path} ({len(data) / 1e6:.1f} MB, {offset:.1f} s) and unseen.parquet")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
