"""One-time preparation (untimed, needs internet): download the official YOLO11l weights and self-test the GPU.

    python prepare.py

* downloads weights/yolo11l.pt (official Ultralytics release asset, SHA-256 checked);
* loads the model on the GPU, runs a test batch and checks that our batched inference + NMS gives the same
  result as Ultralytics' own predict on a test image (so a broken install shows up here, not in the timed run).
After this, `python run.py ...` needs no network.
"""
import hashlib
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "yolo11l.pt")
SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"
URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11l.pt",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    if os.path.isfile(WEIGHTS) and sha256(WEIGHTS) == SHA256:
        print(f"weights present: {WEIGHTS}")
        return
    for url in URLS:
        for attempt in range(3):
            try:
                print(f"downloading {url} (attempt {attempt + 1})")
                tmp = WEIGHTS + ".part"
                urllib.request.urlretrieve(url, tmp)
                if sha256(tmp) == SHA256:
                    os.replace(tmp, WEIGHTS)
                    print(f"saved {WEIGHTS}")
                    return
                print("checksum mismatch, retrying")
            except Exception as e:
                print(f"  failed: {e}")
                time.sleep(2)
    sys.exit("could not download yolo11l.pt")


def self_test():
    os.environ.setdefault("YOLO_OFFLINE", "1")
    sys.path.insert(0, HERE)
    import numpy as np
    import torch

    from vindex.common import CLASS_IDS, CONF, IOU
    from vindex.detector import Detector, letterbox_into

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU visible; run.py will fall back to CPU (very slow)")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    det = Detector(WEIGHTS, "cuda:0")
    yolo = YOLO(WEIGHTS)
    for name in ("bus.jpg", "zidane.jpg"):  # sample images shipped inside the ultralytics package
        img = cv2.imread(str(ASSETS / name))
        x = torch.from_numpy(np.stack([letterbox_into(img)] * 4))
        t = time.time()
        present, people, _, _ = det.detect(x)
        torch.cuda.synchronize()
        r = yolo.predict(img, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
        cls = r.boxes.cls.int().tolist()
        ref = ([c in cls for c in CLASS_IDS], cls.count(0))
        ours = (present[0].tolist(), int(people[0]))
        print(f"self-test {name} (half={det.half}, {time.time() - t:.2f}s): ours={ours} ultralytics={ref} -> "
              f"{'OK' if ours == ref else 'DIFFERENT'}")


if __name__ == "__main__":
    download()
    self_test()
    print("preparation done")
