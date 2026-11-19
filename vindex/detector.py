"""YOLO11l inference with Ultralytics-identical preprocessing and NMS, batched for speed.

Matches ``YOLO("yolo11l.pt").predict(frame, imgsz=640, iou=0.7)`` for a single BGR frame:
  * letterbox: Ultralytics LetterBox(640, auto=True (minimum rectangle, stride 32), scaleup, center,
    pad 114, cv2.INTER_LINEAR) -- reimplemented in ``letterbox_into`` to write straight into a buffer;
  * preprocess: uint8 BHWC -> BCHW, BGR->RGB, float, /255 (same op order as the Ultralytics predictor);
  * model: Ultralytics AutoBackend (fused Conv+BN), FP32 or FP16;
  * NMS: best class over all 80 scores, conf >= 0.4, class-aware NMS IoU 0.7 with the same
    class offset (cls * 7680) as Ultralytics, max_det 300 per image. Images of a batch are separated by a
    float64 offset, so one NMS call serves the whole batch without changing any per-image result.
Only the five reported classes are kept after the argmax (NMS is class-aware, so the other classes
cannot change them).
"""
import os

import cv2
import numpy as np

from .common import CLASS_IDS, CONF, IMGSZ, IOU, MAX_DET

STRIDE = 32
PAD = 114
MAX_WH = 7680  # Ultralytics class offset for class-aware NMS


def letterbox_params(h, w, imgsz=IMGSZ, stride=STRIDE):
    """Ultralytics LetterBox.get_params with auto=True, scaleup=True, center=True."""
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = imgsz - new_w, imgsz - new_h
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return (new_w, new_h), (top, bottom, left, right), (new_h + top + bottom, new_w + left + right)


def letterbox_shape(h, w):
    return letterbox_params(h, w)[2]


def letterbox_into(img, out=None):
    """Letterbox a BGR uint8 image exactly like Ultralytics; write into ``out`` (HxWx3 view) if given."""
    h, w = img.shape[:2]
    (nw, nh), (top, bottom, left, right), (H, W) = letterbox_params(h, w)
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if out is None:
        out = np.empty((H, W, 3), np.uint8)
    else:
        out = out[:H, :W]
    if top or bottom or left or right:
        out[:top] = PAD
        out[top + nh:] = PAD
        out[top:top + nh, :left] = PAD
        out[top:top + nh, left + nw:] = PAD
    out[top:top + nh, left:left + nw] = img
    return out


class Detector:
    """Batched YOLO11l on the GPU. ``detect(batch_uint8_bhwc)`` -> (present[B,5], people[B], unsure, dets)."""

    def __init__(self, weights, device="cuda:0", half=None, verbose=False):
        import torch
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics.nn.autobackend import AutoBackend

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"model weights not found: {weights} (run the preparation command first)")
        self.torch = torch
        self.device = torch.device(device)
        if half is None:  # FP16 only where the GPU has fast FP16 (Volta+); Pascal FP16 is slow
            half = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7
        self.half = bool(half)
        torch.backends.cudnn.benchmark = True
        self.model = AutoBackend(weights, device=self.device, fp16=self.half, fuse=True, verbose=verbose)
        self.model.eval()
        self.cls_ids = torch.tensor(CLASS_IDS, device=self.device)
        # map COCO id -> index in OBJECTS (or -1)
        lut = torch.full((80,), -1, dtype=torch.long)
        for i, c in enumerate(CLASS_IDS):
            lut[c] = i
        self.lut = lut.to(self.device)

    def preprocess(self, x):
        """uint8 BHWC (CPU or GPU tensor) -> model input, same op order as the Ultralytics predictor."""
        torch = self.torch
        x = x.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.flip(1)  # BGR -> RGB
        x = (x.half() if self.half else x.float()).div_(255)
        return x

    def raw(self, x):
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y  # (B, 4 + 80, N)

    def postprocess(self, pred, band=None, return_dets=False):
        """Ultralytics non_max_suppression restricted to our classes, batched.

        Returns (present[B,5] bool, people[B] int, unsure[B] bool or None, dets or None).
        present/people count only detections with conf >= CONF (0.4). With ``band=(lo, hi)``, NMS also keeps
        boxes down to ``lo`` and ``unsure`` flags images having a kept detection with lo <= conf < hi.
        Lower-scoring boxes never suppress higher-scoring ones, so the conf >= 0.4 result is unchanged.
        With ``return_dets`` (needs band or uses CONF), the kept [(object_index, conf)] per image are returned.
        """
        torch = self.torch
        import torchvision

        thr = min(CONF, band[0]) if band else CONF
        pred = pred.float()
        B = pred.shape[0]
        conf, j = pred[:, 4:84, :].max(1)  # best class per box over all 80 classes (B, N)
        keep = (conf >= thr) & (self.lut[j] >= 0)
        present = torch.zeros((B, len(CLASS_IDS)), dtype=torch.bool, device=pred.device)
        people = torch.zeros((B,), dtype=torch.long, device=pred.device)
        unsure = torch.zeros((B,), dtype=torch.bool, device=pred.device) if band else None
        if not bool(keep.any()):
            return present, people, unsure, ([[] for _ in range(B)] if return_dets else None)
        b_idx, n_idx = keep.nonzero(as_tuple=True)
        xywh = pred[b_idx, 0:4, n_idx]  # (M, 4)
        xy, wh = xywh[:, :2], xywh[:, 2:] / 2
        xyxy = torch.cat((xy - wh, xy + wh), 1)  # Ultralytics xywh2xyxy
        cls = j[b_idx, n_idx]
        sc = conf[b_idx, n_idx]
        c = cls.float().unsqueeze(1) * MAX_WH  # class offset, float32 like Ultralytics
        boxes = (xyxy + c).double() + b_idx.double().unsqueeze(1) * float(2 ** 23)  # per-image offset, exact
        k = torchvision.ops.nms(boxes, sc.double(), IOU)  # indices sorted by score, descending
        kb, kc, ks = b_idx[k], self.lut[cls[k]], sc[k]
        counts = torch.bincount(kb, minlength=B)
        if int(counts.max()) > MAX_DET:  # rare: apply max_det per image (keep the top-scoring 300)
            order = torch.argsort(kb, stable=True)  # group by image, scores stay descending inside
            kb_s = kb[order]
            first = torch.cumsum(counts, 0) - counts
            rank = torch.arange(kb_s.numel(), device=kb.device) - first[kb_s]
            ok = rank < MAX_DET
            kb, kc, ks = kb_s[ok], kc[order][ok], ks[order][ok]
        dets = None
        if return_dets:
            dets = [[] for _ in range(B)]
            for b, c_, s_ in zip(kb.tolist(), kc.tolist(), ks.tolist()):
                dets[b].append((c_, round(s_, 4)))
        if band:
            inband = (ks >= band[0]) & (ks < band[1])
            unsure[kb[inband]] = True
        strong = ks >= CONF
        kb, kc = kb[strong], kc[strong]
        present[kb, kc] = True
        people = torch.bincount(kb[kc == 0], minlength=B)
        return present, people, unsure, dets

    def detect(self, batch, band=None, return_dets=False):
        """uint8 BHWC letterboxed batch -> (present, people, unsure, dets); see postprocess."""
        with self.torch.inference_mode():
            x = self.preprocess(batch)
            return self.postprocess(self.raw(x), band, return_dets)
