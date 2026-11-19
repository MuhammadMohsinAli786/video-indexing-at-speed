"""Out-of-memory recovery of the inference threads (no GPU needed): python tests/test_oom_recovery.py"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from vindex.pipeline import GPURunner  # noqa: E402


class FakeDet:
    """present[i, 0] = image value % 2, people = image value; OOM for batches > 2 and always for value 5."""
    weights = "unused"

    def __init__(self):
        self.calls = 0

    def detect(self, x, band=None, return_dets=False):
        self.calls += 1
        v = x[:, 0, 0, 0].long()
        if x.shape[0] > 2 or bool((v == 5).any()):
            raise torch.AcceleratorError("CUDA error: out of memory") if hasattr(torch, "AcceleratorError") \
                else RuntimeError("CUDA error: out of memory")
        present = torch.zeros((x.shape[0], 5), dtype=torch.bool)
        present[:, 0] = (v % 2).bool()
        return present, v, None, None


class CpuDet(FakeDet):
    def detect(self, x, band=None, return_dets=False):
        v = x[:, 0, 0, 0].long()
        present = torch.zeros((x.shape[0], 5), dtype=torch.bool)
        present[:, 0] = (v % 2).bool()
        return present, v, None, None


def test_split_and_cpu_fallback():
    r = GPURunner.__new__(GPURunner)
    r.torch, r.dump = torch, False
    r.oom_events = r.cpu_fallbacks = 0
    r.cpu_det, r.cpu_lock = CpuDet(), threading.Lock()
    r._free_gpu = lambda: None
    import vindex.pipeline as P
    P.time.sleep = lambda s: None  # no waiting in the test
    x = torch.arange(8, dtype=torch.uint8).view(8, 1, 1, 1).expand(8, 4, 4, 3).contiguous()
    present, people, unsure, dets = r._detect_safe(FakeDet(), x, None)
    assert people.tolist() == list(range(8)), people.tolist()  # order preserved, every image answered
    assert present[:, 0].tolist() == [bool(i % 2) for i in range(8)]
    assert r.cpu_fallbacks == 1 and r.oom_events >= 4


if __name__ == "__main__":
    test_split_and_cpu_fallback()
    print("ok test_split_and_cpu_fallback")
