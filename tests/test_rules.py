"""Unit tests of the per-second rules (run: python tests/test_rules.py, or pytest)."""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vindex.common import Sampling, aggregate, pick_indices, second_of  # noqa: E402
from vindex.detector import letterbox_params  # noqa: E402

P = (True, False, False, False, False)  # person only
N = (False,) * 5


def test_aggregate_half_rule_and_median():
    assert aggregate([]) == ([], 0)
    # present in exactly half of the frames -> present
    assert aggregate([(P, 1), (N, 0)]) == (["person"], 0)  # median of [0,1] = 0.5 -> rounded down
    assert aggregate([(P, 1), (N, 0), (N, 0)]) == ([], 0)  # 1 of 3 < half
    assert aggregate([(P, 3), (P, 2), (P, 2)]) == (["person"], 2)
    assert aggregate([(P, 3), (P, 1), (P, 2), (P, 4)]) == (["person"], 2)  # (2+3)/2 = 2.5 -> 2
    assert aggregate([(P, 1), (P, 3)]) == (["person"], 2)  # (1+3)/2 = 2
    car = (False, True, False, False, False)
    assert aggregate([(car, 0), (car, 0), (N, 0)]) == (["car"], 0)


def test_pick_indices():
    assert pick_indices(30, 1) == [15]
    assert pick_indices(30, 3) == [5, 15, 25]
    assert pick_indices(2, 3) == [0, 1]  # fewer frames than k: all of them
    assert pick_indices(25, 0) == list(range(25))  # k=0: every frame
    for n in range(1, 61):
        for k in (1, 2, 3, 5):
            idx = pick_indices(n, k)
            assert idx == sorted(set(idx)) and all(0 <= i < n for i in idx) and len(idx) == min(n, k)


def test_stage_indices_union():
    sp = Sampling(1, 3)
    for n in range(1, 61):
        s1, s2 = sp.stage_indices(n)
        assert not set(s1) & set(s2)
        assert sorted(set(s1) | set(s2)) == sorted(set(pick_indices(n, 3)) | set(pick_indices(n, 1)))


def test_second_boundaries_exact():
    tb = Fraction(1, 15360)
    assert second_of(184320, 0, tb) == 12  # exactly 12.0 s belongs to second 12
    assert second_of(184319, 0, tb) == 11
    tb = Fraction(1001, 30000)  # 29.97 fps, pts in frames
    assert second_of(30, 0, tb) == 1  # 30 * 1001/30000 = 1.001 s
    assert second_of(29, 0, tb) == 0
    assert second_of(1000, 1000, Fraction(1, 1000)) == 0  # relative to the first frame


def test_letterbox_shapes():
    assert letterbox_params(360, 640)[2] == (384, 640)
    assert letterbox_params(640, 360)[2] == (640, 384)
    assert letterbox_params(480, 270)[2] == (640, 384)  # upscaled (scaleup=True), 360 wide -> padded to 384
    assert letterbox_params(2160, 3840)[2] == (384, 640)
    assert letterbox_params(75, 101)[2][1] == 640


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
