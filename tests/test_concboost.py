import types
import numpy as np
from platypus.inference.distill import _dw_concboost


def _rows():
    f = [(b'q', b'a', 0, repr(('fact', 'A%d' % i, 'r', 'C')), None, None) for i in range(18)]
    c = [(b'q', b'a', 0, repr(('conclusion', 'X%d' % i, 'is connected to', 'Y')), None, None) for i in range(2)]
    return {0: f + c, 1: [(b'q', b'a', 1, repr(('fact', 'B', 'r', 'D')), None, None)]}


def test_concboost_off_leaves_weights():
    b = _rows(); w = {}
    _dw_concboost(types.SimpleNamespace(inf_distill_wake_concboost=0, inf_distill_wake_decay=0.5, inf_distill_wake_decay_from=1), b, w, 4)
    assert w == {}


def test_concboost_scales_conclusions():
    b = _rows(); w = {}
    _dw_concboost(types.SimpleNamespace(inf_distill_wake_concboost=1, inf_distill_wake_decay=0.5, inf_distill_wake_decay_from=1), b, w, 4)
    p = w[0]; assert abs(p.sum() - 1) < 1e-9 and len(p) == 20
    assert abs(p[-1] / p[0] - 8.0) < 1e-9                          # 0.5 ** -(4 - 1)
    assert abs(w[1].sum() - 1) < 1e-9


def test_concboost_keeps_spaced_weights():
    b = _rows(); w = {0: np.r_[np.full(18, 1 / 40), np.full(2, 1 / 40) * 0 + (1 - 18 / 40) / 2]}
    _dw_concboost(types.SimpleNamespace(inf_distill_wake_concboost=1, inf_distill_wake_decay=0.5, inf_distill_wake_decay_from=2), b, w, 2)
    assert abs(w[0].sum() - 1) < 1e-9
