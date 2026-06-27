import math
from ccrp.metrics import reduction_pct


def test_reduction_pct():
    assert math.isclose(reduction_pct(1.0, 0.75), 25.0)


def test_reduction_pct_zero_baseline_is_zero():
    assert reduction_pct(0.0, 0.0) == 0.0
