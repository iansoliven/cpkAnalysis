from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import stats


def test_process_capability_lower_basic_ratio() -> None:
    value = stats._process_capability_lower(mean=5.0, stdev=2.0, lsl=2.0)
    assert value == pytest.approx((5.0 - 2.0) / (3 * 2.0))


def test_process_capability_lower_zero_stdev_handles_infinite_bounds() -> None:
    assert math.isinf(stats._process_capability_lower(mean=5.0, stdev=0.0, lsl=4.0))
    negative = stats._process_capability_lower(mean=3.0, stdev=0.0, lsl=4.0)
    assert negative == -math.inf


def test_process_capability_upper_none_returns_none() -> None:
    assert stats._process_capability_upper(mean=5.0, stdev=1.0, usl=None) is None


def test_compute_cpk_prefers_smaller_capability() -> None:
    cpk = stats._compute_cpk(center=5.0, sigma=1.0, lower=2.0, upper=6.0)
    # Lower capability is upper bound: (6-5)/(3*1) = 0.333...
    assert cpk == pytest.approx((6.0 - 5.0) / (3.0 * 1.0))


def test_compute_cpk_single_bound() -> None:
    cpk = stats._compute_cpk(center=10.0, sigma=2.0, lower=None, upper=16.0)
    assert cpk == pytest.approx((16.0 - 10.0) / (3.0 * 2.0))


def test_compute_cpk_returns_none_without_limits() -> None:
    assert stats._compute_cpk(center=1.0, sigma=1.0, lower=None, upper=None) is None


def test_select_limit_uses_what_if_over_spec_and_stdf() -> None:
    info = {
        "what_if_lower": 1.0,
        "spec_lower": 0.5,
        "stdf_lower": 0.0,
    }
    limit, source = stats._select_limit("lower", info)
    assert limit == pytest.approx(1.0)
    assert source == "what_if"


def test_select_limit_returns_unset_when_no_limits() -> None:
    limit, source = stats._select_limit("upper", {})
    assert limit is None
    assert source == "unset"


def test_yield_loss_counts_failures() -> None:
    values = np.array([0.0, 0.5, 1.4, 2.0])
    loss = stats._yield_loss(values, lower=0.5, upper=1.5)
    # Failures: 0.0 (below), 2.0 (above) -> 2/4
    assert loss == pytest.approx(0.5)


def test_yield_loss_without_limits_returns_nan() -> None:
    values = np.array([1.0, 2.0])
    assert math.isnan(stats._yield_loss(values, lower=None, upper=None))


def test_yield_loss_empty_array_returns_nan() -> None:
    assert math.isnan(stats._yield_loss(np.array([]), lower=0.0, upper=1.0))
