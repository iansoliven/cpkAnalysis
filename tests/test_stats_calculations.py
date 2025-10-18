from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_compute_summary_respects_limit_precedence_and_skips_empty_groups() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "test_name": "T1", "test_number": "1", "value": 0.9, "units": ""},
            {"file": "lot1", "test_name": "T1", "test_number": "1", "value": 1.1, "units": "V"},
            {"file": "lot1", "test_name": "T1", "test_number": "1", "value": float("nan"), "units": "ignored"},
            {"file": "lot2", "test_name": "T2", "test_number": "1", "value": float("nan"), "units": "A"},
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "test_name": "T1",
                "test_number": "1",
                "unit": "V",
                "what_if_lower": 0.5,
                "spec_lower": 0.2,
                "stdf_lower": 0.1,
                "what_if_upper": 1.5,
                "spec_upper": 1.2,
                "stdf_upper": 1.1,
            }
        ]
    )

    summary, sources = stats.compute_summary(measurements, limits)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["File"] == "lot1"
    assert row["Test Name"] == "T1"
    assert row["Unit"] == "V"
    assert row["COUNT"] == 2
    assert sources[("lot1", "T1", "1")] == {"lower": "what_if", "upper": "what_if"}


def test_compute_summary_all_nan_returns_empty() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot", "test_name": "T", "test_number": "1", "value": float("nan"), "units": "V"},
        ]
    )
    limits = pd.DataFrame()
    summary, sources = stats.compute_summary(measurements, limits)
    assert summary.empty
    assert sources == {}


def test_compute_yield_pareto_handles_all_pass_and_all_fail() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "test_name": "T1", "test_number": "1", "value": 1.0, "units": "V", "part_status": "PASS"},
            {"file": "lot1", "device_id": "D2", "test_name": "T1", "test_number": "1", "value": 1.1, "units": "V", "part_status": "PASS"},
            {"file": "lot2", "device_id": "D3", "test_name": "T1", "test_number": "1", "value": 2.0, "units": "V", "part_status": "FAIL"},
            {"file": "lot2", "device_id": "D3", "test_name": "T2", "test_number": "2", "value": 5.0, "units": "V", "part_status": "FAIL"},
        ]
    )
    limits = pd.DataFrame(
        [
            {"test_name": "T1", "test_number": "1", "unit": "V", "spec_lower": 0.5, "spec_upper": 1.5},
            {"test_name": "T2", "test_number": "2", "unit": "V", "spec_lower": 0.5, "spec_upper": 1.5},
        ]
    )

    yield_df, pareto_df = stats.compute_yield_pareto(measurements, limits)

    lot1 = yield_df.loc[yield_df["file"] == "lot1"].iloc[0]
    assert lot1["devices_pass"] == 2
    assert lot1["devices_fail"] == 0
    assert lot1["yield_percent"] == pytest.approx(1.0)

    lot2 = yield_df.loc[yield_df["file"] == "lot2"].iloc[0]
    assert lot2["devices_pass"] == 0
    assert lot2["devices_fail"] == 1
    assert lot2["yield_percent"] == pytest.approx(0.0)

    assert not pareto_df.empty
    assert set(pareto_df["test_name"]) == {"T1", "T2"}
