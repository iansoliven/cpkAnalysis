from __future__ import annotations

import pandas as pd
import pytest

from cpkanalysis import stats


def test_compute_yield_pareto_basic() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "part_status": "PASS", "test_name": "T1", "test_number": "1", "value": 0.5},
            {"file": "lot1", "device_id": "D2", "part_status": "FAIL", "test_name": "T1", "test_number": "1", "value": 1.5},
            {"file": "lot1", "device_id": "D2", "part_status": "FAIL", "test_name": "T2", "test_number": "2", "value": 0.1},
            {"file": "lot1", "device_id": "D3", "part_status": "PASS", "test_name": "T1", "test_number": "1", "value": 0.6},
            {"file": "lot2", "device_id": "X1", "part_status": "PASS", "test_name": "T1", "test_number": "1", "value": 0.4},
        ]
    )

    limits = pd.DataFrame(
        [
            {
                "test_name": "T1",
                "test_number": "1",
                "unit": "V",
                "stdf_lower": 0.0,
                "stdf_upper": 1.0,
                "spec_lower": None,
                "spec_upper": None,
                "what_if_lower": None,
                "what_if_upper": None,
            },
            {
                "test_name": "T2",
                "test_number": "2",
                "unit": "V",
                "stdf_lower": 0.2,
                "stdf_upper": 1.0,
                "spec_lower": None,
                "spec_upper": None,
                "what_if_lower": None,
                "what_if_upper": None,
            },
        ]
    )

    yield_df, pareto_df = stats.compute_yield_pareto(measurements, limits)

    assert list(yield_df["file"]) == ["lot1", "lot2"]

    lot1_row = yield_df[yield_df["file"] == "lot1"].iloc[0]
    assert lot1_row["devices_total"] == 3
    assert lot1_row["devices_pass"] == 2
    assert lot1_row["devices_fail"] == 1
    assert lot1_row["yield_percent"] == pytest.approx(2 / 3, rel=1e-3)

    lot2_row = yield_df[yield_df["file"] == "lot2"].iloc[0]
    assert lot2_row["devices_total"] == 1
    assert lot2_row["devices_pass"] == 1
    assert lot2_row["devices_fail"] == 0
    assert lot2_row["yield_percent"] == pytest.approx(1.0)

    lot1_pareto = pareto_df[pareto_df["file"] == "lot1"].reset_index(drop=True)
    assert len(lot1_pareto) == 2
    assert list(lot1_pareto["devices_fail"]) == [1, 1]
    assert list(lot1_pareto["test_name"]) == ["T1", "T2"]
    assert lot1_pareto.loc[0, "fail_rate_percent"] == pytest.approx(1 / 3, rel=1e-3)
    assert lot1_pareto.loc[1, "cumulative_percent"] == pytest.approx(2 / 3, rel=1e-3)

    assert pareto_df[pareto_df["file"] == "lot2"].empty


def test_compute_yield_pareto_without_limits() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "part_status": "PASS", "test_name": "T1", "test_number": "1", "value": 0.3},
            {"file": "lot1", "device_id": "D2", "part_status": "FAIL", "test_name": "T1", "test_number": "1", "value": 0.4},
        ]
    )

    limits = pd.DataFrame()

    yield_df, pareto_df = stats.compute_yield_pareto(measurements, limits)

    assert not yield_df.empty
    assert not pareto_df.empty
    assert pareto_df.iloc[0]["test_name"] == "T1"
    assert pareto_df.iloc[0]["devices_fail"] == 1
    assert pareto_df.iloc[0]["fail_rate_percent"] == pytest.approx(0.5)


def test_pareto_respects_pass_status_even_with_limit_violation() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "part_status": "PASS", "test_name": "T1", "test_number": "1", "value": 1.5},
        ]
    )
    limits = pd.DataFrame([{"test_name": "T1", "test_number": "1", "spec_upper": 1.0}])

    yield_df, pareto_df = stats.compute_yield_pareto(measurements, limits)

    assert yield_df.iloc[0]["yield_percent"] == pytest.approx(1.0)
    assert pareto_df.empty


def test_pareto_first_failure_only_keeps_first_test_per_device() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "part_status": "FAIL", "test_name": "T1", "test_number": "1", "value": 2.0},
            {"file": "lot1", "device_id": "D1", "part_status": "FAIL", "test_name": "T2", "test_number": "2", "value": 3.0},
        ]
    )
    limits = pd.DataFrame(
        [
            {"test_name": "T1", "test_number": "1", "spec_upper": 1.0},
            {"test_name": "T2", "test_number": "2", "spec_upper": 2.0},
        ]
    )

    _, pareto_df = stats.compute_yield_pareto(measurements, limits, first_failure_only=True)
    assert list(pareto_df["test_name"]) == ["T1"]


def test_pareto_uses_file_specific_limits() -> None:
    measurements = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "A1", "test_name": "T1", "test_number": "1", "value": 2.0},
            {"file": "lot2", "device_id": "B1", "test_name": "T1", "test_number": "1", "value": 2.0},
        ]
    )
    limits = pd.DataFrame(
        [
            {"file": "lot1", "test_name": "T1", "test_number": "1", "spec_upper": 1.0},
            {"file": "lot2", "test_name": "T1", "test_number": "1", "spec_upper": 3.0},
        ]
    )

    yield_df, pareto_df = stats.compute_yield_pareto(measurements, limits)

    lot1_yield = yield_df[yield_df["file"] == "lot1"].iloc[0]
    lot2_yield = yield_df[yield_df["file"] == "lot2"].iloc[0]
    assert lot1_yield["devices_fail"] == 1
    assert lot2_yield["devices_fail"] == 0

    assert set(pareto_df["file"]) == {"lot1"}
