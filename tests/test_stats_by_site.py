from __future__ import annotations

import pandas as pd
import pytest

from cpkanalysis import stats


def _make_measurements() -> pd.DataFrame:
    rows = []
    for site, values in [(1, [1.0, 1.2, 1.1]), (2, [2.6, 2.7, 2.8])]:
        for idx, value in enumerate(values, start=1):
            rows.append(
                {
                    "file": "lotA.stdf",
                    "site": site,
                    "device_id": f"S{site}_{idx}",
                    "test_name": "VDD",
                    "test_number": "1",
                    "units": "V",
                    "value": value,
                    "part_status": "PASS" if value < 2.5 else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def test_compute_summary_by_site_groups_correctly() -> None:
    measurements = _make_measurements()
    limits = pd.DataFrame(
        [
            {"test_name": "VDD", "test_number": "1", "unit": "V", "spec_lower": 0.5, "spec_upper": 2.5},
        ]
    )
    summary, limit_sources = stats.compute_summary_by_site(measurements, limits)
    assert len(summary) == 2
    keys = {tuple(summary.iloc[i][["File", "Site", "Test Name", "Test Number"]]) for i in range(len(summary))}
    assert keys == {
        ("lotA.stdf", 1.0, "VDD", "1"),
        ("lotA.stdf", 2.0, "VDD", "1"),
    }
    assert limit_sources == {
        ("lotA.stdf", 1.0, "VDD", "1"): {"lower": "spec", "upper": "spec"},
        ("lotA.stdf", 2.0, "VDD", "1"): {"lower": "spec", "upper": "spec"},
    }


def test_compute_summary_by_site_empty_without_site_column() -> None:
    measurements = pd.DataFrame(
        [{"file": "lot", "test_name": "T", "test_number": "1", "value": 1.0}],
    )
    limits = pd.DataFrame()
    summary, limit_sources = stats.compute_summary_by_site(measurements, limits)
    assert summary.empty
    assert limit_sources == {}


def test_compute_yield_pareto_by_site_groups_by_file_and_site() -> None:
    measurements = _make_measurements()
    limits = pd.DataFrame(
        [
            {"test_name": "VDD", "test_number": "1", "spec_lower": 0.5, "spec_upper": 2.5},
        ]
    )
    yield_df, pareto_df = stats.compute_yield_pareto_by_site(measurements, limits)
    assert set(zip(yield_df["file"], yield_df["site"])) == {("lotA.stdf", 1.0), ("lotA.stdf", 2.0)}
    # Site 2 should contribute to pareto (values higher)
    assert set(pareto_df["site"]) == {2.0}


def test_compute_yield_pareto_by_site_empty_without_site_column() -> None:
    measurements = pd.DataFrame(
        [{"file": "lot", "test_name": "T", "test_number": "1", "value": 1.0}],
    )
    limits = pd.DataFrame()
    yield_df, pareto_df = stats.compute_yield_pareto_by_site(measurements, limits)
    assert yield_df.empty
    assert pareto_df.empty


def test_compute_summary_by_site_single_device_has_zero_stdev() -> None:
    measurements = pd.DataFrame(
        [
            {
                "file": "lotB.stdf",
                "site": 7,
                "device_id": "S7_1",
                "test_name": "IDDQ",
                "test_number": "200",
                "units": "mA",
                "value": 0.12,
                "part_status": "PASS",
            }
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "test_name": "IDDQ",
                "test_number": "200",
                "unit": "mA",
                "spec_lower": 0.05,
                "spec_upper": 0.2,
            }
        ]
    )
    summary, limit_sources = stats.compute_summary_by_site(measurements, limits)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["File"] == "lotB.stdf"
    assert row["Site"] == pytest.approx(7.0)
    assert row["COUNT"] == 1
    assert row["MEAN"] == pytest.approx(0.12)
    assert row["STDEV"] == 0.0
    limit_key = ("lotB.stdf", 7.0, "IDDQ", "200")
    assert limit_sources[limit_key] == {"lower": "spec", "upper": "spec"}


def test_compute_yield_pareto_by_site_accepts_string_sites() -> None:
    measurements = pd.DataFrame(
        [
            {
                "file": "lotC.stdf",
                "site": "A",
                "device_id": "A1",
                "test_name": "VTH",
                "test_number": "10",
                "units": "V",
                "value": 1.0,
                "part_status": "PASS",
            },
            {
                "file": "lotC.stdf",
                "site": "B",
                "device_id": "B1",
                "test_name": "VTH",
                "test_number": "10",
                "units": "V",
                "value": 2.5,
                "part_status": "FAIL",
            },
        ]
    )
    limits = pd.DataFrame(
        [
            {
                "test_name": "VTH",
                "test_number": "10",
                "spec_lower": 0.5,
                "spec_upper": 2.0,
            }
        ]
    )
    yield_df, pareto_df = stats.compute_yield_pareto_by_site(measurements, limits)
    assert set(yield_df["site"]) == {"A", "B"}
    assert set(pareto_df["site"]) == {"B"}
