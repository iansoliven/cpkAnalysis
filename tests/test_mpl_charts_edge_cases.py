from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import mpl_charts


def _is_png(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_histogram_handles_all_nan_values_with_rug() -> None:
    values = np.array([np.nan, np.nan])
    png = mpl_charts.render_histogram(values, rug=True, test_label="All NaN")
    assert _is_png(png)


def test_render_cdf_returns_png_for_empty_input() -> None:
    png = mpl_charts.render_cdf(np.array([]), test_label="Empty")
    assert _is_png(png)


def test_render_time_series_replaces_non_finite_points() -> None:
    x = np.array([np.nan, np.nan])
    y = np.array([np.nan, np.nan])
    png = mpl_charts.render_time_series(x, y, test_label="NaN series")
    assert _is_png(png)


def test_sample_for_rug_limits_max_points() -> None:
    values = np.linspace(0, 1, 10_000)
    sampled = mpl_charts._sample_for_rug(values, max_points=100)
    assert sampled.size == 100
    assert np.all(np.diff(sampled) >= 0)
