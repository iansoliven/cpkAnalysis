from __future__ import annotations

import math
from io import BytesIO
from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def render_histogram(
    values: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    x_range: Optional[tuple[float, float]] = None,
) -> bytes:
    bins = _freedman_diaconis_bins(values)
    hist, edges = np.histogram(values, bins=bins)
    x_points, y_points = _hist_step_points(edges, hist)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_points,
            y=y_points,
            line={"color": "#1f77b4", "width": 3, "shape": "hv"},
            fill="tozeroy",
            name="Histogram",
            hoverinfo="x+y",
        )
    )
    peak = max(y_points) if y_points else 0
    _add_limit_shapes(fig, lower_limit, upper_limit, peak)
    fig.update_layout(
        template="plotly_white",
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        margin={"l": 60, "r": 30, "t": 40, "b": 60},
        showlegend=False,
        xaxis_title="Measurement Value",
        yaxis_title="Count",
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=[0, peak * 1.05 if peak else 1])
    return _figure_to_png(fig)


def render_cdf(
    values: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    x_range: Optional[tuple[float, float]] = None,
) -> bytes:
    sorted_values = np.sort(values)
    cumulative = np.linspace(0, 1, sorted_values.size, endpoint=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=sorted_values,
            y=cumulative,
            mode="lines",
            line={"color": "#ff7f0e", "width": 3},
            name="CDF",
            hoverinfo="x+y",
        )
    )
    _add_limit_shapes(fig, lower_limit, upper_limit, 1, yref="y")
    fig.update_layout(
        template="plotly_white",
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        margin={"l": 60, "r": 30, "t": 40, "b": 60},
        showlegend=False,
        xaxis_title="Measurement Value",
        yaxis_title="Cumulative Probability",
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=[0, 1])
    return _figure_to_png(fig)


def render_time_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
) -> bytes:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines+markers",
            marker={"size": 6, "color": "#2ca02c"},
            line={"color": "#2ca02c", "width": 2},
            name="Measurement",
            hoverinfo="x+y",
        )
    )
    _add_limit_shapes(fig, lower_limit, upper_limit, float(np.nanmax(y) if y.size else 0), xref="x", yref="y")
    fig.update_layout(
        template="plotly_white",
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        margin={"l": 60, "r": 30, "t": 40, "b": 60},
        showlegend=False,
        xaxis_title="Timestamp / Index",
        yaxis_title="Measurement Value",
    )
    return _figure_to_png(fig)


def _figure_to_png(fig: go.Figure) -> bytes:
    buffer = fig.to_image(format="png", width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, scale=1, engine="kaleido")
    return buffer


def _freedman_diaconis_bins(values: np.ndarray) -> int:
    clean = values[np.isfinite(values)]
    n = clean.size
    if n <= 1:
        return 1
    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    iqr = q3 - q1
    if iqr <= 0 or not math.isfinite(iqr):
        return max(int(math.ceil(math.sqrt(n))), 1)
    bin_width = 2 * iqr / (n ** (1 / 3))
    if bin_width <= 0:
        return max(int(math.ceil(math.sqrt(n))), 1)
    data_range = clean.max() - clean.min()
    if data_range == 0:
        return 1
    bins = int(math.ceil(data_range / bin_width))
    return max(bins, 1)


def _hist_step_points(edges: np.ndarray, counts: np.ndarray) -> tuple[list[float], list[float]]:
    x_points: list[float] = []
    y_points: list[float] = []
    if not len(counts):
        return x_points, y_points
    x_points.append(float(edges[0]))
    y_points.append(0.0)
    for idx, count in enumerate(counts):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        x_points.extend([left, right])
        y_points.extend([float(count), float(count)])
    x_points.append(float(edges[-1]))
    y_points.append(0.0)
    return x_points, y_points


def _add_limit_shapes(fig: go.Figure, lower: Optional[float], upper: Optional[float], height: float, *, xref: str = "x", yref: str = "paper") -> None:
    if lower is not None and math.isfinite(lower):
        fig.add_shape(
            type="line",
            x0=lower,
            x1=lower,
            y0=0,
            y1=1 if yref == "paper" else height,
            xref=xref,
            yref=yref,
            line={"color": "#d62728", "dash": "dash", "width": 2},
            name="Lower Limit",
        )
    if upper is not None and math.isfinite(upper):
        fig.add_shape(
            type="line",
            x0=upper,
            x1=upper,
            y0=0,
            y1=1 if yref == "paper" else height,
            xref=xref,
            yref=yref,
            line={"color": "#9467bd", "dash": "dash", "width": 2},
            name="Upper Limit",
        )
