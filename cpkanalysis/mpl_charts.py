"""Matplotlib-based chart renderers used when building the CPK workbook.

All helpers here return raw PNG bytes that can be embedded into Excel via
openpyxl.  The module assumes Matplotlib's Agg backend is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import numpy as np

DEFAULT_FIGSIZE = (8, 4)  # Reduced from (9, 4.5) for faster rendering
DEFAULT_DPI = 100  # Reduced from 140 for faster generation
HIST_COLOR = "#1f77b4"
CDF_COLOR = "#ff7f0e"
TIME_SERIES_COLOR = "#2ca02c"
LIMIT_LOW_COLOR = "#C0504D"
LIMIT_HIGH_COLOR = "#9BBB59"

MarkerOrientation = Literal["vertical", "horizontal"]


@dataclass(frozen=True)
class ChartMarker:
    label: str
    value: float
    orientation: MarkerOrientation = "vertical"
    color: str = LIMIT_LOW_COLOR
    linestyle: str = "--"
    linewidth: float = 1.5
    alpha: float = 1.0


def render_histogram(
    values: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    test_label: str = "",
    cpk: Optional[float] = None,
    unit_label: str = "",
    extra_markers: Optional[Sequence[ChartMarker]] = None,
    title_font_size: int = 11,
    cpk_font_size: int = 9,
    cpk_position: Optional[Tuple[float, float]] = None,
) -> bytes:
    """Render a histogram PNG for the supplied measurements."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        clean = np.array([0.0])

    bins = _freedman_diaconis_bins(clean)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    # Adaptive styling based on data size and bin count to improve visibility
    if clean.size <= 10:
        # Very small dataset - use step histogram for better visibility
        counts, bin_edges = np.histogram(clean, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.step(bin_centers, counts, where='mid', color=HIST_COLOR, linewidth=2.5, alpha=1.0)  # Max intensity
        ax.fill_between(bin_centers, counts, step='mid', color=HIST_COLOR, alpha=0.4)  # Slightly higher fill
    elif bins <= 10:
        # For few bins (thin bars), use maximum opacity and thicker edges
        alpha = 1.0  # Maximum intensity for best visibility
        edgecolor = "darkblue"
        linewidth = 1.2
        ax.hist(clean, bins=bins, color=HIST_COLOR, alpha=alpha, 
                edgecolor=edgecolor, linewidth=linewidth)
    elif bins <= 20:
        # Medium number of bins
        alpha = 0.85  # Slightly higher than before for better visibility
        edgecolor = "white" 
        linewidth = 0.8
        ax.hist(clean, bins=bins, color=HIST_COLOR, alpha=alpha,
                edgecolor=edgecolor, linewidth=linewidth)
    else:
        # Many bins (normal case)
        alpha = 0.75
        edgecolor = "white"
        linewidth = 0.5
        ax.hist(clean, bins=bins, color=HIST_COLOR, alpha=alpha,
                edgecolor=edgecolor, linewidth=linewidth)
    
    ax.set_xlabel("Measurement Value", fontsize=9)  # Smaller font for smaller chart
    ax.set_ylabel("Count", fontsize=9)  # Smaller font for smaller chart
    ax.tick_params(labelsize=8)  # Smaller tick labels
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    markers = list(extra_markers or [])
    if lower_limit is not None and np.isfinite(lower_limit):
        markers.append(ChartMarker("Lower Limit", float(lower_limit), "vertical", LIMIT_LOW_COLOR))
    if upper_limit is not None and np.isfinite(upper_limit):
        markers.append(ChartMarker("Upper Limit", float(upper_limit), "vertical", LIMIT_HIGH_COLOR))
    _draw_markers(ax, markers)

    if x_range:
        xmin, xmax = x_range
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)

    _finalize_chart(
        fig,
        ax,
        test_label,
        cpk,
        unit_label,
        title_font_size=title_font_size,
        cpk_font_size=cpk_font_size,
        cpk_position=cpk_position,
    )
    fig.tight_layout(rect=(0, 0.2, 0.75, 1))
    return _figure_to_png(fig)


def render_cdf(
    values: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    test_label: str = "",
    cpk: Optional[float] = None,
    unit_label: str = "",
    extra_markers: Optional[Sequence[ChartMarker]] = None,
    title_font_size: int = 11,
    cpk_font_size: int = 9,
    cpk_position: Optional[Tuple[float, float]] = None,
) -> bytes:
    """Render a cumulative distribution plot as PNG bytes."""
    clean = np.sort(values[np.isfinite(values)])
    if clean.size == 0:
        clean = np.array([0.0])
    cumulative = np.linspace(0, 1, clean.size, endpoint=True)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(clean, cumulative, color=CDF_COLOR, linewidth=2.0)
    ax.set_xlabel("Measurement Value", fontsize=9)  # Smaller font for smaller chart
    ax.set_ylabel("Cumulative Probability", fontsize=9)  # Smaller font for smaller chart
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)  # Smaller tick labels
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)

    markers = list(extra_markers or [])
    if lower_limit is not None and np.isfinite(lower_limit):
        markers.append(ChartMarker("Lower Limit", float(lower_limit), "vertical", LIMIT_LOW_COLOR))
    if upper_limit is not None and np.isfinite(upper_limit):
        markers.append(ChartMarker("Upper Limit", float(upper_limit), "vertical", LIMIT_HIGH_COLOR))
    _draw_markers(ax, markers)

    if x_range:
        xmin, xmax = x_range
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)

    _finalize_chart(
        fig,
        ax,
        test_label,
        cpk,
        unit_label,
        title_font_size=title_font_size,
        cpk_font_size=cpk_font_size,
        cpk_position=cpk_position,
    )
    fig.tight_layout(rect=(0, 0.2, 0.75, 1))
    return _figure_to_png(fig)


def render_time_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    y_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    test_label: str = "",
    cpk: Optional[float] = None,
    unit_label: str = "",
    x_label: str = "Timestamp / Index",
    extra_markers: Optional[Sequence[ChartMarker]] = None,
    title_font_size: int = 11,
    cpk_font_size: int = 9,
    cpk_position: Optional[Tuple[float, float]] = None,
) -> bytes:
    """Render a time-series trend chart for the given measurements."""
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        x = np.arange(len(y))
        y = np.zeros_like(x, dtype=float)
    else:
        x = x[mask]
        y = y[mask]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(x, y, color=TIME_SERIES_COLOR, linewidth=1.6, marker="o", markersize=3)  # Smaller markers
    ax.set_xlabel(x_label, fontsize=9)  # Smaller font for smaller chart
    ax.set_ylabel("Measurement Value", fontsize=9)  # Smaller font for smaller chart
    ax.tick_params(labelsize=8)  # Smaller tick labels
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)

    markers = list(extra_markers or [])
    if lower_limit is not None and np.isfinite(lower_limit):
        markers.append(ChartMarker("Lower Limit", float(lower_limit), "horizontal", LIMIT_LOW_COLOR))
    if upper_limit is not None and np.isfinite(upper_limit):
        markers.append(ChartMarker("Upper Limit", float(upper_limit), "horizontal", LIMIT_HIGH_COLOR))
    _draw_markers(ax, markers)
    if y_range:
        ymin, ymax = y_range
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

    _finalize_chart(
        fig,
        ax,
        test_label,
        cpk,
        unit_label,
        title_font_size=title_font_size,
        cpk_font_size=cpk_font_size,
        cpk_position=cpk_position,
    )
    fig.tight_layout(rect=(0, 0.2, 0.75, 1))
    return _figure_to_png(fig)


def _freedman_diaconis_bins(values: np.ndarray) -> int:
    """Return a bin count using the Freedman–Diaconis rule with safe fallbacks."""
    clean = values[np.isfinite(values)]
    n = clean.size
    if n <= 1:
        return 1
    q1 = np.percentile(clean, 25)
    q3 = np.percentile(clean, 75)
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        return max(int(np.ceil(np.sqrt(n))), 1)
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width <= 0 or not np.isfinite(bin_width):
        return max(int(np.ceil(np.sqrt(n))), 1)
    data_range = clean.max() - clean.min()
    if data_range <= 0 or not np.isfinite(data_range):
        return 1
    bins = int(np.ceil(data_range / bin_width))
    
    # Ensure reasonable bin count for visibility (avoid too few or too many bins)
    bins = max(bins, 5)   # Minimum 5 bins for better visualization
    bins = min(bins, 50)  # Maximum 50 bins to avoid overcrowding
    
    return bins


def _draw_markers(ax, markers: Sequence[ChartMarker]) -> None:
    seen_labels = set()
    for marker in markers:
        if marker.value is None or not np.isfinite(marker.value):
            continue
        label = marker.label if marker.label not in seen_labels else "_nolegend_"
        if marker.label not in seen_labels:
            seen_labels.add(marker.label)
        if marker.orientation == "horizontal":
            ax.axhline(
                marker.value,
                color=marker.color,
                linestyle=marker.linestyle,
                linewidth=marker.linewidth,
                alpha=marker.alpha,
                label=label,
            )
        else:
            ax.axvline(
                marker.value,
                color=marker.color,
                linestyle=marker.linestyle,
                linewidth=marker.linewidth,
                alpha=marker.alpha,
                label=label,
            )


def _finalize_chart(
    fig,
    ax,
    test_label: str,
    cpk: Optional[float],
    unit_label: str,
    *,
    title_font_size: int = 11,
    cpk_font_size: int = 9,
    cpk_position: Optional[Tuple[float, float]] = None,
) -> None:
    test_label = _clean_text(test_label)
    unit_label = _clean_text(unit_label)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
        )
    if test_label:
        ax.set_title(test_label, fontsize=title_font_size, fontweight="bold")
    if cpk is not None and math.isfinite(cpk):
        position = cpk_position or (1.02, 0.12)
        ha = "left"
        va = "bottom"
        if position[0] <= 1.0:
            ha = "right" if position[0] >= 0.5 else "left"
        if position[1] >= 0.5:
            va = "top"
        ax.text(
            position[0],
            position[1],
            f"CPK: {cpk:.3f}",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=cpk_font_size,
        )
    if unit_label:
        ax.text(
            0.5,
            -0.22,
            f"Unit: {unit_label}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,  # Reduced from 10
        )
    fig.subplots_adjust(right=0.78, bottom=0.28)


def _figure_to_png(fig) -> bytes:
    buffer = BytesIO()
    # Use only supported matplotlib savefig parameters
    fig.savefig(buffer, format="png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _clean_text(value: str) -> str:
    if not value:
        return ""
    cleaned = []
    for ch in value:
        code = ord(ch)
        if code < 32:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned).strip()
