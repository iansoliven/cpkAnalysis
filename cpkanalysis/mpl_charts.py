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
from matplotlib.ticker import MaxNLocator, PercentFormatter

DEFAULT_FIGSIZE = (8, 4)  # Reduced from (9, 4.5) for faster rendering
DEFAULT_DPI = 65  # Lower DPI for faster rendering with acceptable clarity
RUG_MAX_POINTS = 5000
RUG_COLOR = "#404040"
RUG_ALPHA = 0.45
HIST_COLOR = "#1f77b4"
CDF_COLOR = "#ff7f0e"
TIME_SERIES_COLOR = "#2ca02c"
LIMIT_LOW_COLOR = "#C0504D"
LIMIT_HIGH_COLOR = "#9BBB59"
YIELD_PASS_COLOR = "#2ca02c"
YIELD_FAIL_COLOR = "#d62728"
PARETO_BAR_COLOR = HIST_COLOR
PARETO_LINE_COLOR = CDF_COLOR

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


def _sample_for_rug(values: np.ndarray, max_points: int = RUG_MAX_POINTS) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    sorted_vals = np.sort(values)
    if sorted_vals.size <= max_points:
        return sorted_vals
    indices = np.linspace(0, sorted_vals.size - 1, max_points, dtype=int)
    return sorted_vals[indices]


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
    rug: bool = False,
) -> bytes:
    """Render a histogram PNG for the supplied measurements."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        clean = np.array([0.0])

    bins = _freedman_diaconis_bins(clean)

    rug_values = _sample_for_rug(clean, RUG_MAX_POINTS) if rug else np.array([], dtype=float)
    use_rug_panel = rug and rug_values.size > 0

    if use_rug_panel:
        fig = plt.figure(figsize=DEFAULT_FIGSIZE)
        gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        rug_ax = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        rug_ax = None
    
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

    if use_rug_panel and rug_ax is not None:
        rug_ax.vlines(
            rug_values,
            0.0,
            0.6,
            colors=RUG_COLOR,
            linewidth=1.0,
            alpha=RUG_ALPHA,
        )
        rug_ax.set_ylim(0, 1)
        rug_ax.set_yticks([])
        rug_ax.set_ylabel("")
        rug_ax.tick_params(axis="y", left=False)
        rug_ax.tick_params(axis="x", labelsize=8)
        rug_ax.spines["left"].set_visible(False)
        rug_ax.spines["right"].set_visible(False)
        rug_ax.spines["top"].set_visible(False)
        rug_ax.grid(False)
        ax.tick_params(axis="x", labelbottom=False)
        rug_ax.set_xlabel("Measurement Value", fontsize=9)

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
    tight_bottom = 0.12 if use_rug_panel else 0.2
    fig.tight_layout(rect=(0, tight_bottom, 0.75, 1))
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


def render_yield_chart(
    pass_count: float,
    fail_count: float,
    *,
    yield_percent: Optional[float] = None,
    title: str = "",
) -> bytes:
    counts = [max(float(pass_count or 0), 0.0), max(float(fail_count or 0), 0.0)]
    labels = ["Pass", "Fail"]
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.bar(labels, counts, color=[YIELD_PASS_COLOR, YIELD_FAIL_COLOR], alpha=0.85)
    ax.set_ylabel("Units", fontsize=9)
    ax.set_title(_clean_text(title), fontsize=11)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    max_count = max(counts) if counts else 0.0
    if max_count <= 0:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, max_count * 1.15)

    padding = max(0.02 * max_count, 0.1)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + padding,
            f"{int(round(count))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if yield_percent is None or not math.isfinite(yield_percent):
        yield_text = "Yield: n/a"
    else:
        yield_text = f"Yield: {yield_percent:.2f}%"
    ax.text(0.5, 0.92, yield_text, transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return _figure_to_png(fig)


def render_pareto_chart(
    labels: Sequence[str],
    counts: Sequence[float],
    cumulative_percent: Sequence[float],
    *,
    title: str = "",
) -> bytes:
    if not counts:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.axis("off")
        ax.text(0.5, 0.5, "No failing tests", ha="center", va="center", fontsize=12)
        return _figure_to_png(fig)

    clean_labels = [_clean_text(label) or "(unnamed)" for label in labels]
    indices = np.arange(len(counts))
    fig, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax1.bar(indices, counts, color=PARETO_BAR_COLOR, alpha=0.85)
    ax1.set_title(_clean_text(title), fontsize=11)
    ax1.set_xlabel("Test", fontsize=9)
    ax1.set_ylabel("Fail Count", fontsize=9)
    ax1.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax1.tick_params(axis="y", labelsize=8)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    max_count = max(counts)
    padding = max(0.02 * max_count, 0.1)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + padding,
            f"{int(round(count))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_xticks(indices)
    ax1.set_xticklabels(clean_labels)

    cumulative = np.clip(np.array(list(cumulative_percent), dtype=float), 0.0, 100.0)
    ax2 = ax1.twinx()
    ax2.plot(indices, cumulative, color=PARETO_LINE_COLOR, linewidth=2.0, marker="o", markersize=4)
    ax2.set_ylabel("Cumulative %", fontsize=9, color=PARETO_LINE_COLOR)
    ax2.tick_params(axis="y", labelsize=8, colors=PARETO_LINE_COLOR)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(PercentFormatter(100))

    fig.tight_layout()
    return _figure_to_png(fig)


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
