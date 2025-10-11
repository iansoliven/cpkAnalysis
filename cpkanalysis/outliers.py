from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd

from .models import OutlierMethod

GROUP_KEYS = ["file", "test_name", "test_number"]


def apply_outlier_filter(frame: pd.DataFrame, method: OutlierMethod, k: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the requested outlier filter to the measurements."""
    if frame.empty or method == "none" or k <= 0:
        return frame.copy(), {"method": "none", "k": 0, "removed": 0}

    filtered_groups: list[pd.DataFrame] = []
    removed = 0

    for _, group in frame.groupby(GROUP_KEYS, dropna=False, sort=False):
        values = pd.to_numeric(group["value"], errors="coerce")
        if values.empty:
            filtered_groups.append(group)
            continue
        if method == "iqr":
            q1 = np.nanpercentile(values, 25)
            q3 = np.nanpercentile(values, 75)
            iqr = q3 - q1
            if not math.isfinite(iqr) or iqr <= 0:
                filtered_groups.append(group)
                continue
            lower = q1 - k * iqr
            upper = q3 + k * iqr
        else:  # stdev
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1))
            if not math.isfinite(std) or std <= 0:
                filtered_groups.append(group)
                continue
            lower = mean - k * std
            upper = mean + k * std

        # Create mask: keep values within bounds OR NaN/Inf (preserve non-finite values)
        # NaN and Inf values have already passed ingestion validation and should be preserved
        is_finite = np.isfinite(values)
        is_within_bounds = (values >= lower) & (values <= upper)
        mask = is_within_bounds | ~is_finite

        kept = group.loc[mask]
        removed += int(len(group) - len(kept))
        filtered_groups.append(kept)

    filtered = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else frame.iloc[0:0]
    summary = {"method": method, "k": k, "removed": removed}
    return filtered, summary
