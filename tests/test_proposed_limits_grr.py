from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math

from cpkanalysis.postprocess.proposed_limits_grr import compute_proposed_limits


def test_compute_proposed_limits_prefers_half_guardband_when_full_cpk_too_low():
    result = compute_proposed_limits(
        mean=0.008708,
        stdev=0.002884,
        spec_lower=-0.01,
        spec_upper=0.1,
        guardband_full=0.017527215293359422,
        cpk_min=2.0,
        cpk_max=10.0,
    )

    assert result.guardband_label == "50% GRR"
    assert math.isclose(result.guardband_value, 0.008763607646679711, rel_tol=1e-12)
    assert math.isclose(result.ft_cpk, 2.0)
    assert result.spec_lower < -0.01
    assert result.spec_widened