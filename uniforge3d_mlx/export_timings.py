"""Flatten O-Voxel export diagnostics into the timings every run must keep."""

from __future__ import annotations


def export_timing_breakdown(diagnostics: dict) -> dict[str, float]:
    """Return export stage seconds, including nested UV sub-stages.

    diagnostics.json already stores these. Run records that only keep the
    outer export clock hide the chart-growth cost, so callers copy this dict
    into the run report and the studio phase entry.
    """
    timings: dict[str, float] = {}
    for name, seconds in (diagnostics.get("timings_sec") or {}).items():
        timings[str(name)] = round(float(seconds), 3)
    uv_stats = ((diagnostics.get("stages") or {}).get("uv") or {}).get("stats") or {}
    for name, seconds in (uv_stats.get("timings_sec") or {}).items():
        timings[str(name)] = round(float(seconds), 3)
    return timings
