"""Export sub-stage timings must be part of every run record, not only diagnostics.json."""

from uniforge3d_mlx.export_timings import export_timing_breakdown
from studio.server.local_runner import completed_stage


def test_export_breakdown_includes_uv_chart_growth():
    diagnostics = {
        "timings_sec": {"uv": 946.9, "texture_bake": 11.5, "simplify_mesh": 11.4},
        "stages": {
            "uv": {
                "stats": {
                    "timings_sec": {
                        "uv.grow_charts": 917.37,
                        "uv.parameterize": 20.67,
                        "uv.pack": 0.09,
                    }
                }
            }
        },
    }

    breakdown = export_timing_breakdown(diagnostics)

    assert breakdown["uv"] == 946.9
    assert breakdown["uv.grow_charts"] == 917.37
    assert breakdown["texture_bake"] == 11.5


def test_decode_stage_record_keeps_the_export_breakdown():
    entry = completed_stage("decode", 1068.0, {"uv.grow_charts": 917.37, "uv": 946.9})

    assert entry["phase"] == "decode"
    assert entry["seconds"] == 1068.0
    assert entry["export_stages"]["uv.grow_charts"] == 917.37


def test_other_phases_do_not_invent_an_export_breakdown():
    assert "export_stages" not in completed_stage("texture", 379.8, {"uv": 1.0})
