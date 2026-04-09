"""Smoke tests: VBC Intelligence OS namespaces must import after pip install."""

from __future__ import annotations

import pandas as pd
import pytest


def test_vbc_intel_packages_import():
    import vbc_intel_benchmarks  # noqa: PLC0415
    import vbc_intel_careops  # noqa: PLC0415
    import vbc_intel_core  # noqa: PLC0415
    import vbc_intel_episodes  # noqa: PLC0415
    import vbc_intel_policy  # noqa: PLC0415

    assert hasattr(vbc_intel_core, "train_model_suite")
    assert hasattr(vbc_intel_core, "merge_medical_and_pharmacy_claims")
    assert hasattr(vbc_intel_episodes, "build_bundled_episodes")
    assert hasattr(vbc_intel_episodes, "EPISODE_ARCHETYPES")
    assert hasattr(vbc_intel_policy, "simulate_policy")
    assert hasattr(vbc_intel_benchmarks, "calculate_pmpm")
    assert hasattr(vbc_intel_careops, "run_agentic_pipeline")


def test_episodes_pipeline_minimal():
    from vbc_intel_episodes import build_bundled_episodes, score_episode_risk  # noqa: PLC0415

    df = pd.DataFrame(
        {
            "member_id": [1, 1],
            "service_date": ["2024-01-01", "2024-02-01"],
            "allowed_amount": [100.0, 200.0],
        }
    )
    episodes = build_bundled_episodes(df)
    scored = score_episode_risk(episodes)
    assert len(episodes) == 2
    assert len(scored) >= 1


def test_episodes_missing_columns():
    from vbc_intel_episodes import build_bundled_episodes  # noqa: PLC0415

    bad = pd.DataFrame({"member_id": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_bundled_episodes(bad)
