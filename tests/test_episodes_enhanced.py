from __future__ import annotations

import pandas as pd
import pytest

from carevalue_claims_ml.episodes import build_bundled_episodes, score_episode_risk


def test_invalid_archetype():
    df = pd.DataFrame(
        {
            "member_id": [1],
            "service_date": ["2024-01-01"],
            "allowed_amount": [1.0],
        }
    )
    with pytest.raises(ValueError, match="archetype"):
        build_bundled_episodes(df, archetype="invalid_type")


def test_score_episode_with_icd_cpt_breadth():
    df = pd.DataFrame(
        {
            "episode_id": ["e1", "e1"],
            "member_id": [1, 1],
            "service_date": ["2024-01-01", "2024-01-02"],
            "allowed_amount": [100.0, 200.0],
            "diagnosis_code": ["M17.11", "M17.12"],
            "procedure_code": ["27447", "27447"],
        }
    )
    scored = score_episode_risk(
        df, diagnosis_code_col="diagnosis_code", procedure_code_col="procedure_code"
    )
    assert len(scored) == 1
    assert scored.iloc[0]["clinical_condition_breadth"] == 2
    assert scored.iloc[0]["procedural_intensity_breadth"] == 1
    assert "episode_severity_percentile" in scored.columns
    assert "episode_financial_intensity" in scored.columns
