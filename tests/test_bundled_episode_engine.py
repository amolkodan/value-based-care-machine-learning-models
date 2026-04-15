from __future__ import annotations

import pandas as pd
import pytest

from carevalue_claims_ml.bundled_episode_engine import (
    BundledEpisodeAttributionModel,
    EpisodeCodeDefinitions,
    claim_row_to_feature_text,
    fit_bundled_episode_attribution_model,
    learn_episode_definitions_from_labels,
    materialize_gap_episodes_per_family,
    training_frame_from_gap_bundles,
)


def test_claim_row_to_feature_text_codes():
    row = pd.Series(
        {
            "care_domain": "clinical_procedural",
            "diagnosis_code": "M17.11",
            "procedure_code": "27447",
            "ndc": "12345-678-90",
            "allowed_amount": 100.0,
        }
    )
    text = claim_row_to_feature_text(row)
    assert "icd:M17.11" in text
    assert "icdp:M17" in text
    assert "proc:27447" in text
    assert "ndc:" in text
    assert "dom:clinical_procedural" in text


def test_learn_episode_definitions_from_labels():
    df = pd.DataFrame(
        {
            "episode_family": ["orthopedic", "orthopedic", "cardiac"],
            "diagnosis_code": ["M17.11", "M17.12", "I25.10"],
            "procedure_code": ["27447", "27447", "92928"],
            "ndc": [None, None, "55555123401"],
        }
    )
    defs = learn_episode_definitions_from_labels(df, episode_family_col="episode_family", min_support=1)
    assert "orthopedic" in defs.by_episode_family
    assert defs.by_episode_family["orthopedic"]["labeled_row_count"] == 2


def test_fit_predict_multi_and_materialize(tmp_path):
    train = pd.DataFrame(
        {
            "member_id": [1, 1, 1, 2, 2],
            "service_date": pd.to_datetime(
                ["2024-01-01", "2024-01-10", "2024-06-01", "2024-02-01", "2024-02-05"]
            ),
            "allowed_amount": [100.0, 50.0, 200.0, 80.0, 90.0],
            "care_domain": ["clinical_procedural"] * 5,
            "diagnosis_code": ["M17.11", "M17.11", "I25.10", "I25.10", "I25.10"],
            "procedure_code": ["27447", "99213", "92928", "92928", "99214"],
            "ndc": [None, None, None, None, None],
            "rendering_npi": ["111", "111", "222", "222", "333"],
            "episode_labels": ["orthopedic", "orthopedic", "cardiac", "cardiac", "cardiac"],
        }
    )
    model = fit_bundled_episode_attribution_model(
        train, episode_labels_list_col="episode_labels", n_features=256, max_iter=500
    )
    model_path = tmp_path / "ep_attr.joblib"
    model.save(model_path)
    loaded = BundledEpisodeAttributionModel.load(model_path)
    infer = train.drop(columns=["episode_labels"])
    multi = loaded.predict_multi_attribution(infer, min_probability=0.01, max_labels_per_row=3)
    assert "claim_row_index" in multi.columns
    assert "episode_family" in multi.columns
    episodes_df, claim_ep = materialize_gap_episodes_per_family(
        infer, multi, window_days=30, min_probability=0.01
    )
    assert len(episodes_df) >= 1
    assert "episode_instance_id" in claim_ep.columns


def test_training_frame_from_gap_bundles():
    claims = pd.DataFrame(
        {
            "member_id": [1, 1, 1],
            "service_date": ["2024-01-01", "2024-01-05", "2024-05-01"],
            "allowed_amount": [10.0, 20.0, 30.0],
        }
    )
    out = training_frame_from_gap_bundles(claims, archetype="general", window_days=90)
    assert "episode_family" in out.columns
    assert "episode_id" in out.columns


def test_episode_definitions_json_roundtrip(tmp_path):
    defs = EpisodeCodeDefinitions(
        by_episode_family={"a": {"icd_prefixes": [{"code_prefix": "M17", "count": 3, "share": 1.0}]}},
        metadata={"k": "v"},
    )
    p = tmp_path / "d.json"
    defs.save(p)
    loaded = EpisodeCodeDefinitions.load(p)
    assert loaded.by_episode_family["a"]["icd_prefixes"][0]["code_prefix"] == "M17"


def test_fit_requires_label_column():
    df = pd.DataFrame({"member_id": [1], "service_date": ["2024-01-01"], "allowed_amount": [1.0]})
    with pytest.raises(ValueError, match="exactly one"):
        fit_bundled_episode_attribution_model(df)
    with pytest.raises(ValueError, match="missing"):
        fit_bundled_episode_attribution_model(df, episode_labels_col="episode_labels")
