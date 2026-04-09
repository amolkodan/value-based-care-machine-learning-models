from __future__ import annotations

import pandas as pd
import pytest

from carevalue_claims_ml.journey_signals import (
    diagnosis_morbidity_breadth_by_member,
    distinct_ndc_count_by_member,
    merge_medical_and_pharmacy_claims,
    monthly_utilization_features,
    procedure_intensity_by_member,
)


def test_merge_medical_pharmacy_aligns_columns():
    med = pd.DataFrame(
        {
            "member_id": [1, 1],
            "service_date": ["2024-01-01", "2024-01-15"],
            "allowed_amount": [500.0, 200.0],
            "diagnosis_code": ["M54.5", "M54.5"],
        }
    )
    rx = pd.DataFrame(
        {
            "member_id": [1],
            "service_date": ["2024-01-10"],
            "allowed_amount": [30.0],
        }
    )
    out = merge_medical_and_pharmacy_claims(med, rx)
    assert len(out) == 3
    assert set(out["care_domain"]) == {"clinical_procedural", "pharmacy"}
    assert "diagnosis_code" in out.columns


def test_monthly_utilization_with_domain():
    df = pd.DataFrame(
        {
            "member_id": [1, 1, 1],
            "service_date": ["2024-01-05", "2024-01-20", "2024-02-01"],
            "allowed_amount": [10.0, 20.0, 15.0],
            "care_domain": ["clinical_procedural", "pharmacy", "clinical_procedural"],
        }
    )
    feat = monthly_utilization_features(df)
    assert "care_domain" in feat.columns or "claim_line_volume" in feat.columns
    assert len(feat) >= 2


def test_distinct_ndc():
    rx = pd.DataFrame({"member_id": [1, 1, 2], "ndc": ["111", "222", "111"]})
    d = distinct_ndc_count_by_member(rx)
    assert dict(zip(d["member_id"], d["distinct_ndc_count"], strict=True)) == {1: 2, 2: 1}


def test_procedure_and_diagnosis_breadth():
    claims = pd.DataFrame({"member_id": [1, 1], "procedure_code": ["99213", "99214"]})
    dx = pd.DataFrame({"member_id": [1, 1], "diagnosis_code": ["E11.9", "I10"]})
    p = procedure_intensity_by_member(claims)
    d = diagnosis_morbidity_breadth_by_member(dx)
    assert p.iloc[0]["distinct_procedure_count"] == 2
    assert d.iloc[0]["distinct_diagnosis_count"] == 2


def test_merge_pharmacy_none():
    med = pd.DataFrame(
        {"member_id": [1], "service_date": ["2024-01-01"], "allowed_amount": [1.0]}
    )
    out = merge_medical_and_pharmacy_claims(med, None)
    assert len(out) == 1
    assert (out["care_domain"] == "clinical_procedural").all()


def test_merge_rejects_bad_medical():
    with pytest.raises(ValueError, match="missing columns"):
        merge_medical_and_pharmacy_claims(pd.DataFrame({"member_id": [1]}), None)
