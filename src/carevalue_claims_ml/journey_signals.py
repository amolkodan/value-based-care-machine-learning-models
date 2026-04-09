from __future__ import annotations

import pandas as pd


def merge_medical_and_pharmacy_claims(
    medical_claims: pd.DataFrame,
    pharmacy_claims: pd.DataFrame | None,
    *,
    member_col: str = "member_id",
    date_col: str = "service_date",
    amount_col: str = "allowed_amount",
    medical_domain_label: str = "clinical_procedural",
    pharmacy_domain_label: str = "pharmacy",
) -> pd.DataFrame:
    """
    Union medical (professional / institutional) and pharmacy claim lines into one
    longitudinal frame for patient-journey modeling. Adds ``care_domain`` for
    modality-aware features and episode builders.
    """
    required_med = {member_col, date_col, amount_col}
    missing = sorted(required_med.difference(medical_claims.columns))
    if missing:
        raise ValueError(f"medical_claims missing columns: {missing}")

    med = medical_claims.copy()
    med[date_col] = pd.to_datetime(med[date_col])
    med["care_domain"] = medical_domain_label

    if pharmacy_claims is None or pharmacy_claims.empty:
        return med

    missing_rx = sorted(required_med.difference(pharmacy_claims.columns))
    if missing_rx:
        raise ValueError(f"pharmacy_claims missing columns: {missing_rx}")

    rx = pharmacy_claims.copy()
    rx[date_col] = pd.to_datetime(rx[date_col])
    rx["care_domain"] = pharmacy_domain_label

    union_cols = sorted(set(med.columns) | set(rx.columns))
    for col in union_cols:
        if col not in med.columns:
            med[col] = pd.NA
        if col not in rx.columns:
            rx[col] = pd.NA
    aligned_med = med[union_cols]
    aligned_rx = rx[union_cols]
    return pd.concat([aligned_med, aligned_rx], ignore_index=True).sort_values(
        [member_col, date_col], kind="mergesort"
    )


def monthly_utilization_features(
    claims_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    date_col: str = "service_date",
    amount_col: str = "allowed_amount",
    care_domain_col: str | None = "care_domain",
) -> pd.DataFrame:
    """
    Member-month aggregates: claim-line volume and allowed spend — inputs for
    trend, seasonality, and utilization-acceleration pattern detection.
    """
    required = {member_col, date_col, amount_col}
    missing = sorted(required.difference(claims_df.columns))
    if missing:
        raise ValueError(f"claims_df missing columns: {missing}")

    frame = claims_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["year_month"] = frame[date_col].dt.to_period("M").astype(str)
    group_keys: list[str] = [member_col, "year_month"]
    if care_domain_col and care_domain_col in frame.columns:
        group_keys.insert(1, care_domain_col)

    out = (
        frame.groupby(group_keys, as_index=False)
        .agg(
            claim_line_volume=(amount_col, "size"),
            allowed_spend=(amount_col, "sum"),
        )
        .rename(columns={"year_month": "period_month"})
    )
    return out


def distinct_ndc_count_by_member(
    pharmacy_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    ndc_col: str = "ndc",
) -> pd.DataFrame:
    """
    Distinct NDC count per member — a pharmacy complexity / polypharmacy proxy
    for risk and medication-therapy management signals.
    """
    required = {member_col, ndc_col}
    missing = sorted(required.difference(pharmacy_df.columns))
    if missing:
        raise ValueError(f"pharmacy_df missing columns: {missing}")

    return (
        pharmacy_df.groupby(member_col, as_index=False)[ndc_col]
        .nunique()
        .rename(columns={ndc_col: "distinct_ndc_count"})
    )


def procedure_intensity_by_member(
    claims_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    procedure_col: str = "procedure_code",
) -> pd.DataFrame:
    """
    Distinct procedure (CPT/HCPCS) count per member — procedural intensity signal
    for surgical bundles and specialty episodic patterns.
    """
    required = {member_col, procedure_col}
    missing = sorted(required.difference(claims_df.columns))
    if missing:
        raise ValueError(f"claims_df missing columns: {missing}")

    return (
        claims_df.groupby(member_col, as_index=False)[procedure_col]
        .nunique()
        .rename(columns={procedure_col: "distinct_procedure_count"})
    )


def diagnosis_morbidity_breadth_by_member(
    diagnosis_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    diagnosis_col: str = "diagnosis_code",
) -> pd.DataFrame:
    """
    Distinct ICD-10 count per member from diagnosis rows — comorbidity / condition
    burden signal for HCC-adjacent analytics and episode stratification.
    """
    required = {member_col, diagnosis_col}
    missing = sorted(required.difference(diagnosis_df.columns))
    if missing:
        raise ValueError(f"diagnosis_df missing columns: {missing}")

    return (
        diagnosis_df.groupby(member_col, as_index=False)[diagnosis_col]
        .nunique()
        .rename(columns={diagnosis_col: "distinct_diagnosis_count"})
    )
