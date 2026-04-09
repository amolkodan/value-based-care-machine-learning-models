from __future__ import annotations

import pandas as pd

# Episode archetypes for bundled-payment style analytics (orthopedic MS-DRG families, etc.).
EPISODE_ARCHETYPES = frozenset({"general", "orthopedic", "cardiac", "maternity", "oncology"})


def build_bundled_episodes(
    claims_df: pd.DataFrame,
    archetype: str = "general",
    window_days: int = 90,
    *,
    member_col: str = "member_id",
    date_col: str = "service_date",
    amount_col: str = "allowed_amount",
    care_domain_col: str | None = "care_domain",
) -> pd.DataFrame:
    """
    Build episode groupings from longitudinal claim lines using a gap-based rule
    (default 90 days). Suitable for bundled episode analytics, BPCI-style windows,
    and specialty surgical or medical episode prototypes.

    Optional ``care_domain`` (e.g. clinical_procedural vs pharmacy) is preserved
    for multimodal patient-journey models.
    """
    if archetype not in EPISODE_ARCHETYPES:
        raise ValueError(f"archetype must be one of {sorted(EPISODE_ARCHETYPES)}, got {archetype!r}")

    required = {member_col, date_col, amount_col}
    missing = sorted(required.difference(claims_df.columns))
    if missing:
        raise ValueError(f"claims_df missing required columns: {missing}")

    frame = claims_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([member_col, date_col], kind="mergesort")
    spacing = frame.groupby(member_col)[date_col].diff().dt.days.fillna(window_days + 1)
    frame["episode_open"] = (spacing > int(window_days)).astype(int)
    frame["episode_idx"] = frame.groupby(member_col)["episode_open"].cumsum()
    frame["episode_id"] = (
        frame[member_col].astype(str) + "_" + archetype + "_" + frame["episode_idx"].astype(str)
    )
    frame["episode_archetype"] = archetype
    if care_domain_col and care_domain_col not in frame.columns:
        frame[care_domain_col] = "unspecified"
    return frame


def score_episode_risk(
    episodes_df: pd.DataFrame,
    *,
    member_col: str = "member_id",
    date_col: str = "service_date",
    amount_col: str = "allowed_amount",
    episode_id_col: str = "episode_id",
    diagnosis_code_col: str | None = None,
    procedure_code_col: str | None = None,
) -> pd.DataFrame:
    """
    Episode-level financial and clinical-density markers: spend, span, intensity,
    and optional ICD/CPT breadth for morbidity and procedural complexity signals.
    """
    required = {episode_id_col, member_col, amount_col, date_col}
    missing = sorted(required.difference(episodes_df.columns))
    if missing:
        raise ValueError(f"episodes_df missing required columns: {missing}")

    agg: dict[str, tuple[str, str]] = {
        "episode_allowed_total": (amount_col, "sum"),
        "claim_count": (amount_col, "size"),
        "episode_start": (date_col, "min"),
        "episode_end": (date_col, "max"),
    }
    if diagnosis_code_col and diagnosis_code_col in episodes_df.columns:
        agg["clinical_condition_breadth"] = (diagnosis_code_col, "nunique")
    if procedure_code_col and procedure_code_col in episodes_df.columns:
        agg["procedural_intensity_breadth"] = (procedure_code_col, "nunique")

    grouped = (
        episodes_df.groupby([episode_id_col, member_col], as_index=False).agg(**agg).copy()
    )
    grouped["episode_span_days"] = (
        pd.to_datetime(grouped["episode_end"]) - pd.to_datetime(grouped["episode_start"])
    ).dt.days + 1
    grouped["episode_risk_score"] = (
        grouped["episode_allowed_total"] / grouped["episode_span_days"].clip(lower=1)
    )
    grouped["episode_financial_intensity"] = grouped["episode_allowed_total"] / grouped[
        "claim_count"
    ].clip(lower=1)
    if len(grouped) > 0:
        grouped["episode_severity_percentile"] = grouped["episode_risk_score"].rank(pct=True)
    else:
        grouped["episode_severity_percentile"] = pd.Series(dtype=float)
    return grouped
