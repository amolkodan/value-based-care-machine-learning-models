from __future__ import annotations

import pandas as pd


def build_bundled_episodes(
    claims_df: pd.DataFrame,
    archetype: str = "general",
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Build simple episode groupings from claims rows.

    This additive implementation is intentionally lightweight so teams can
    start episode-level modeling without changing existing claim workflows.
    """
    required = {"member_id", "service_date", "allowed_amount"}
    missing = sorted(required.difference(claims_df.columns))
    if missing:
        raise ValueError(f"claims_df missing required columns: {missing}")

    frame = claims_df.copy()
    frame["service_date"] = pd.to_datetime(frame["service_date"])
    frame = frame.sort_values(["member_id", "service_date"])
    spacing = frame.groupby("member_id")["service_date"].diff().dt.days.fillna(window_days + 1)
    frame["episode_open"] = (spacing > int(window_days)).astype(int)
    frame["episode_idx"] = frame.groupby("member_id")["episode_open"].cumsum()
    frame["episode_id"] = (
        frame["member_id"].astype(str) + "_" + archetype + "_" + frame["episode_idx"].astype(str)
    )
    return frame


def score_episode_risk(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additive baseline episode-level risk/cost markers.
    """
    required = {"episode_id", "member_id", "allowed_amount", "service_date"}
    missing = sorted(required.difference(episodes_df.columns))
    if missing:
        raise ValueError(f"episodes_df missing required columns: {missing}")

    grouped = (
        episodes_df.groupby(["episode_id", "member_id"], as_index=False)
        .agg(
            episode_allowed_total=("allowed_amount", "sum"),
            claim_count=("allowed_amount", "size"),
            episode_start=("service_date", "min"),
            episode_end=("service_date", "max"),
        )
        .copy()
    )
    grouped["episode_span_days"] = (
        pd.to_datetime(grouped["episode_end"]) - pd.to_datetime(grouped["episode_start"])
    ).dt.days + 1
    grouped["episode_risk_score"] = (
        grouped["episode_allowed_total"] / grouped["episode_span_days"].clip(lower=1)
    )
    return grouped
