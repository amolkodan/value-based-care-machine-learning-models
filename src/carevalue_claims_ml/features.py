from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def build_member_month_features(engine: Engine, feature_window_months: int) -> pd.DataFrame:
    query = text(
        '''
        with ordered as (
          select
            member_id,
            month,
            allowed_amount,
            inpatient_admits,
            ed_visits,
            age,
            case when gender = 'F' then 1 else 0 end as is_female
          from vbc.member_months
        ),
        rolling as (
          select
            o.member_id,
            o.month,
            o.age,
            o.is_female,
            sum(o.allowed_amount) over (
              partition by o.member_id
              order by o.month
              rows between :window preceding and 1 preceding
            ) as allowed_last_window,
            sum(o.inpatient_admits) over (
              partition by o.member_id
              order by o.month
              rows between :window preceding and 1 preceding
            ) as ip_last_window,
            sum(o.ed_visits) over (
              partition by o.member_id
              order by o.month
              rows between :window preceding and 1 preceding
            ) as ed_last_window,
            lag(o.allowed_amount, 1) over (partition by o.member_id order by o.month) as allowed_last_month
          from ordered o
        )
        select
          member_id,
          month,
          age,
          is_female,
          coalesce(allowed_last_window, 0) as allowed_last_window,
          coalesce(ip_last_window, 0) as ip_last_window,
          coalesce(ed_last_window, 0) as ed_last_window,
          coalesce(allowed_last_month, 0) as allowed_last_month
        from rolling
        '''
    )
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"window": int(feature_window_months)})
    return df


def build_high_cost_label(engine: Engine, horizon_months: int, threshold_quantile: float = 0.9) -> pd.DataFrame:
    with engine.connect() as connection:
        base = pd.read_sql(
            text("select member_id, month, allowed_amount from vbc.member_months"),
            connection,
        )
    base["month"] = pd.to_datetime(base["month"])
    base = base.sort_values(["member_id", "month"]).reset_index(drop=True)

    shifted = []
    for k in range(1, int(horizon_months) + 1):
        tmp = base[["member_id", "month", "allowed_amount"]].copy()
        tmp["month"] = tmp["month"] - pd.offsets.DateOffset(months=k)
        tmp = tmp.rename(columns={"allowed_amount": f"allowed_plus_{k}"})
        shifted.append(tmp)

    merged = base[["member_id", "month"]].copy()
    for tmp in shifted:
        merged = merged.merge(tmp, on=["member_id", "month"], how="left")

    future_cols = [c for c in merged.columns if c.startswith("allowed_plus_")]
    merged["future_allowed_sum"] = merged[future_cols].fillna(0).sum(axis=1)
    threshold = float(merged["future_allowed_sum"].quantile(float(threshold_quantile)))
    merged["label_high_cost"] = (merged["future_allowed_sum"] >= threshold).astype(int)
    return merged[["member_id", "month", "label_high_cost", "future_allowed_sum"]]
