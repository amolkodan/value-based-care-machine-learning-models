from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def calculate_pmpm(engine: Engine) -> pd.DataFrame:
    query = text(
        '''
        select
          month,
          sum(allowed_amount)::numeric(12,2) as total_allowed,
          count(distinct member_id) as members,
          (sum(allowed_amount) / nullif(count(distinct member_id), 0))::numeric(12,2) as pmpm
        from vbc.member_months
        group by month
        order by month
        '''
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def score_shared_savings(engine: Engine, contract_id: str) -> pd.DataFrame:
    query = text(
        '''
        with actual as (
          select
            month,
            (sum(allowed_amount) / nullif(count(distinct member_id), 0))::numeric(12,2) as actual_pmpm
          from vbc.member_months
          group by month
        ),
        target as (
          select
            month,
            target_pmpm,
            shared_savings_rate
          from vbc.benchmarks
          where contract_id = :contract_id
        )
        select
          a.month,
          a.actual_pmpm,
          t.target_pmpm,
          (t.target_pmpm - a.actual_pmpm)::numeric(12,2) as pmpm_savings,
          greatest(t.target_pmpm - a.actual_pmpm, 0)::numeric(12,2) as pmpm_positive_savings,
          (greatest(t.target_pmpm - a.actual_pmpm, 0) * t.shared_savings_rate)::numeric(12,2) as pmpm_shared_savings
        from actual a
        join target t on t.month = a.month
        order by a.month
        '''
    )
    with engine.connect() as connection:
        return pd.read_sql(query, connection, params={"contract_id": contract_id})
