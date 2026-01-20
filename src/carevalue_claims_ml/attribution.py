from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def attribute_members_to_providers(engine: Engine, contract_id: str) -> int:
    query = text(
        '''
        with prof_visits as (
          select
            ch.member_id,
            date_trunc('month', ch.service_from)::date as month,
            ch.provider_id,
            count(*) as visit_count
          from vbc.claims_header ch
          where ch.claim_type in ('PROF', 'OP')
            and ch.provider_id is not null
          group by 1, 2, 3
        ),
        ranked as (
          select
            member_id,
            month,
            provider_id,
            visit_count,
            row_number() over (partition by member_id, month order by visit_count desc, provider_id) as rn
          from prof_visits
        )
        insert into vbc.attribution (contract_id, member_id, month, attributed_provider_id, attribution_method)
        select
          :contract_id,
          member_id,
          month,
          provider_id,
          'most_visits'
        from ranked
        where rn = 1
        on conflict (contract_id, member_id, month) do update
        set
          attributed_provider_id = excluded.attributed_provider_id,
          attribution_method = excluded.attribution_method
        '''
    )
    with engine.begin() as connection:
        result = connection.execute(query, {"contract_id": contract_id})
    return int(getattr(result, "rowcount", 0) or 0)
