with eligible as (
  select member_id, month, payer, product
  from vbc.eligibility
),
claims_mm as (
  select
    c.member_id,
    date_trunc('month', c.service_from)::date as month,
    sum(c.allowed_amount) as allowed_amount,
    sum(case when c.claim_type = 'IP' then 1 else 0 end) as inpatient_admits,
    sum(case when c.claim_type = 'OP' and exists (
      select 1
      from vbc.claims_line cl
      where cl.claim_id = c.claim_id and cl.place_of_service = '23'
    ) then 1 else 0 end) as ed_visits
  from vbc.claims_header c
  group by 1, 2
),
member_demo as (
  select
    m.member_id,
    m.gender,
    date_part('year', age(current_date, m.dob))::int as age
  from vbc.members m
)
insert into vbc.member_months (member_id, month, age, gender, payer, product, pcp_provider_id, allowed_amount, inpatient_admits, ed_visits)
select
  e.member_id,
  e.month,
  d.age,
  d.gender,
  e.payer,
  e.product,
  null::text as pcp_provider_id,
  coalesce(c.allowed_amount, 0)::numeric(12,2) as allowed_amount,
  coalesce(c.inpatient_admits, 0)::int as inpatient_admits,
  coalesce(c.ed_visits, 0)::int as ed_visits
from eligible e
join member_demo d on d.member_id = e.member_id
left join claims_mm c on c.member_id = e.member_id and c.month = e.month
on conflict (member_id, month) do update
set
  age = excluded.age,
  gender = excluded.gender,
  payer = excluded.payer,
  product = excluded.product,
  allowed_amount = excluded.allowed_amount,
  inpatient_admits = excluded.inpatient_admits,
  ed_visits = excluded.ed_visits;
