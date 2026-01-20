select
  month,
  sum(allowed_amount)::numeric(12,2) as total_allowed,
  count(distinct member_id) as members,
  (sum(allowed_amount) / nullif(count(distinct member_id), 0))::numeric(12,2) as pmpm
from vbc.member_months
group by month
order by month;
