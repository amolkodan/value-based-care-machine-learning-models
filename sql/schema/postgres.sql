create schema if not exists vbc;

create table if not exists vbc.members (
  member_id text primary key,
  first_name text,
  last_name text,
  dob date,
  gender text,
  zip text
);

create table if not exists vbc.eligibility (
  member_id text not null references vbc.members(member_id),
  month date not null,
  product text,
  payer text,
  primary key (member_id, month)
);

create table if not exists vbc.providers (
  provider_id text primary key,
  npi text,
  provider_name text,
  specialty text
);

create table if not exists vbc.claims_header (
  claim_id text primary key,
  member_id text not null references vbc.members(member_id),
  provider_id text references vbc.providers(provider_id),
  claim_type text,
  admit_date date,
  discharge_date date,
  service_from date not null,
  service_to date not null,
  allowed_amount numeric(12,2) not null,
  paid_amount numeric(12,2) not null
);

create table if not exists vbc.claims_line (
  claim_line_id text primary key,
  claim_id text not null references vbc.claims_header(claim_id),
  line_number int not null,
  cpt_hcpcs text,
  revenue_code text,
  allowed_amount numeric(12,2) not null,
  place_of_service text
);

create table if not exists vbc.diagnosis (
  diag_id text primary key,
  claim_id text not null references vbc.claims_header(claim_id),
  icd10 text not null,
  diag_position int not null
);

create table if not exists vbc.member_months (
  member_id text not null references vbc.members(member_id),
  month date not null,
  age int,
  gender text,
  payer text,
  product text,
  pcp_provider_id text references vbc.providers(provider_id),
  allowed_amount numeric(12,2) not null default 0,
  inpatient_admits int not null default 0,
  ed_visits int not null default 0,
  primary key (member_id, month)
);

create table if not exists vbc.benchmarks (
  contract_id text not null,
  month date not null,
  target_pmpm numeric(12,2) not null,
  trend_factor numeric(8,5) not null default 1.0,
  shared_savings_rate numeric(6,5) not null default 0.5,
  primary key (contract_id, month)
);

create table if not exists vbc.attribution (
  contract_id text not null,
  member_id text not null references vbc.members(member_id),
  month date not null,
  attributed_provider_id text references vbc.providers(provider_id),
  attribution_method text not null,
  primary key (contract_id, member_id, month)
);

create table if not exists vbc.model_scores (
  model_name text not null,
  run_id text not null,
  member_id text not null references vbc.members(member_id),
  month date not null,
  score numeric(12,6) not null,
  label int,
  primary key (model_name, run_id, member_id, month)
);
