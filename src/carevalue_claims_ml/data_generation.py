from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True)
class SyntheticDataConfig:
    member_count: int
    months: int
    start_month: date
    random_seed: int


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def generate_synthetic_dataset(output_dir: Path, config: SyntheticDataConfig) -> None:
    random.seed(config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    members_path = output_dir / "members.csv"
    providers_path = output_dir / "providers.csv"
    eligibility_path = output_dir / "eligibility.csv"
    claims_header_path = output_dir / "claims_header.csv"
    claims_line_path = output_dir / "claims_line.csv"
    diagnosis_path = output_dir / "diagnosis.csv"
    benchmarks_path = output_dir / "benchmarks.csv"
    member_context_path = output_dir / "member_context.csv"
    interventions_path = output_dir / "interventions.csv"

    providers = []
    for i in range(1, 31):
        provider_id = f"P{i:04d}"
        specialty = random.choice(["FAMILY", "INTERNAL", "PEDIATRICS", "CARDIO", "ENDO", "OBGYN"])
        providers.append((provider_id, f"{1000000000 + i}", f"Provider {i}", specialty))

    with providers_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["provider_id", "npi", "provider_name", "specialty"])
        w.writerows(providers)

    members = []
    genders = ["F", "M"]
    for i in range(1, config.member_count + 1):
        member_id = f"M{i:06d}"
        dob_year = random.randint(1945, 2018)
        dob_month = random.randint(1, 12)
        dob_day = random.randint(1, 28)
        gender = random.choice(genders)
        members.append(
            (
                member_id,
                f"First{i}",
                f"Last{i}",
                f"{dob_year}-{dob_month:02d}-{dob_day:02d}",
                gender,
                f"{random.randint(10000, 99999)}",
            )
        )

    with members_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["member_id", "first_name", "last_name", "dob", "gender", "zip"])
        w.writerows(members)

    with member_context_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "member_id",
                "sdoh_risk_index",
                "adherence_proxy",
                "digital_engagement_score",
                "dual_status_proxy",
                "chronic_burden_score",
            ]
        )
        for member_id, *_ in members:
            sdoh = round(random.uniform(0.0, 1.0), 3)
            adherence = round(random.uniform(0.2, 0.98), 3)
            engagement = round(random.uniform(0.0, 1.0), 3)
            dual_proxy = int(random.random() < 0.2)
            chronic_burden = round(1.0 + 4.0 * sdoh + random.uniform(0.0, 3.0), 3)
            w.writerow([member_id, sdoh, adherence, engagement, dual_proxy, chronic_burden])

    months = [add_months(month_start(config.start_month), m) for m in range(config.months)]
    with eligibility_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["member_id", "month", "product", "payer"])
        for member_id, *_ in members:
            payer = random.choice(["PAYER_A", "PAYER_B"])
            product = random.choice(["HMO", "PPO"])
            for m in months:
                if random.random() < 0.04:
                    continue
                w.writerow([member_id, m.isoformat(), product, payer])

    contract_id = "DEMO"
    with benchmarks_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["contract_id", "month", "target_pmpm", "trend_factor", "shared_savings_rate"])
        base = 420.0
        trend = 1.006
        for idx, m in enumerate(months):
            # Introduce controlled temporal drift to stress-test models.
            drift = 1.0 + (0.0015 if idx > (len(months) // 2) else 0.0)
            target = base * (trend**idx) * drift
            w.writerow([contract_id, m.isoformat(), f"{target:.2f}", f"{(trend * drift):.5f}", "0.50"])

    claim_rows = []
    line_rows = []
    diag_rows = []
    claim_id_counter = 1
    line_id_counter = 1
    diag_id_counter = 1

    high_risk_members = set(
        random.sample([m[0] for m in members], k=max(10, config.member_count // 20))
    )

    for member_id, *_ in members:
        for m in months:
            if random.random() < 0.12:
                continue
            provider_id = random.choice(providers)[0]
            claim_type = random.choice(["PROF", "OP", "IP"])
            service_from = m + timedelta(days=random.randint(0, 25))
            service_to = service_from + timedelta(days=random.randint(0, 3 if claim_type != "IP" else 10))
            base_allowed = random.uniform(40, 450)
            if claim_type == "IP":
                base_allowed *= random.uniform(8, 20)
            if member_id in high_risk_members:
                base_allowed *= random.uniform(1.3, 2.2)
            allowed = round(base_allowed, 2)
            paid = round(allowed * random.uniform(0.7, 0.98), 2)
            admit_date = service_from if claim_type == "IP" else ""
            discharge_date = service_to if claim_type == "IP" else ""

            claim_id = f"C{claim_id_counter:09d}"
            claim_id_counter += 1
            claim_rows.append(
                [
                    claim_id,
                    member_id,
                    provider_id,
                    claim_type,
                    admit_date,
                    discharge_date,
                    service_from.isoformat(),
                    service_to.isoformat(),
                    f"{allowed:.2f}",
                    f"{paid:.2f}",
                ]
            )

            line_count = random.randint(1, 3 if claim_type != "IP" else 6)
            for line_num in range(1, line_count + 1):
                claim_line_id = f"CL{line_id_counter:09d}"
                line_id_counter += 1
                cpt = random.choice(["99213", "99214", "93000", "80053", "85025", "36415", "71046", "99223"])
                revenue = random.choice(["0450", "0360", "0250", "0300", "0001"])
                pos = random.choice(["11", "22", "21", "23"])
                line_allowed = round(allowed / line_count * random.uniform(0.7, 1.3), 2)
                line_rows.append([claim_line_id, claim_id, line_num, cpt, revenue, f"{line_allowed:.2f}", pos])

            diag_count = random.randint(1, 4)
            for pos in range(1, diag_count + 1):
                diag_id = f"D{diag_id_counter:09d}"
                diag_id_counter += 1
                icd10 = random.choice(["E11.9", "I10", "E78.5", "J45.909", "F32.9", "M54.5", "K21.9", "I25.10"])
                diag_rows.append([diag_id, claim_id, icd10, pos])

    with claims_header_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "claim_id",
                "member_id",
                "provider_id",
                "claim_type",
                "admit_date",
                "discharge_date",
                "service_from",
                "service_to",
                "allowed_amount",
                "paid_amount",
            ]
        )
        w.writerows(claim_rows)

    with claims_line_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_line_id", "claim_id", "line_number", "cpt_hcpcs", "revenue_code", "allowed_amount", "place_of_service"])
        w.writerows(line_rows)

    with diagnosis_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["diag_id", "claim_id", "icd10", "diag_position"])
        w.writerows(diag_rows)

    with interventions_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["member_id", "month", "care_management_touch", "intervention_type", "engagement_response"])
        for member_id, *_ in members:
            propensity = 0.12 + (0.16 if member_id in high_risk_members else 0.0)
            for m in months:
                touched = int(random.random() < propensity)
                intervention_type = random.choice(["none", "outreach", "pharmacy", "care_navigation"])
                if not touched:
                    intervention_type = "none"
                response = round(
                    random.uniform(0.0, 1.0) * (1.2 if touched and intervention_type != "none" else 0.6), 3
                )
                w.writerow([member_id, m.isoformat(), touched, intervention_type, response])
