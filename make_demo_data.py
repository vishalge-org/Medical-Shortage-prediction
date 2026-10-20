"""
Generates a SCHEMA-FAITHFUL *DEMO* file (data/DEMO_sample_shortages.csv) so the code can be smoke-tested without
internet access. It is NOT real FDA data - none of the metrics produced from it mean anything. The app and the
metrics file flag it as demo data. For real results use the Phase 2 file `shortages_raw.csv`.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
cats = ["Anesthesia", "Psychiatry", "Cardiovascular", "Analgesia/Addiction", "Anti-Infective", "Oncology",
        "Endocrinology/Metabolism", "Gastroenterology", "Neurology", "Unspecified"]
forms = ["Injection, solution", "Tablet", "Capsule", "Injection, powder, lyophilized, for solution", "Solution", "Suspension", "Cream"]
routes = ["INTRAVENOUS", "ORAL", "SUBCUTANEOUS", "INTRAMUSCULAR", "TOPICAL"]
companies = [f"Pharma Co {i}" for i in range(1, 60)]
n_drugs = 330
drug_cat = rng.choice(cats, n_drugs, p=[.14, .08, .13, .06, .14, .12, .10, .08, .08, .07])
propensity = rng.gamma(0.8, 1.2, n_drugs)
rows = []
for d in range(n_drugs):
    k = rng.poisson(propensity[d] * 3.2) + 1
    # shortages cluster in time for the same drug (episodes)
    start = rng.uniform(2012, 2026.5)
    for j in range(k):
        t = start + abs(rng.normal(0, 2.5)) * (j > 0) + rng.exponential(0.3) * j
        if t > 2026.7:
            t = rng.uniform(2018, 2026.7)
        dt = pd.Timestamp("2012-01-01") + pd.Timedelta(days=int((t - 2012) * 365.25))
        age = (pd.Timestamp("2026-10-01") - dt).days
        dur = int(rng.lognormal(5.6, 0.9))
        resolved = dur < age and rng.random() < 0.88
        status = "Resolved" if resolved else rng.choice(["Current", "Current", "Current", "To Be Discontinued"])
        upd = dt + pd.Timedelta(days=dur) if resolved else min(dt + pd.Timedelta(days=int(rng.integers(30, 400))), pd.Timestamp("2026-10-10"))
        f = rng.choice(forms, p=[.35, .2, .1, .1, .12, .08, .05])
        rows.append(dict(
            generic_name=f"Demodrug {d:03d} {'Injection' if 'Inject' in f else 'Oral'}", brand_name=f"DemoBrand{d:03d}" if rng.random() > .1 else None,
            dosage_form=f if rng.random() > .012 else None, route=rng.choice(routes) if rng.random() > .11 else None,
            therapeutic_category=drug_cat[d] if drug_cat[d] != "Unspecified" else "", pharm_class_epc="Demo class" if rng.random() > .3 else "",
            company_name=rng.choice(companies), status=status, update_type=rng.choice(["New", "Revised", "Reverified"]),
            initial_posting_date=dt.strftime("%m/%d/%Y"), update_date=upd.strftime("%m/%d/%Y"),
            discontinued_date=upd.strftime("%m/%d/%Y") if status == "To Be Discontinued" else None,
            related_info="Demo text" if rng.random() > .3 else None, presentation=f"Demo {rng.integers(1, 9)} mg vial",
            package_ndc=",".join(f"0000-{rng.integers(1000, 9999)}-{rng.integers(10, 99)}" for _ in range(rng.integers(1, 4)))))
df = pd.DataFrame(rows)
df.to_csv("data/DEMO_sample_shortages.csv", index=False)
print(len(df), "demo rows ->", "data/DEMO_sample_shortages.csv")
