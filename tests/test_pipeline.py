"""Run:  python tests/test_pipeline.py   - sanity checks that guard against time leakage."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as pl

HERE = Path(__file__).resolve().parents[1]
src = HERE / "data" / "shortages_raw.csv"
df = pl.clean(pl.load_raw(src if src.exists() else HERE / "data" / "DEMO_sample_shortages.csv"))
pan = pl.build_panel(df, horizon=6)
months = pan.months

# 1. features at month t only count postings up to and including t; the label only counts t+1..t+H
rng = np.random.default_rng(0)
for i in rng.choice(len(pan.X), 200, replace=False):
    d, t = pan.drug[i], pan.t[i]
    sub = df[df.drug == d]
    mi = (sub.post_month - months[0]).map(lambda x: x.n).values
    assert pan.X.c_total.iloc[i] == (mi <= t).sum(), "c_total uses future data"
    assert pan.X.c12.iloc[i] == ((mi <= t) & (mi > t - 12)).sum(), "c12 wrong"
    if pan.labelable[i]:
        assert pan.y[i] == float(((mi > t) & (mi <= t + 6)).any()), "label wrong"
# 2. no leaky columns in the status-classification features
X = pl.record_features(df, np.arange(int(len(df) * .7)))
for bad in ["update_type", "update_date", "discontinued", "related_info", "status", "is_shortage", "duration"]:
    assert not any(bad in c for c in X.columns), f"leaky feature: {bad}"
# 3. time splits are strictly ordered with a purge gap
out, *_ = pl.task_early_warning(df, 6)
sp = out["split_months"]
assert sp["train_end"] < sp["val"][0] and sp["val"][1] < sp["test"][0]
print("all leakage checks passed ·", len(pan.X), "panel rows ·", pan.X.shape[1], "features")
