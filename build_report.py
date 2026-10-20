"""Writes Phase3_Model_Report.md from models/metrics.json  (python build_report.py)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
R = json.load(open(HERE / "models" / "metrics.json"))
f = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"          # noqa: E731
p = lambda v: "n/a" if v is None else f"{v*100:.1f}%"            # noqa: E731
M = R["meta"]
L = []
w = L.append

NOTES = {
    "Logistic Regression": "A linear, easily explained model. It cannot represent interactions (e.g. 'recent postings matter more for injectables'), so it is a strong baseline but usually the first to be overtaken when such interactions exist.",
    "Decision Tree": "Fully transparent rules, but a single tree is high-variance: small changes in the data change the tree, and it tends to give coarse, poorly ranked probabilities.",
    "Naive Bayes": "Assumes features are independent, which is clearly false here (e.g. postings in the last 3, 6 and 12 months overlap). It often reaches high recall by over-flagging, at the cost of precision.",
    "K-Nearest Neighbours": "Sensitive to feature scaling and to many irrelevant one-hot columns; slow at prediction time on large panels, so it is only used on the small record-level table.",
    "SVM (RBF)": "Can model non-linear boundaries on small data; probabilities come from an extra calibration step and training does not scale to the large drug-month panel, so it is only used on the record-level table.",
    "Random Forest": "Averages many de-correlated trees, which lowers variance and handles interactions and non-linearity without scaling. Probabilities are usually less well calibrated than boosting or logistic regression.",
    "Gradient Boosting (Hist)": "Builds trees sequentially to fix earlier errors; typically the strongest family on tabular data and fast thanks to histogram binning.",
    "XGBoost": "Regularised gradient boosting with built-in handling of imbalance and per-prediction explanations (SHAP-style contributions), which the app uses to show why a medicine is flagged.",
    "Baseline: base rate": "Predicts the class prior for everyone. Its high accuracy on rare events is exactly why accuracy alone is misleading.",
    "Baseline: majority class": "Always predicts the majority class; the floor every model must beat.",
    "Rule: posted in last 12 months": "A one-line heuristic a pharmacist could apply by hand; any ML model must clearly beat it to justify its complexity.",
}


def clf_table(models, split="test"):
    w("| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | CV PR-AUC (±sd) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in models:
        x = r[split]
        w(f"| {r['model']} | {f(x['accuracy'])} | {f(x['precision'])} | {f(x['recall'])} | {f(x['f1'])} | {f(x['f2'])} | "
          f"{f(x['roc_auc'])} | {f(x['pr_auc'])} | {f(r['cv_pr_auc_mean'])} ± {f(r['cv_pr_auc_std'])} |")
    w("")


def discuss(models, best):
    for r in models:
        x = r["test"]
        tag = " **(selected)**" if r["model"] == best else ""
        w(f"- **{r['model']}**{tag}: recall {p(x['recall'])}, precision {p(x['precision'])}, F2 {f(x['f2'])}, PR-AUC {f(x['pr_auc'])} "
          f"(TP {x['tp']}, FP {x['fp']}, FN {x['fn']}, TN {x['tn']}). {NOTES.get(r['model'], '')}")
    w("")


w("# Phase 3 - Model Building and Performance Comparison")
w("## AI-Powered Medicine Availability & Shortage Prediction Network\n")
if M.get("is_demo"):
    w("> **WARNING - DEMO DATA.** This file was generated from a synthetic test sample (`" + M["data_source"] +
      "`). None of the numbers below are real results. Retrain on the real `shortages_raw.csv` and rebuild this report.\n")
w(f"*Data: {M['data_source']} - {M['n_clean']:,} cleaned shortage records, {M['n_medicines']:,} distinct medicines, postings {M['first_posting']} to {M['last_posting']}. Generated {M['trained_at']}.*\n")

w("## 1. Modelling tasks")
w("Phase 2 concluded that shortage risk and demand forecasting are different problems and that synthetic sales data carries no evidence about shortages (r ≈ -0.09). "
  "Phase 3 therefore models the real FDA data directly, as five tasks:\n")
w("| Task | Type | Why |\n|---|---|---|")
w("| **A. Status classification** | Binary classification | Phase 1 objective 1: is a shortage record still active? |")
w(f"| **B. Early-warning model** | Binary classification on a drug × month panel | Predicts a **new** shortage posting within {R['early_warning']['horizon']} months - the model that powers the app |")
w("| **C. Duration** | Regression | Phase 1 objective 3: days until resolution |")
w("| **D. Volume forecast** | Time-series | Phase 1 objective 2: monthly count of new postings |")
w("| **E. Segmentation** | Clustering (silhouette) | Group medicines by shortage behaviour |\n")

w("## 2. Evaluation design")
for t in ["**Time-based splits.** Train → validation → test always move forward in time. For the early-warning model, gaps equal to the prediction horizon separate the sets so labels cannot overlap.",
          "**No label leakage.** `update_type`, `update_date`, `discontinued_date`, `related_info` and `status` are never inputs; company frequency is learned from training rows only. `tests/test_pipeline.py` verifies features use no future information.",
          "**Selection on validation only.** The best model is chosen by validation PR-AUC (F2 as tie-break); the test set is used once for reporting. Thresholds are tuned on validation to maximise F2 (recall weighted 2×), matching the Phase 1 decision that a missed shortage costs more than a false alarm.",
          "**Metrics.** Accuracy, precision, recall, F1, F2, ROC-AUC and PR-AUC (plus Brier score in `metrics.json`); expanding-window cross-validation PR-AUC to show stability. Regression: MAE, RMSE, median AE, R². Forecasting: MAE, RMSE, sMAPE, MASE. Clustering: silhouette, Davies-Bouldin, Calinski-Harabasz.",
          "**Baselines.** Base-rate / majority-class predictor and the one-line rule 'a medicine with a posting in the last 12 months will have another'."]:
    w("- " + t)
w("")

ew = R["early_warning"]
w(f"## 3. Early-warning model (task B)")
sm = ew["split_months"]
w(f"Observations: {ew['n_train']:,} train / {ew['n_val']:,} validation / {ew['n_test']:,} test drug-months. Train through {sm['train_end']}, validation {sm['val'][0]} to {sm['val'][1]}, test {sm['test'][0]} to {sm['test'][1]}. "
  f"Positive rate: train {p(ew['positive_rate']['train'])}, validation {p(ew['positive_rate']['val'])}, test {p(ew['positive_rate']['test'])}.\n")
clf_table(ew["models"])
discuss(ew["models"], ew["best"])
b = next(r for r in ew["models"] if r["model"] == ew["best"])
rule = next(r for r in ew["models"] if r["model"].startswith("Rule"))
rest = sorted([r for r in ew["models"] if r["model"] != ew["best"] and not r["model"].startswith(("Baseline", "Rule"))], key=lambda r: -(r["val"]["pr_auc"] or 0))
w("### Why the selected model is the better solution")
w(f"1. **Selected: {ew['best']}** - highest validation PR-AUC ({f(b['val']['pr_auc'])}), chosen without touching the test set.")
w(f"2. **It beats the baselines.** Test PR-AUC {f(b['test']['pr_auc'])} vs {f(next(r for r in ew['models'] if r['model'].startswith('Baseline'))['test']['pr_auc'])} for the base rate; "
  f"F2 {f(b['test']['f2'])} vs {f(rule['test']['f2'])} for the simple 12-month rule.")
w(f"3. **Operating point.** At the validation-chosen threshold ({f(b['test']['threshold'])}) it catches {p(b['test']['recall'])} of medicines that really received a new shortage posting, with precision {p(b['test']['precision'])} "
  f"- i.e. about {f(1/max(b['test']['precision'], 1e-9), 1)} flagged medicines per true shortage versus {f(1/max(b['test']['positive_rate'], 1e-9), 1)} if medicines were picked at random.")
if rest:
    r2 = rest[0]
    gap = abs((b["cv_pr_auc_mean"] or 0) - (r2["cv_pr_auc_mean"] or 0))
    sd = max(b["cv_pr_auc_std"] or 0, r2["cv_pr_auc_std"] or 0)
    w(f"4. **Versus the runner-up ({r2['model']}).** CV PR-AUC {f(b['cv_pr_auc_mean'])} ± {f(b['cv_pr_auc_std'])} vs {f(r2['cv_pr_auc_mean'])} ± {f(r2['cv_pr_auc_std'])}; "
      + ("the gap is within fold-to-fold variation, so the two are statistically close and the choice rests on simplicity, speed and explainability as much as on score." if gap <= sd
         else "the gap exceeds fold-to-fold variation, which supports the choice."))
w("5. **Most influential inputs:** " + ", ".join(f"{i['feature']} ({f(i['importance'])})" for i in ew["importance"][:5]) + ".\n")

sc = R["status_classification"]
w("## 4. Status classification (task A)")
if sc.get("skipped"):
    w(sc["skipped"] + "\n")
else:
    w(f"Stratified random split ({sc['n_train']} / {sc['n_val']} / {sc['n_test']}); positive rate {p(sc['positive_rate']['train'])}. Status is a snapshot label, so a stratified split is the primary yardstick.\n")
    clf_table(sc["models"])
    discuss(sc["models"], sc["best"])
    tp = sc["temporal"]
    w(f"**Chronological check.** Positive share by period: train {p(tp['positive_rate']['train'])}, validation {p(tp['positive_rate']['val'])}, test {p(tp['positive_rate']['test'])}. {tp['note']}\n")
    w("| Model | Test recall | Test precision | Test F2 |\n|---|---|---|---|")
    for r in tp["models"]:
        w(f"| {r['model']} | {f(r['test']['recall'])} | {f(r['test']['precision'])} | {f(r['test']['f2'])} |")
    w("")

du = R["duration"]
w("## 5. Shortage duration (task C)")
if du.get("skipped"):
    w(du["skipped"] + "\n")
else:
    w(du["note"] + "\n")
    w("| Model | MAE (days) | RMSE (days) | Median AE | R² (log) |\n|---|---|---|---|---|")
    for r in du["models"]:
        x = r["test"]
        w(f"| {r['model']}{' **(selected)**' if r['model'] == du['best'] else ''} | {f(x['mae_days'], 1)} | {f(x['rmse_days'], 1)} | {f(x['median_ae_days'], 1)} | {f(x['r2_log'])} |")
    w("")

fc = R["forecast"]
w("## 6. Monthly volume forecast (task D)")
if fc.get("skipped"):
    w(fc["skipped"] + "\n")
else:
    w(f"Trained through {fc['train_end']}; evaluated on the next {fc['test_months']} months. Mean monthly volume: {f(fc['mean_monthly'], 1)} postings.\n")
    w("| Model | MAE | RMSE | sMAPE % | MASE | Rolling 1-step MAE |\n|---|---|---|---|---|---|")
    for r in fc["models"]:
        h, o = r["holdout_12m"], r["rolling_1step"]
        w(f"| {r['model']}{' **(selected)**' if r['model'] == fc['best'] else ''} | {f(h['mae'], 2)} | {f(h['rmse'], 2)} | {f(h['smape'], 1)} | {f(h['mase'], 2)} | {f(o['mae'], 2)} |")
    w("\nWith only 12 test points, differences between close models are within noise; the naive and seasonal-naive rows show whether any model adds real value.\n")

cl = R["clustering"]
w("## 7. Segmentation (task E)")
ch = cl["chosen"]
w(f"Chosen: **{ch['algorithm']}, k = {ch['k']}**, silhouette = {f(ch['silhouette'])}. {cl['note']}\n")
w("| Algorithm | k | Silhouette | Davies-Bouldin | Calinski-Harabasz |\n|---|---|---|---|---|")
for g in cl["grid"]:
    w(f"| {g['algorithm']} | {g['k']} | {f(g['silhouette'])} | {f(g['davies_bouldin'])} | {f(g['calinski_harabasz'], 1)} |")
w("")
w("## 8. Limitations")
for t in ["Only medicines that have appeared in the FDA database can be scored.",
          "Signals come from FDA posting history, not hospital inventory or demand; the Phase 2 synthetic sales series is deliberately not used.",
          "Duration is only observable for resolved shortages (right-censoring).",
          "Forecast intervals are approximate and the hold-out window is short."]:
    w("- " + t)
(HERE / "Phase3_Model_Report.md").write_text("\n".join(L), encoding="utf-8")
print("wrote Phase3_Model_Report.md,", len(L), "lines")
