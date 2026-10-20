# MedShortage AI – early warning for medicine shortages

Predicts which medicines are likely to get a **new FDA shortage posting in the next 6 months**, forecasts monthly shortage volume,
and compares every model with honest, time-based evaluation. Built on the openFDA Drug Shortages data from Phases 1–2.

## Run it (3 steps)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open **Data & Training** in the sidebar, upload the `shortages_raw.csv` your Phase 2 notebook saved (or press
*Download live from openFDA*), choose a horizon and press **Train models** (1–3 min, progress bar shown).
The six dashboard pages unlock automatically.

Prefer the command line? `python train_models.py path/to/shortages_raw.csv` then `python build_report.py` writes
`Phase3_Model_Report.md` from the real metrics. `python tests/test_pipeline.py` checks there is no time leakage.

## Pages
Overview · Risk Checker (search any generic/brand name) · Watchlist (ranked, filterable, CSV download) · Forecast ·
Model Comparison (5 tasks, ROC/PR curves, confusion matrices, "why this model") · Data Explorer · Data & Training.

## What is modelled
| Task | Models compared |
|---|---|
| Early-warning (drug × month panel) – **powers the app** | base rate, 12-month rule, Logistic Regression, Decision Tree, Naive Bayes, Random Forest, Hist-Gradient Boosting, XGBoost |
| Status classification (`is_shortage`) | + K-NN, SVM |
| Duration regression | median baseline, Ridge, Tree, Random Forest, Hist-GB, XGBoost |
| Monthly volume forecast | naive, seasonal naive, moving average, Holt-Winters, ARIMA, SARIMA, Random Forest on lags |
| Segmentation | K-Means, Agglomerative, Gaussian Mixture (silhouette, Davies-Bouldin, Calinski-Harabasz) |

Selection uses validation data only; the test set is used once. Leaky columns (`update_type`, `update_date`,
`discontinued_date`, `related_info`) are never model inputs.

## Files
`app.py` dashboard · `pipeline.py` cleaning, features, training, evaluation · `train_models.py` CLI ·
`build_report.py` report writer · `make_demo_data.py` generates a **synthetic demo file** for testing only (everything built from it is
labelled DEMO) · `tests/` leakage checks · `models/` created at training time.
