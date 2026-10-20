"""
MedShortage AI - data prep, feature engineering, model training and evaluation.

Four modelling tasks (all on the openFDA Drug Shortages data):
  A. Status classification   : is a shortage record still active (is_shortage)?          (Phase 1, objective 1)
  B. Early-warning model     : will a medicine get a NEW FDA shortage posting in the next H months?
                               (the model behind the "predict before it occurs" app)
  C. Duration regression     : how many days does a shortage last until resolution?       (Phase 1, objective 3)
  D. Forecasting             : monthly count of new shortage postings                     (Phase 1, objective 2)
  E. Clustering              : segment medicines by shortage behaviour (silhouette score)

Everything is evaluated with TIME-BASED splits (never a random shuffle) because the data is a time series
of events; random splits would let the model peek at the future.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, f1_score, fbeta_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import statsmodels.tsa.arima.model, statsmodels.tsa.holtwinters, statsmodels.tsa.statespace.sarimax  # noqa: E401,F401  (import first: statsmodels resets warning filters on import)

warnings.filterwarnings("ignore")
SEED = 42
BASE_URL = "https://api.fda.gov/drug/shortages.json"
RAW_COLS = ["generic_name", "brand_name", "dosage_form", "route", "therapeutic_category", "pharm_class_epc",
            "company_name", "status", "update_type", "initial_posting_date", "update_date",
            "discontinued_date", "related_info", "presentation", "package_ndc"]
ACTIVE_STATUSES = {"current", "to be discontinued"}


class DataError(Exception):
    """Raised with a user-readable message when the input data cannot be used."""


def _noop(frac: float, msg: str) -> None:
    pass


# --------------------------------------------------------------------------------------------------
# 1. Data access
# --------------------------------------------------------------------------------------------------
def _flatten(rec: dict) -> dict:
    o = rec.get("openfda", {}) or {}
    return {
        "generic_name": rec.get("generic_name"),
        "brand_name": (o.get("brand_name") or [None])[0],
        "dosage_form": rec.get("dosage_form"),
        "route": (o.get("route") or [None])[0],
        "therapeutic_category": ", ".join(rec.get("therapeutic_category", []) or []),
        "pharm_class_epc": ", ".join(o.get("pharm_class_epc", []) or []),
        "company_name": rec.get("company_name"),
        "status": rec.get("status"),
        "update_type": rec.get("update_type"),
        "initial_posting_date": rec.get("initial_posting_date"),
        "update_date": rec.get("update_date"),
        "discontinued_date": rec.get("discontinued_date"),
        "related_info": rec.get("related_info"),
        "presentation": rec.get("presentation"),
        "package_ndc": rec.get("package_ndc"),
    }


def fetch_live(progress=_noop, page_size: int = 100, retries: int = 3) -> pd.DataFrame:
    """Download every record from the openFDA Drug Shortages API (same logic as the Phase 2 notebook)."""
    import requests
    try:
        first = requests.get(BASE_URL, params={"limit": 1, "skip": 0}, timeout=30)
        first.raise_for_status()
        total = first.json()["meta"]["results"]["total"]
    except Exception as e:  # noqa: BLE001
        raise DataError(f"Could not reach the openFDA API ({type(e).__name__}). Check your internet connection "
                        f"or upload shortages_raw.csv instead.") from e
    rows, skip = [], 0
    while skip < total:
        for attempt in range(retries):
            try:
                r = requests.get(BASE_URL, params={"limit": page_size, "skip": skip}, timeout=60)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.5 * (attempt + 1))
        else:
            raise DataError(f"openFDA stopped responding at record {skip} of {total}. Try again in a minute.")
        batch = r.json().get("results", [])
        if not batch:
            break
        rows.extend(_flatten(x) for x in batch)
        skip += page_size
        progress(min(skip / total, 1.0), f"Downloaded {min(skip, total)} / {total} FDA records")
        time.sleep(0.25)
    return pd.DataFrame(rows, columns=RAW_COLS)


def load_raw(src) -> pd.DataFrame:
    """Load shortages_raw.csv (path or file-like) and validate that the expected columns exist."""
    try:
        df = pd.read_csv(src)
    except Exception as e:  # noqa: BLE001
        raise DataError(f"The file could not be read as a CSV ({e}).") from e
    missing = [c for c in ["generic_name", "status", "initial_posting_date", "company_name"] if c not in df.columns]
    if missing:
        raise DataError("This does not look like the Phase 2 shortages_raw.csv. Missing required column(s): "
                        + ", ".join(missing) + ". Expected columns: " + ", ".join(RAW_COLS))
    for c in RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[RAW_COLS]


# --------------------------------------------------------------------------------------------------
# 2. Cleaning (same rules as Phase 2, plus the derived columns used for modelling)
# --------------------------------------------------------------------------------------------------
def _form_group(s: str) -> str:
    s = str(s).lower()
    for key, grp in [("inject", "Injection"), ("tablet", "Tablet"), ("capsule", "Capsule"), ("solution", "Solution"),
                     ("suspension", "Suspension"), ("powder", "Powder"), ("cream", "Topical"), ("ointment", "Topical"),
                     ("gel", "Topical"), ("patch", "Topical"), ("spray", "Spray/Inhal"), ("inhal", "Spray/Inhal"),
                     ("liquid", "Solution"), ("syrup", "Solution"), ("drops", "Solution")]:
        if key in s:
            return grp
    return "Unknown" if s in ("nan", "", "none") else "Other"


def _route_group(s: str) -> str:
    s = str(s).lower()
    for key, grp in [("intraven", "IV"), ("oral", "Oral"), ("subcut", "Subcutaneous"), ("intramusc", "IM"),
                     ("topical", "Topical"), ("ophth", "Ophthalmic"), ("inhal", "Inhalation")]:
        if key in s:
            return grp
    return "Unknown" if s in ("nan", "", "none") else "Other"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    n0 = len(df)
    df = df.drop_duplicates().copy()
    for c in ["generic_name", "brand_name", "dosage_form", "route", "company_name", "status", "update_type",
              "therapeutic_category", "pharm_class_epc"]:
        df[c] = df[c].astype("string").str.strip()
    df = df[df["generic_name"].notna() & (df["generic_name"] != "")]
    df = df[df["company_name"].notna() & (df["company_name"] != "")]
    for c in ["initial_posting_date", "update_date", "discontinued_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df[df["initial_posting_date"].notna()]
    df["therapeutic_category"] = df["therapeutic_category"].fillna("").replace("", "Unspecified")
    df["status"] = df["status"].fillna("Unknown")
    df["is_shortage"] = df["status"].str.lower().isin(ACTIVE_STATUSES).astype(int)
    df["drug"] = df["generic_name"].str.upper().str.replace(r"\s+", " ", regex=True)
    df["primary_category"] = df["therapeutic_category"].str.split(",").str[0].str.strip()
    df["form_group"] = df["dosage_form"].map(_form_group)
    df["route_group"] = df["route"].map(_route_group)
    df["post_month"] = df["initial_posting_date"].dt.to_period("M")
    df["n_ndc"] = df["package_ndc"].astype("string").fillna("").map(lambda s: 0 if not s else s.count(",") + 1)
    df["presentation_len"] = df["presentation"].astype("string").fillna("").str.len()
    df["has_brand"] = df["brand_name"].notna().astype(int)
    df["has_epc"] = (df["pharm_class_epc"].fillna("") != "").astype(int)
    # resolution time (only observable for records that are no longer active)
    end = df["update_date"].copy()
    disc = df["status"].str.lower().eq("discontinued") & df["discontinued_date"].notna()
    end[disc] = df.loc[disc, "discontinued_date"]
    df["duration_days"] = (end - df["initial_posting_date"]).dt.days
    df.loc[(df["is_shortage"] == 1) | (df["duration_days"] < 0), "duration_days"] = np.nan
    df = df.sort_values("initial_posting_date").reset_index(drop=True)
    if len(df) < 200:
        raise DataError(f"Only {len(df)} usable records after cleaning - need at least 200 to train reliable models.")
    df.attrs["dropped_rows"] = n0 - len(df)
    return df


# --------------------------------------------------------------------------------------------------
# 3. Shared evaluation helpers
# --------------------------------------------------------------------------------------------------
def _best_threshold(y, p, beta: float = 2.0) -> float:
    """Pick the decision threshold that maximises F-beta on validation data (beta=2 -> recall counts 2x)."""
    cands = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    if len(cands) < 2:
        return 0.5
    scores = [fbeta_score(y, p >= t, beta=beta, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(scores))])


def clf_metrics(y, p, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    one_class = len(np.unique(y)) < 2
    return dict(
        accuracy=accuracy_score(y, yhat), balanced_accuracy=balanced_accuracy_score(y, yhat),
        precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0), f2=fbeta_score(y, yhat, beta=2, zero_division=0),
        mcc=matthews_corrcoef(y, yhat), roc_auc=np.nan if one_class else roc_auc_score(y, p),
        pr_auc=np.nan if one_class else average_precision_score(y, p), brier=brier_score_loss(y, np.clip(p, 0, 1)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(thr), positive_rate=float(np.mean(y)),
    )


def _curves(y, p, n=60):
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    ix = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    jx = np.linspace(0, len(pr) - 1, min(n, len(pr))).astype(int)
    return dict(fpr=fpr[ix].round(4).tolist(), tpr=tpr[ix].round(4).tolist(),
                precision=pr[jx].round(4).tolist(), recall=rc[jx].round(4).tolist())


def _scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def expanding_folds(order_key: np.ndarray, n_folds: int = 4, purge: float = 0):
    """Expanding-window CV on a sortable key (time). Each fold validates on the chunk after its training data."""
    keys = np.sort(np.unique(order_key))
    chunks = np.array_split(keys, n_folds + 1)
    for k in range(1, n_folds + 1):
        val_keys = chunks[k]
        tr = order_key < (val_keys[0] - purge)
        va = np.isin(order_key, val_keys)
        if tr.sum() > 30 and va.sum() > 10:
            yield np.where(tr)[0], np.where(va)[0]


def run_classifiers(models: dict, X, y, tr, va, te, cv_key=None, purge=0, progress=_noop, label="", cv_folds=4, cv_splitter=None):
    """Train every model on `tr`, choose a threshold on `va`, report on `te`; add time-series CV on tr+va."""
    results, fitted = [], {}
    dev = np.concatenate([tr, va])
    for i, (name, mk) in enumerate(models.items()):
        progress(i / len(models), f"{label}: training {name}")
        t0 = time.time()
        m = mk()
        m.fit(X.iloc[tr], y[tr])
        fit_s = time.time() - t0
        pv, pt = _scores(m, X.iloc[va]), _scores(m, X.iloc[te])
        if name == "Rule: posted in last 12 months" or isinstance(m, DummyClassifier) and name.startswith("Baseline"):
            thr = 0.5
        else:
            thr = _best_threshold(y[va], pv)
        val_m, test_m = clf_metrics(y[va], pv, thr), clf_metrics(y[te], pt, thr)
        cv = []
        folds = cv_splitter(dev) if cv_splitter else expanding_folds(cv_key[dev], cv_folds, purge)
        for a, b in folds:
            try:
                mm = mk()
                mm.fit(X.iloc[dev[a]], y[dev[a]])
                pp = _scores(mm, X.iloc[dev[b]])
                yy = y[dev[b]]
                if len(np.unique(yy)) > 1:
                    cv.append((average_precision_score(yy, pp), roc_auc_score(yy, pp)))
            except Exception:  # noqa: BLE001
                continue
        cv = np.array(cv) if cv else np.full((1, 2), np.nan)
        results.append(dict(model=name, fit_seconds=round(fit_s, 2), val=val_m, test=test_m,
                            cv_pr_auc_mean=float(np.nanmean(cv[:, 0])), cv_pr_auc_std=float(np.nanstd(cv[:, 0])),
                            cv_roc_auc_mean=float(np.nanmean(cv[:, 1])), cv_roc_auc_std=float(np.nanstd(cv[:, 1])),
                            curves=_curves(y[te], pt)))
        fitted[name] = (m, thr)
    return results, fitted


def pick_best(results, exclude_prefix=("Baseline", "Rule")):
    """Model selection uses VALIDATION data only (PR-AUC, then F2) so the test set stays an honest final exam."""
    cand = [r for r in results if not r["model"].startswith(exclude_prefix)]
    cand.sort(key=lambda r: (np.nan_to_num(r["val"]["pr_auc"]), r["val"]["f2"]), reverse=True)
    return cand[0]["model"]


def split_by_fraction(n, f_tr=0.70, f_va=0.15):
    a, b = int(n * f_tr), int(n * (f_tr + f_va))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


# --------------------------------------------------------------------------------------------------
# 4. Task A - status classification (record level)
# --------------------------------------------------------------------------------------------------
def record_features(df: pd.DataFrame, train_idx) -> pd.DataFrame:
    """Only information available when a shortage is first POSTED. Deliberately excluded (label leakage):
    update_type, update_date, discontinued_date, related_info - they are written as the shortage evolves."""
    X = pd.DataFrame(index=df.index)
    for g in ["Injection", "Tablet", "Capsule", "Solution", "Suspension", "Powder", "Topical", "Spray/Inhal", "Other", "Unknown"]:
        X[f"form_{g}"] = (df["form_group"] == g).astype(int)
    for g in ["IV", "Oral", "Subcutaneous", "IM", "Topical", "Ophthalmic", "Inhalation", "Other", "Unknown"]:
        X[f"route_{g}"] = (df["route_group"] == g).astype(int)
    top_cats = df["primary_category"].iloc[train_idx].value_counts().index[:12]
    for c in top_cats:
        X[f"cat_{c[:28]}"] = (df["primary_category"] == c).astype(int)
    comp_freq = df["company_name"].iloc[train_idx].value_counts(normalize=True)   # fit on TRAIN rows only
    X["company_freq"] = df["company_name"].map(comp_freq).fillna(0.0)
    X["n_ndc"], X["presentation_len"] = df["n_ndc"], df["presentation_len"]
    X["has_brand"], X["has_epc"] = df["has_brand"], df["has_epc"]
    X["posting_year"], X["posting_month"] = df["initial_posting_date"].dt.year, df["initial_posting_date"].dt.month
    # history known at posting time (df is sorted by posting date)
    X["drug_prior_postings"] = df.groupby("drug").cumcount()
    X["company_prior_postings"] = df.groupby("company_name").cumcount()
    d = df["initial_posting_date"].values.astype("datetime64[D]").astype(int)
    prior90 = np.zeros(len(df))
    for _, ix in df.groupby("primary_category").indices.items():
        dd = d[ix]
        prior90[ix] = np.arange(len(ix)) - np.searchsorted(dd, dd - 90, side="left")
    X["category_postings_prior_90d"] = prior90
    return X


def clf_zoo_small():
    return {
        "Baseline: majority class": lambda: DummyClassifier(strategy="prior"),
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
        "Naive Bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
        "K-Nearest Neighbours": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=SEED)),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "Gradient Boosting (Hist)": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
                                         colsample_bytree=0.85, eval_metric="logloss", n_jobs=-1, random_state=SEED),
    }


def _summ(res):
    return [dict(model=r["model"], test=r["test"], val=r["val"]) for r in res]


def task_status(df, progress=_noop):
    """Status is a *snapshot* label (what the FDA lists today), so the primary evaluation is a stratified split.
    A chronological hold-out is reported as a secondary check, because the newest postings are almost all still
    'active' (a single-class test set), which makes it a weak yardstick for this particular label."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    n = len(df)
    y = df["is_shortage"].values
    idx = np.arange(n)
    if len(np.unique(y)) < 2:
        return dict(task="Status classification", skipped="All records have the same status - nothing to classify.")
    dev, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
    tr, va = train_test_split(dev, test_size=0.1765, stratify=y[dev], random_state=SEED)
    tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    X = record_features(df, tr)
    skf = lambda d: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, y[d]))   # noqa: E731
    res, fitted = run_classifiers(clf_zoo_small(), X, y, tr, va, te, progress=lambda f, m: progress(f * 0.7, m),
                                  label="Status classification", cv_splitter=skf)
    best = pick_best(res)
    m, thr = fitted[best]
    imp = permutation_importance(m, X.iloc[te], y[te], scoring="average_precision", n_repeats=5, random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1][:12]

    # secondary: chronological hold-out
    ttr, tva, tte = split_by_fraction(n)
    Xt = record_features(df, ttr)
    key = df["initial_posting_date"].values.astype("datetime64[D]").astype(int)
    tres, _ = run_classifiers(clf_zoo_small(), Xt, y, ttr, tva, tte, key, progress=lambda f, m: progress(0.7 + f * 0.3, m),
                              label="Status (chronological check)", cv_folds=3)
    return dict(
        task="Status classification (is the shortage record still active?)", split="stratified random 70/15/15",
        n_train=len(tr), n_val=len(va), n_test=len(te), features=list(X.columns),
        positive_rate=dict(train=float(y[tr].mean()), val=float(y[va].mean()), test=float(y[te].mean())),
        models=res, best=best,
        importance=[dict(feature=X.columns[i], importance=float(imp.importances_mean[i])) for i in order],
        temporal=dict(
            split="chronological 70/15/15", n_train=len(ttr), n_val=len(tva), n_test=len(tte),
            positive_rate=dict(train=float(y[ttr].mean()), val=float(y[tva].mean()), test=float(y[tte].mean())),
            split_dates=dict(train_end=str(df["initial_posting_date"].iloc[ttr[-1]].date()),
                             test_start=str(df["initial_posting_date"].iloc[tte[0]].date())),
            models=_summ(tres), best=pick_best(tres),
            note="The newest postings are overwhelmingly still active, so the share of positives changes sharply over time "
                 "(see positive_rate). Scores here measure robustness to that drift, not everyday accuracy."),
    )


# --------------------------------------------------------------------------------------------------
# 5. Task B - early-warning model on a drug x month panel
# --------------------------------------------------------------------------------------------------
class RuleLast12:
    """Transparent heuristic baseline: 'a medicine that had a shortage posting in the last 12 months will have another'."""
    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = (X["c12"].values > 0).astype(float)
        return np.c_[1 - p, p]


@dataclass
class Panel:
    X: pd.DataFrame
    y: np.ndarray
    t: np.ndarray
    drug: np.ndarray
    months: pd.PeriodIndex
    labelable: np.ndarray
    horizon: int
    drugs: list


def _roll(M: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(M, axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def build_panel(df: pd.DataFrame, horizon: int = 6) -> Panel:
    last = df["post_month"].max()
    if df["initial_posting_date"].max().day < 25:          # last month is probably incomplete -> don't learn from it
        last = last - 1
    d = df[df["post_month"] <= last]
    months = pd.period_range(d["post_month"].min(), last, freq="M")
    T = len(months)
    drugs = sorted(d["drug"].unique())
    di = {x: i for i, x in enumerate(drugs)}
    didx = d["drug"].map(di).values
    midx = (d["post_month"] - months[0]).map(lambda x: x.n).values.astype(int)
    P = np.zeros((len(drugs), T))
    np.add.at(P, (didx, midx), 1)
    cum = P.cumsum(1)
    first = np.array([np.argmax(P[i] > 0) for i in range(len(drugs))])
    tt = np.arange(T)
    last_post = np.maximum.accumulate(np.where(P > 0, tt[None, :], -1), axis=1)
    since = tt[None, :] - last_post
    episodes = np.cumsum(P > 0, axis=1)
    # distinct companies seen so far
    cf = d.assign(d_=didx, m_=midx).groupby(["d_", "company_name"])["m_"].min().reset_index()
    newc = np.zeros_like(P)
    np.add.at(newc, (cf["d_"].values, cf["m_"].values), 1)
    ncomp = newc.cumsum(1)
    # category context
    drug_cat = d.groupby("drug")["primary_category"].agg(lambda s: s.value_counts().index[0]).reindex(drugs)
    cats = sorted(drug_cat.unique())
    ci = drug_cat.map({c: i for i, c in enumerate(cats)}).values
    PC = np.zeros((len(cats), T))
    np.add.at(PC, (ci[didx], midx), 1)   # postings counted under each drug's main category
    drug_form = d.groupby("drug")["form_group"].agg(lambda s: s.value_counts().index[0]).reindex(drugs)
    G = P.sum(0, keepdims=True)
    feats = {
        "c_total": cum, "c3": _roll(P, 3), "c6": _roll(P, 6), "c12": _roll(P, 12), "c24": _roll(P, 24),
        "months_since_last": since, "age_months": tt[None, :] - first[:, None], "episodes": episodes, "n_companies": ncomp,
        "cat_c3": _roll(PC, 3)[ci], "cat_c12": _roll(PC, 12)[ci],
        "glob_c3": np.repeat(_roll(G, 3), len(drugs), 0), "glob_c12": np.repeat(_roll(G, 12), len(drugs), 0),
    }
    labelable = tt <= T - 1 - horizon
    fut = np.full_like(P, np.nan)
    fut[:, :T - horizon] = cum[:, horizon:] - cum[:, :T - horizon]
    active = tt[None, :] >= first[:, None]
    rows_d, rows_t = np.where(active & (labelable[None, :] | (tt[None, :] == T - 1)))
    X = pd.DataFrame({k: v[rows_d, rows_t] for k, v in feats.items()})
    moy = months.month.values[rows_t]
    X["moy_sin"], X["moy_cos"] = np.sin(2 * np.pi * moy / 12), np.cos(2 * np.pi * moy / 12)
    X["is_injection"] = (drug_form.values[rows_d] == "Injection").astype(int)
    top = d["primary_category"].value_counts().index[:12]
    for c in top:
        X[f"cat_{c[:28]}"] = (drug_cat.values[rows_d] == c).astype(int)
    y = (fut[rows_d, rows_t] > 0).astype(float)
    return Panel(X=X, y=y, t=rows_t, drug=np.array(drugs)[rows_d], months=months, labelable=labelable[rows_t],
                 horizon=horizon, drugs=drugs)


def clf_zoo_panel():
    return {
        "Baseline: base rate": lambda: DummyClassifier(strategy="prior"),
        "Rule: posted in last 12 months": lambda: RuleLast12(),
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=SEED),
        "Naive Bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                                                        n_jobs=-1, random_state=SEED),
        "Gradient Boosting (Hist)": lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                                                           min_samples_leaf=40, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85,
                                         colsample_bytree=0.85, min_child_weight=5, eval_metric="logloss",
                                         tree_method="hist", n_jobs=-1, random_state=SEED),
    }


def task_early_warning(df, horizon=6, progress=_noop, outdir: Path | None = None):
    pan = build_panel(df, horizon)
    lab = pan.labelable
    T = len(pan.months)
    last_lab = T - 1 - horizon
    n_test = min(24, max(6, (last_lab + 1) // 6))
    n_val = n_test
    te_lo = last_lab - n_test + 1
    va_hi = te_lo - 1 - horizon
    va_lo = va_hi - n_val + 1
    tr_hi = va_lo - 1 - horizon
    if tr_hi < 24:
        raise DataError("Not enough monthly history to build a time-based train/validation/test split.")
    t = pan.t
    tr = np.where(lab & (t <= tr_hi))[0]
    va = np.where(lab & (t >= va_lo) & (t <= va_hi))[0]
    te = np.where(lab & (t >= te_lo) & (t <= last_lab))[0]
    y = pan.y
    res, fitted = run_classifiers(clf_zoo_panel(), pan.X, y, tr, va, te, t, purge=horizon, progress=progress,
                                  label="Early-warning", cv_folds=3)
    best = pick_best(res)
    m, thr = fitted[best]
    imp = permutation_importance(m, pan.X.iloc[te], y[te], scoring="average_precision", n_repeats=3, random_state=SEED)
    order = np.argsort(imp.importances_mean)[::-1][:12]

    # --- deploy: refit best model on every labelled row, score the latest month for every medicine
    progress(0.95, "Early-warning: refitting best model on all history and scoring every medicine")
    final = clf_zoo_panel()[best]()
    all_lab = np.where(lab)[0]
    final.fit(pan.X.iloc[all_lab], y[all_lab])
    now = np.where(t == T - 1)[0]
    p_now = _scores(final, pan.X.iloc[now])
    scores = pd.DataFrame({"drug": pan.drug[now], "risk_probability": p_now})
    scores = pd.concat([scores.reset_index(drop=True), pan.X.iloc[now].reset_index(drop=True)[
        ["c_total", "c3", "c6", "c12", "c24", "months_since_last", "age_months", "episodes", "n_companies", "cat_c12"]]], axis=1)
    contrib = None
    if hasattr(final, "get_booster"):
        import xgboost as xgb
        c = final.get_booster().predict(xgb.DMatrix(pan.X.iloc[now]), pred_contribs=True)
        contrib = pd.DataFrame(c[:, :-1], columns=pan.X.columns)
        contrib.insert(0, "drug", pan.drug[now])
    base_rate = float(y[all_lab].mean())
    artefact = dict(model=final, name=best, features=list(pan.X.columns), threshold=thr, horizon=horizon,
                    scored_month=str(pan.months[-1]), base_rate=base_rate)
    out = dict(
        task=f"Early-warning: new FDA shortage posting for a medicine within the next {horizon} months",
        horizon=horizon, n_train=len(tr), n_val=len(va), n_test=len(te), n_medicines=len(pan.drugs),
        scored_month=str(pan.months[-1]), base_rate=base_rate,
        positive_rate=dict(train=float(y[tr].mean()), val=float(y[va].mean()), test=float(y[te].mean())),
        split_months=dict(train_end=str(pan.months[tr_hi]), val=[str(pan.months[va_lo]), str(pan.months[va_hi])],
                          test=[str(pan.months[te_lo]), str(pan.months[last_lab])]),
        features=list(pan.X.columns), models=res, best=best, threshold=thr,
        importance=[dict(feature=pan.X.columns[i], importance=float(imp.importances_mean[i])) for i in order],
    )
    return out, artefact, scores, contrib


HORIZONS = (3, 6, 9, 12)


def task_horizon_forecast(df, best: str, horizons=HORIZONS, progress=_noop):
    """Medicine-level forecast: for EVERY medicine, the chance of a new FDA shortage posting within 3, 6, 9 and 12 months.

    One model per horizon, all of the same family as the selected early-warning model, each with its own time-based
    train / validation / test split (purge gap = horizon) and its own validation-tuned threshold, so every horizon
    has honest test-set numbers. Probabilities are then made non-decreasing across horizons (a shortage within
    3 months is also a shortage within 6), which is a logical constraint, not a tweak to improve scores."""
    mk = clf_zoo_panel()[best]
    info, cols, notes = [], {}, []
    for k, h in enumerate(horizons):
        progress(k / len(horizons), f"Forecast: training the {h}-month model")
        try:
            pan = build_panel(df, h)
            lab, t, y, T = pan.labelable, pan.t, pan.y, len(pan.months)
            last_lab = T - 1 - h
            n_test = min(24, max(6, (last_lab + 1) // 6))
            te_lo = last_lab - n_test + 1
            va_hi = te_lo - 1 - h
            va_lo = va_hi - n_test + 1
            tr_hi = va_lo - 1 - h
            if tr_hi < 24:
                raise DataError("not enough monthly history")
            tr = np.where(lab & (t <= tr_hi))[0]
            va = np.where(lab & (t >= va_lo) & (t <= va_hi))[0]
            te = np.where(lab & (t >= te_lo) & (t <= last_lab))[0]
            m = mk()
            m.fit(pan.X.iloc[tr], y[tr])
            thr = _best_threshold(y[va], _scores(m, pan.X.iloc[va]))
            pt = _scores(m, pan.X.iloc[te])
            tm = clf_metrics(y[te], pt, thr)
            rule = clf_metrics(y[te], RuleLast12().predict_proba(pan.X.iloc[te])[:, 1], 0.5)
            final = mk()
            all_lab = np.where(lab)[0]
            final.fit(pan.X.iloc[all_lab], y[all_lab])
            now = np.where(t == T - 1)[0]
            cols[h] = pd.Series(_scores(final, pan.X.iloc[now]), index=pan.drug[now])
            info.append(dict(horizon=h, threshold=thr, base_rate=float(y[all_lab].mean()), n_train=len(tr), n_test=len(te),
                             test={k2: tm[k2] for k2 in ("recall", "precision", "f2", "pr_auc", "roc_auc", "positive_rate")},
                             rule={k2: rule[k2] for k2 in ("recall", "precision", "f2")}))
        except DataError as e:
            notes.append(f"{h}-month horizon skipped: {e}.")
    if not cols:
        return dict(horizons=[], notes=notes or ["No horizon could be trained."], model=best), None
    hz = pd.DataFrame(cols)
    hz = pd.DataFrame(np.maximum.accumulate(hz.values, axis=1), index=hz.index, columns=[f"p_{h}" for h in hz.columns])
    hz.index.name = "drug"
    return dict(horizons=info, notes=notes, model=best,
                method="One model per horizon (same family as the selected early-warning model); probabilities made "
                       "non-decreasing across horizons."), hz.reset_index()


# --------------------------------------------------------------------------------------------------
# 6. Task C - shortage duration regression
# --------------------------------------------------------------------------------------------------
def reg_metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(np.clip(p_log, 0, None))
    return dict(mae_days=float(mean_absolute_error(y, p)), rmse_days=float(np.sqrt(mean_squared_error(y, p))),
                median_ae_days=float(np.median(np.abs(y - p))), r2_log=float(r2_score(y_log, p_log)),
                r2_days=float(r2_score(y, p)))


def task_duration(df, progress=_noop):
    n_all = len(df)
    X_all = record_features(df, np.arange(int(n_all * 0.7)))
    ok = df["duration_days"].notna().values
    d = df[ok].reset_index(drop=True)
    if len(d) < 150:
        return dict(task="Shortage duration regression", skipped=f"Only {len(d)} resolved records with valid dates.")
    X = X_all[ok].reset_index(drop=True)
    y = np.log1p(d["duration_days"].values)
    tr, va, te = split_by_fraction(len(d))
    zoo = {
        "Baseline: median duration": lambda: DummyRegressor(strategy="median"),
        "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=SEED),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "Gradient Boosting (Hist)": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, random_state=SEED),
        "XGBoost": lambda: XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85,
                                        colsample_bytree=0.85, n_jobs=-1, random_state=SEED),
    }
    res = []
    for i, (name, mk) in enumerate(zoo.items()):
        progress(i / len(zoo), f"Duration regression: training {name}")
        m = mk().fit(X.iloc[tr], y[tr])
        res.append(dict(model=name, val=reg_metrics(y[va], m.predict(X.iloc[va])), test=reg_metrics(y[te], m.predict(X.iloc[te]))))
    cand = [r for r in res if not r["model"].startswith("Baseline")]
    best = min(cand, key=lambda r: r["val"]["mae_days"])["model"]
    return dict(task="Shortage duration regression (days from posting to resolution)", n_train=len(tr), n_val=len(va),
                n_test=len(te), median_days=float(d["duration_days"].median()), models=res, best=best,
                note="Only resolved/discontinued records have an observed duration, so recent long-running shortages are "
                     "absent (right-censoring). Treat predictions as 'typical duration if resolved', not a guarantee.")


# --------------------------------------------------------------------------------------------------
# 7. Task D - forecasting monthly posting volume
# --------------------------------------------------------------------------------------------------
def monthly_series(df) -> pd.Series:
    last = df["post_month"].max()
    if df["initial_posting_date"].max().day < 25:
        last = last - 1
    s = df[df["post_month"] <= last].groupby("post_month").size()
    idx = pd.period_range(s.index.min(), last, freq="M")
    return s.reindex(idx, fill_value=0).astype(float)


def _fc_naive(y, h, **_):
    return np.repeat(y[-1], h)


def _fc_snaive(y, h, **_):
    return np.array([y[-12 + (i % 12)] for i in range(h)])


def _fc_ma12(y, h, **_):
    return np.repeat(y[-12:].mean(), h)


def _fc_hw(y, h, **_):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    f = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
    return np.clip(f.forecast(h), 0, None)


def _fc_arima(y, h, order=(1, 0, 1), **_):
    from statsmodels.tsa.arima.model import ARIMA
    f = ARIMA(y, order=order, trend="c" if order[1] == 0 else "n").fit()
    return np.clip(f.forecast(h), 0, None)


def _fc_sarima(y, h, order=(1, 0, 1), seas=(1, 0, 0, 12), **_):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    f = SARIMAX(y, order=order, seasonal_order=seas, trend="c", enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False, maxiter=100)
    return np.clip(f.forecast(h), 0, None)


def _lag_features(hist, month):
    h = np.asarray(hist, dtype=float)
    return [h[-1], h[-2], h[-3], h[-6], h[-12], h[-3:].mean(), h[-12:].mean(), np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]


def _fc_rf(y, h, start_month=1, **_):
    rows, tg = [], []
    for i in range(12, len(y)):
        rows.append(_lag_features(y[:i], (start_month + i - 1) % 12 + 1))
        tg.append(y[i])
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(rows, tg)
    hist, out = list(y), []
    for k in range(h):
        mo = (start_month + len(y) + k - 1) % 12 + 1
        v = float(max(m.predict([_lag_features(hist, mo)])[0], 0))
        out.append(v)
        hist.append(v)
    return np.array(out)


def _select_orders(y):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    best, best_aic = (1, 0, 1), np.inf
    for p in range(0, 4):
        for d in (0, 1):
            for q in range(0, 3):
                try:
                    a = ARIMA(y, order=(p, d, q), trend="c" if d == 0 else "n").fit().aic
                    if a < best_aic:
                        best, best_aic = (p, d, q), a
                except Exception:  # noqa: BLE001
                    continue
    sbest, saic = ((1, 0, 1), (1, 0, 0, 12)), np.inf
    for o in [(1, 0, 1), (2, 0, 1), (1, 0, 0)]:
        for s in [(1, 0, 0, 12), (0, 1, 1, 12), (1, 0, 1, 12)]:
            try:
                a = SARIMAX(y, order=o, seasonal_order=s, trend="c", enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False, maxiter=100).aic
                if a < saic:
                    sbest, saic = (o, s), a
            except Exception:  # noqa: BLE001
                continue
    return best, sbest


def fc_metrics(actual, pred, y_train):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    scale = np.mean(np.abs(y_train[12:] - y_train[:-12])) if len(y_train) > 24 else np.mean(np.abs(np.diff(y_train)))
    denom = (np.abs(actual) + np.abs(pred))
    smape = np.mean(np.where(denom == 0, 0, 2 * np.abs(actual - pred) / np.where(denom == 0, 1, denom))) * 100
    return dict(mae=float(mean_absolute_error(actual, pred)), rmse=float(np.sqrt(mean_squared_error(actual, pred))),
                smape=float(smape), mase=float(np.mean(np.abs(actual - pred)) / max(scale, 1e-9)))


def task_forecast(df, test_months=12, progress=_noop):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _task_forecast(df, test_months, progress)


def _task_forecast(df, test_months=12, progress=_noop):
    s = monthly_series(df)
    y = s.values
    if len(y) < 48:
        return dict(task="Forecast", skipped=f"Only {len(y)} months of history; need at least 48."), None
    start_month = s.index[0].month
    n_tr = len(y) - test_months
    progress(0.05, "Forecasting: selecting ARIMA / SARIMA orders by AIC")
    order, (sorder, sseas) = _select_orders(y[:n_tr])
    models = {
        "Naive (last value)": (_fc_naive, {}),
        "Seasonal naive (same month last year)": (_fc_snaive, {}),
        "Moving average (12 mo)": (_fc_ma12, {}),
        "Holt-Winters (damped, seasonal)": (_fc_hw, {}),
        f"ARIMA{order}": (_fc_arima, dict(order=order)),
        f"SARIMA{sorder}x{sseas}": (_fc_sarima, dict(order=sorder, seas=sseas)),
        "Random Forest on lags": (_fc_rf, {}),
    }
    res = []
    for i, (name, (fn, kw)) in enumerate(models.items()):
        progress(0.1 + 0.8 * i / len(models), f"Forecasting: evaluating {name}")
        try:
            kw2 = dict(kw, start_month=start_month)
            static = fn(y[:n_tr], test_months, **kw2)
            roll = np.array([fn(y[:n_tr + k], 1, **kw2)[0] for k in range(test_months)])
        except Exception:  # noqa: BLE001
            continue
        res.append(dict(model=name,
                        holdout_12m=fc_metrics(y[n_tr:], static, y[:n_tr]),
                        rolling_1step=fc_metrics(y[n_tr:], roll, y[:n_tr]),
                        static_forecast=static.round(2).tolist(), rolling_forecast=roll.round(2).tolist()))
    cand = [r for r in res if not r["model"].startswith(("Naive", "Seasonal naive", "Moving"))]
    best = min(cand, key=lambda r: r["holdout_12m"]["mae"])["model"] if cand else res[0]["model"]
    fn, kw = models[best]
    kw2 = dict(kw, start_month=start_month)
    h = 12
    point = fn(y, h, **kw2)
    sig = np.std(np.array(next(r for r in res if r["model"] == best)["rolling_forecast"]) - y[n_tr:], ddof=1)
    sig = max(sig, 1e-6)
    lo = np.clip(point - 1.28 * sig * np.sqrt(np.arange(1, h + 1)), 0, None)
    hi = point + 1.28 * sig * np.sqrt(np.arange(1, h + 1))
    future = pd.period_range(s.index[-1] + 1, periods=h, freq="M")
    fc = pd.DataFrame({"month": future.to_timestamp(), "forecast": point, "lower_80": lo, "upper_80": hi})
    hist = pd.DataFrame({"month": s.index.to_timestamp(), "actual": y})
    out = dict(task="Monthly new-shortage posting volume", n_months=len(y), test_months=test_months,
               train_end=str(s.index[n_tr - 1]), test_start=str(s.index[n_tr]), models=res, best=best,
               test_actual=y[n_tr:].tolist(), test_index=[str(p) for p in s.index[n_tr:]],
               mean_monthly=float(y.mean()), interval_note="80% band is approximate (residual-based).")
    return out, (hist, fc)


# --------------------------------------------------------------------------------------------------
# 8. Task E - clustering medicines by behaviour
# --------------------------------------------------------------------------------------------------
def drug_profiles(df) -> pd.DataFrame:
    last = df["initial_posting_date"].max()
    g = df.groupby("drug")
    p = pd.DataFrame({
        "n_records": g.size(), "n_companies": g["company_name"].nunique(), "n_forms": g["form_group"].nunique(),
        "share_active": g["is_shortage"].mean(), "median_duration_days": g["duration_days"].median(),
        "first_year": g["initial_posting_date"].min().dt.year, "last_year": g["initial_posting_date"].max().dt.year,
        "postings_last_24m": g["initial_posting_date"].apply(lambda s: int((s > last - pd.DateOffset(months=24)).sum())),
        "category": g["primary_category"].agg(lambda s: s.value_counts().index[0]),
        "form": g["form_group"].agg(lambda s: s.value_counts().index[0]),
        "brand": g["brand_name"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
    })
    p["span_years"] = p["last_year"] - p["first_year"] + 1
    p["median_duration_days"] = p["median_duration_days"].fillna(p["median_duration_days"].median())
    return p.reset_index()


def task_clustering(profiles, progress=_noop):
    cols = ["n_records", "n_companies", "n_forms", "share_active", "median_duration_days", "span_years", "postings_last_24m"]
    Z = profiles[cols].copy()
    for c in ["n_records", "n_companies", "median_duration_days", "postings_last_24m"]:
        Z[c] = np.log1p(Z[c])
    Zs = StandardScaler().fit_transform(Z)
    rows, best = [], dict(sil=-1)
    algos = {
        "K-Means": lambda k: KMeans(n_clusters=k, n_init=20, random_state=SEED),
        "Agglomerative (Ward)": lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "Gaussian Mixture": lambda k: GaussianMixture(n_components=k, n_init=3, random_state=SEED),
    }
    for a, mk in algos.items():
        for k in range(2, 9):
            progress(0.0, f"Clustering: {a}, k={k}")
            lab = mk(k).fit_predict(Zs)
            if len(np.unique(lab)) < 2:
                continue
            r = dict(algorithm=a, k=k, silhouette=float(silhouette_score(Zs, lab)),
                     davies_bouldin=float(davies_bouldin_score(Zs, lab)), calinski_harabasz=float(calinski_harabasz_score(Zs, lab)))
            rows.append(r)
            if k >= 3 and r["silhouette"] > best["sil"]:
                best = dict(sil=r["silhouette"], algo=a, k=k, labels=lab)
    prof = profiles.copy()
    prof["cluster"] = best["labels"]
    summ = prof.groupby("cluster")[cols].mean()
    z = (summ - summ.mean()) / summ.std(ddof=0).replace(0, 1)
    names = {}
    for c in summ.index:
        top = z.loc[c].sort_values(ascending=False)
        hi = top.index[0]
        tag = {"postings_last_24m": "Recently hot", "n_records": "Frequent repeaters", "n_companies": "Multi-manufacturer",
               "share_active": "Still unresolved", "median_duration_days": "Long-lasting", "span_years": "Long-running",
               "n_forms": "Many formulations"}[hi]
        if z.loc[c].max() < 0.3:
            tag = "Occasional / low-activity"
        names[c] = f"{tag}"
    # make names unique
    seen = {}
    for c in list(names):
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] = f"{names[c]} ({seen[names[c]]})"
    prof["segment"] = prof["cluster"].map(names)
    out = dict(task="Medicine segmentation by shortage behaviour", features=cols, n_medicines=len(prof),
               chosen=dict(algorithm=best["algo"], k=int(best["k"]), silhouette=float(best["sil"])),
               grid=rows, segment_summary=[dict(segment=names[c], n=int((prof.cluster == c).sum()),
                                                **{k: float(v) for k, v in summ.loc[c].items()}) for c in summ.index],
               note="k is chosen by silhouette among k>=3 (k=2 usually just splits 'one-off' from 'repeat' medicines).")
    return out, prof


# --------------------------------------------------------------------------------------------------
# 9. Orchestration
# --------------------------------------------------------------------------------------------------
def _clean_json(o):
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def run_all(raw: pd.DataFrame, outdir: str | Path, data_source: str, horizon: int = 6, progress=_noop) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    progress(0.01, "Cleaning data")
    df = clean(raw)
    df.drop(columns=["post_month"]).to_csv(out / "clean_data.csv", index=False)

    seg = lambda a, b: (lambda f, m: progress(a + (b - a) * f, m))   # noqa: E731
    report = dict(meta=dict(
        data_source=data_source, trained_at=datetime.now().isoformat(timespec="seconds"), n_raw=int(len(raw)),
        n_clean=int(len(df)), n_medicines=int(df["drug"].nunique()), first_posting=str(df["initial_posting_date"].min().date()),
        last_posting=str(df["initial_posting_date"].max().date()), active_share=float(df["is_shortage"].mean()),
        is_demo=("demo" in data_source.lower() or "sample" in data_source.lower())))

    report["early_warning"], art, scores, contrib = task_early_warning(df, horizon, seg(0.05, 0.50), out)
    joblib.dump(art, out / "early_warning_model.joblib")
    prof = drug_profiles(df)
    report["clustering"], prof = task_clustering(prof, seg(0.50, 0.52))
    scores = scores.merge(prof[["drug", "category", "form", "brand", "n_records", "share_active", "segment",
                                "median_duration_days"]], on="drug", how="left")
    thr = art["threshold"]
    elev = min(max(thr * 0.5, art["base_rate"]), thr)      # Elevated never starts below the average medicine
    scores["risk_level"] = np.where(scores.risk_probability >= thr, "High",
                                    np.where(scores.risk_probability >= elev, "Elevated", "Low"))
    scores["vs_baseline"] = scores.risk_probability / max(art["base_rate"], 1e-9)
    scores = scores.sort_values("risk_probability", ascending=False)
    scores.to_csv(out / "risk_scores.csv", index=False)
    if contrib is not None:
        contrib.to_csv(out / "risk_contributions.csv", index=False)
    prof.to_csv(out / "drug_profiles.csv", index=False)

    report["horizon_forecast"], hz = task_horizon_forecast(df, report["early_warning"]["best"], HORIZONS, seg(0.52, 0.64))
    if hz is not None:
        hz.to_csv(out / "horizon_scores.csv", index=False)
    elif (out / "horizon_scores.csv").exists():
        (out / "horizon_scores.csv").unlink()

    report["status_classification"] = task_status(df, seg(0.64, 0.80))
    report["duration"] = task_duration(df, seg(0.80, 0.88))
    fc_out, fc = task_forecast(df, 12, seg(0.88, 0.99))
    report["forecast"] = fc_out
    if fc is not None:
        fc[0].to_csv(out / "forecast_history.csv", index=False)
        fc[1].to_csv(out / "forecast_future.csv", index=False)
    report["meta"]["train_seconds"] = round(time.time() - t0, 1)
    with open(out / "metrics.json", "w") as f:
        json.dump(_clean_json(report), f, indent=1)
    progress(1.0, "Done")
    return report


def artifacts_ready(outdir: str | Path) -> bool:
    o = Path(outdir)
    return all((o / f).exists() for f in ["metrics.json", "risk_scores.csv", "clean_data.csv", "early_warning_model.joblib"])
