"""
Regenerate every number quoted in the Phase 1 report.

The report is written by hand, but no figure in it is typed from memory: this
script recomputes each one and writes report/stats.txt. Before submitting, run

    python src/report_stats.py

and diff the output against the manuscript. Any drift is a bug in one or the
other.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, mean_absolute_error,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_load import clean, load_analysis, load_raw, theatre_seat_counts
from features import (
    CATEGORICAL_FEATURES, BOOLEAN_FEATURES, add_same_week_derived,
    build_features, make_splits, model_features, numeric_features,
)

H = config.FORECAST_HORIZON_WEEKS
S: dict[str, object] = {}


def pipe(model, numeric):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                              ("s", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30),
             CATEGORICAL_FEATURES),
        ])),
        ("model", model),
    ])


def band_of(s):
    return pd.cut(s, bins=[-np.inf, *config.CAPACITY_BAND_CUTS, np.inf],
                  labels=config.CAPACITY_BAND_LABELS)


# --------------------------------------------------------------------------
# Dataset shape and quality
# --------------------------------------------------------------------------
raw = load_raw()
full = add_same_week_derived(clean(raw))
full = full.merge(theatre_seat_counts(full), left_on="theatre", right_index=True, how="left")
df = build_features(load_analysis())

S["raw_rows"] = len(raw)
S["raw_cols"] = raw.shape[1]
S["date_min"] = str(raw["date"].min().date())
S["date_max"] = str(raw["date"].max().date())
S["n_weeks"] = int(raw["date"].nunique())
S["n_shows"] = int(raw["Show.Name"].nunique())
S["n_theatres"] = int(raw["Show.Theatre"].nunique())
S["n_productions_full"] = int(full["production_id"].nunique())
S["analysis_rows"] = len(df)
S["analysis_productions"] = int(df["production_id"].nunique())
S["median_shows_per_week"] = int(full.groupby("date").size().median())

S["perf_zero"] = int(full["perf_missing"].sum())
S["perf_zero_pct"] = round(100 * full["perf_missing"].mean(), 1)
S["gp_zero"] = int(full["gp_missing"].sum())
S["gp_zero_pct"] = round(100 * full["gp_missing"].mean(), 1)
S["gp_zero_1996_pct"] = round(100 * full.loc[full["year"] == 1996, "gp_missing"].mean(), 0)
S["capacity_censored"] = int(full["capacity_censored"].sum())
S["gp_over_100"] = int((full["gross_potential_pct"] > 100).sum())

dates = pd.Series(sorted(full["date"].unique()))
gaps = dates.diff().dt.days
S["weeks_missing_total"] = int(((gaps[gaps > 7] // 7) - 1).sum())

types = full["show_type"].value_counts(normalize=True).mul(100).round(1)
S["pct_musical"] = float(types["Musical"])
S["pct_play"] = float(types["Play"])
S["pct_special"] = float(types["Special"])

# --------------------------------------------------------------------------
# Capacity identity validation
# --------------------------------------------------------------------------
PUBLISHED = {"Majestic": 1645, "Imperial": 1417, "Gershwin": 1933, "Broadway": 1761,
             "Minskoff": 1710, "Palace": 1743, "Ambassador": 1125, "Lunt-Fontanne": 1509}
seats = theatre_seat_counts(full)
err = (seats.reindex(PUBLISHED) - pd.Series(PUBLISHED)).abs() / pd.Series(PUBLISHED) * 100
S["seat_recovery_median_err_pct"] = round(float(err.median()), 1)
S["seat_recovery_max_err_pct"] = round(float(err.max()), 1)

chk = full.dropna(subset=["performances", "theatre_seats"]).copy()
recon = 100 * chk["attendance"] / (chk["theatre_seats"] * chk["performances"])
S["capacity_reconstruction_median_err_pp"] = round(float((recon - chk["capacity_pct"]).abs().median()), 2)

# --------------------------------------------------------------------------
# Temporal
# --------------------------------------------------------------------------
market = full.groupby("date").agg(gross=("gross", "sum"), productions=("production_id", "nunique"))
S["gross_911_before_m"] = round(float(market.loc["2001-09-09", "gross"]) / 1e6, 2)
S["gross_911_week_m"] = round(float(market.loc["2001-09-16", "gross"]) / 1e6, 2)
S["drop_911_pct"] = round(100 * (market.loc["2001-09-16", "gross"] / market.loc["2001-09-09", "gross"] - 1), 0)
S["strike_productions_before"] = int(market.loc["2007-11-04", "productions"])
S["strike_productions"] = int(market.loc["2007-11-18", "productions"])
S["strike_gross_before_m"] = round(float(market.loc["2007-11-04", "gross"]) / 1e6, 1)
S["strike_gross_m"] = round(float(market.loc["2007-11-18", "gross"]) / 1e6, 1)

by_week = df.groupby("iso_week")["capacity_pct"].median()
by_month = df.groupby("month")["capacity_pct"].median()
S["cap_week_52_53"] = round(float(by_week.loc[[52, 53]].mean()), 0)
S["cap_week_36_44"] = round(float(by_week.loc[[36, 44]].mean()), 0)
S["week_seasonal_contrast_pp"] = round(S["cap_week_52_53"] - S["cap_week_36_44"], 0)
S["month_dec_sep_contrast_pp"] = round(float(by_month.loc[12] - by_month.loc[9]), 1)

yearly = df.groupby("year").agg(price=("avg_ticket_price", "median"), cap=("capacity_pct", "median"))
S["price_1996"] = round(float(yearly.loc[1996, "price"]), 1)
S["price_2015"] = round(float(yearly.loc[2015, "price"]), 1)
S["price_growth_pct"] = round(100 * (yearly.loc[2015, "price"] / yearly.loc[1996, "price"] - 1), 0)
S["cap_1996"] = round(float(yearly.loc[1996, "cap"]), 0)
S["cap_2015"] = round(float(yearly.loc[2015, "cap"]), 0)

# --------------------------------------------------------------------------
# Lifecycles
# --------------------------------------------------------------------------
runs = df.groupby(["production_id", "show_type"]).agg(weeks=("date", "size"), end=("date", "max")).reset_index()
S["run_median_weeks"] = int(runs["weeks"].median())
S["run_p90_weeks"] = int(runs["weeks"].quantile(0.9))
S["run_max_weeks"] = int(runs["weeks"].max())
S["run_censored"] = int((runs["end"] >= df["date"].max() - pd.Timedelta(weeks=1)).sum())
S["autocorr_lag1"] = round(float(df["capacity_pct"].corr(df.groupby("production_id")["capacity_pct"].shift(1))), 3)

# --------------------------------------------------------------------------
# Candidate probes
# --------------------------------------------------------------------------
splits = make_splits(df)
S["train_rows"] = len(splits["train"])
S["test_rows"] = len(splits["test"])
S["straddling_productions"] = splits["n_straddling_productions"]

FEATS = model_features(H)
tr = splits["train"].dropna(subset=["target_band", f"capacity_lag{H}"])
te = splits["test"].dropna(subset=["target_band", f"capacity_lag{H}"])
S["n_features"] = len(FEATS)
S["model_train_rows"] = len(tr)
S["model_test_rows"] = len(te)

# horizon sweep
sweep = {}
for h in config.HORIZONS:
    f_h, n_h = model_features(h), numeric_features(h)
    a = splits["train"].dropna(subset=["target_band", f"capacity_lag{h}"])
    b = splits["test"].dropna(subset=["target_band", f"capacity_lag{h}"])
    c = pipe(LogisticRegression(max_iter=2000, random_state=config.SEED), n_h).fit(a[f_h], a["target_band"])
    r = pipe(LinearRegression(), n_h).fit(a[f_h], a["target_capacity"])
    sweep[h] = {
        "persistence_acc": round(float((band_of(b[f"capacity_lag{h}"]) == b["target_band"]).mean()), 3),
        "logreg_acc": round(float((c.predict(b[f_h]) == b["target_band"]).mean()), 3),
        "persistence_mae": round(float(mean_absolute_error(b["target_capacity"], b[f"capacity_lag{h}"])), 2),
        "linreg_mae": round(float(mean_absolute_error(b["target_capacity"], r.predict(b[f_h]))), 2),
    }
S["horizon_sweep"] = sweep

# leakage demonstration
leak = ["Statistics.Attendance", "Statistics.Performances"]
for tag, extra in [("safe", []), ("leak1", leak[:1]), ("leak2", leak)]:
    p = pipe(LogisticRegression(max_iter=2000, random_state=config.SEED),
             numeric_features(H) + extra).fit(tr[FEATS + extra], tr["target_band"])
    S[f"acc_{tag}"] = round(float((p.predict(te[FEATS + extra]) == te["target_band"]).mean()), 3)

# candidate 1
dummy = DummyClassifier(strategy="most_frequent").fit(tr[FEATS], tr["target_band"])
S["c1_majority_acc"] = round(float((dummy.predict(te[FEATS]) == te["target_band"]).mean()), 3)
S["c1_persistence_acc"] = sweep[H]["persistence_acc"]
clf = pipe(LogisticRegression(max_iter=2000, random_state=config.SEED), numeric_features(H))
clf.fit(tr[FEATS], tr["target_band"])
pred = clf.predict(te[FEATS])
S["c1_logreg_acc"] = round(float((pred == te["target_band"]).mean()), 3)
S["c1_balanced_acc"] = round(float(balanced_accuracy_score(te["target_band"], pred)), 3)

unseen = splits["test_unseen_productions"].dropna(subset=["target_band", f"capacity_lag{H}"])
S["c1_acc_unseen_productions"] = round(float((clf.predict(unseen[FEATS]) == unseen["target_band"]).mean()), 3)
S["c1_unseen_rows"] = len(unseen)

reg = pipe(LinearRegression(), numeric_features(H)).fit(tr[FEATS], tr["target_capacity"])
rp = reg.predict(te[FEATS])
S["c1_mean_mae"] = round(float(mean_absolute_error(te["target_capacity"],
                          np.full(len(te), tr["target_capacity"].mean()))), 2)
S["c1_persistence_mae"] = sweep[H]["persistence_mae"]
S["c1_linreg_mae"] = round(float(mean_absolute_error(te["target_capacity"], rp)), 2)
S["c1_linreg_r2"] = round(float(r2_score(te["target_capacity"], rp)), 3)
S["c1_persistence_r2"] = round(float(r2_score(te["target_capacity"], te[f"capacity_lag{H}"])), 3)
S["c1_mae_gain_pct"] = round(100 * (1 - S["c1_linreg_mae"] / S["c1_persistence_mae"]), 1)

# candidate 2
lab = df.dropna(subset=["target_closes_soon"])
S["c2_positive_rate"] = round(float(lab["target_closes_soon"].mean()), 3)
S["c2_censored_rows"] = int(df["is_right_censored"].sum())
c_tr = splits["train"].dropna(subset=["target_closes_soon", f"capacity_lag{H}"])
c_te = splits["test"].dropna(subset=["target_closes_soon", f"capacity_lag{H}"])
cl = pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED),
          numeric_features(H)).fit(c_tr[FEATS], c_tr["target_closes_soon"])
proba = cl.predict_proba(c_te[FEATS])[:, 1]
pc = (proba >= 0.5).astype(int)
yc = c_te["target_closes_soon"].astype(int)
S["c2_test_positive_rate"] = round(float(yc.mean()), 3)
S["c2_accuracy"] = round(float((pc == yc).mean()), 3)
S["c2_always_open_acc"] = round(float(1 - yc.mean()), 3)
S["c2_balanced_acc"] = round(float(balanced_accuracy_score(yc, pc)), 3)
S["c2_roc_auc"] = round(float(roc_auc_score(yc, proba)), 3)
S["c2_pr_auc"] = round(float(average_precision_score(yc, proba)), 3)
S["c2_minority_recall"] = round(float(recall_score(yc, pc)), 3)

# candidate 3
cold = df[df["is_opening_month"]]
CN = ["theatre_seats", "n_shows_running", "iso_week", "month", "year", "run_week_index"]
CF = CN + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
c3tr, c3te = cold[cold["date"] <= config.TRAIN_END], cold[cold["date"] > config.VAL_END]
c3 = pipe(LinearRegression(), CN).fit(c3tr[CF], np.log(c3tr["gross"]))
S["c3_rows"] = len(cold)
S["c3_r2"] = round(float(r2_score(np.log(c3te["gross"]), c3.predict(c3te[CF]))), 3)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    out = config.PROJECT_ROOT / "report" / "stats.txt"
    lines = [f"{k:38s} {json.dumps(v) if isinstance(v, dict) else v}" for k, v in S.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
