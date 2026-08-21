# %% [markdown]
# # 04 - Candidate Problem Exploration
#
# Phase 1 does not require a final problem choice. It requires evidence that the
# problems we are considering are **derivable, leakage-free, and carry enough
# signal to be worth modelling** - and honest evidence about how much signal.
#
# We evaluate three candidates against the project's scoping checklist:
#
# | | Candidate | Type | Stakeholder |
# |---|---|---|---|
# | 1 | Demand for a running production, 4 weeks ahead | regression **and** classification | theatre owner / general manager |
# | 2 | Will this production close within 8 weeks? | classification | producer / investor |
# | 3 | Opening-month weekly gross for a new production | regression | marketing / investor pro-forma |
#
# Every probe below is a **deliberately untuned baseline**: plain logistic or
# linear regression inside a leakage-proof pipeline. The purpose is to bound the
# achievable signal and to choose a horizon, not to produce a Phase 2 model.
# Where a naive rule is already strong, we say so.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_load import load_analysis
from features import (
    BOOLEAN_FEATURES, CATEGORICAL_FEATURES, assert_no_leakage, build_features,
    make_splits, model_features, numeric_features,
)
from viz import save_fig, use_report_style

use_report_style()
pd.set_option("display.width", 170)

H = config.FORECAST_HORIZON_WEEKS
df = build_features(load_analysis())
splits = make_splits(df)
print(f"{len(df):,} rows, {df['production_id'].nunique():,} productions, horizon = {H} weeks")


# %%
def build_pipe(model, numeric):
    """Preprocessing + model in one object, so no fold ever sees test statistics."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30),
             CATEGORICAL_FEATURES),
        ])),
        ("model", model),
    ])


def band_of(series):
    """Apply the demand-band cut points to any capacity series."""
    return pd.cut(series, bins=[-np.inf, *config.CAPACITY_BAND_CUTS, np.inf],
                  labels=config.CAPACITY_BAND_LABELS)


# %% [markdown]
# ## 4.1 The split, and why it is not random
#
# A random split would be badly optimistic here. *The Phantom of the Opera*
# alone contributes 994 weekly rows; a random split places its week 300 in train
# and its week 301 in test, so the model is graded on interpolating a run it has
# already memorised. This is the same failure mode the project brief warns about
# with `customer_id` versus `customer_unique_id` - the unit that must not
# straddle the split is the production, not the row.
#
# We split **chronologically** and additionally report a test subset containing
# only productions never seen in training.

# %%
for k in ["train", "val", "test", "test_unseen_productions"]:
    s = splits[k]
    print(f"{k:26s} {len(s):6,} rows | {s['production_id'].nunique():4d} productions | "
          f"{s['date'].min().date()} -> {s['date'].max().date()}")
print(f"\nproductions straddling train and test: {splits['n_straddling_productions']}")

# %% [markdown]
# ## 4.2 The leakage gate
#
# Notebook 03 showed that same-week attendance reconstructs capacity to within
# half a percentage point, because capacity *is* an identity over attendance,
# performances and seat count. Those columns are outcomes of the week we are
# trying to forecast, so they are unavailable at decision time.
#
# `assert_no_leakage` is the gate every modelling cell must pass. The cell below
# also quantifies what the gate buys us: adding one banned column inflates
# accuracy by more than 20 points of what would be a fictitious result.

# %%
FEATS = model_features(H)
assert_no_leakage(FEATS)
print(f"{len(FEATS)} features cleared the leakage gate at horizon {H}.")

try:
    assert_no_leakage(FEATS + ["Statistics.Attendance"])
except ValueError as exc:
    print(f"\nGate correctly rejects a same-week column:\n  {exc}")

# %%
tr = splits["train"].dropna(subset=["target_band", f"capacity_lag{H}"])
te = splits["test"].dropna(subset=["target_band", f"capacity_lag{H}"])
print(f"modelling rows: train {len(tr):,} | test {len(te):,}\n")

leak_cols = ["Statistics.Attendance", "Statistics.Performances"]
for label, feats, num in [
    ("leakage-safe (what we report)", FEATS, numeric_features(H)),
    ("+ same-week attendance", FEATS + leak_cols[:1], numeric_features(H) + leak_cols[:1]),
    ("+ same-week attendance & performances", FEATS + leak_cols, numeric_features(H) + leak_cols),
]:
    p = build_pipe(LogisticRegression(max_iter=2000, random_state=config.SEED), num)
    p.fit(tr[feats], tr["target_band"])
    a = (p.predict(te[feats]) == te["target_band"]).mean()
    flag = "" if "safe" in label else "   <- LEAKED, not reportable"
    print(f"{label:40s} accuracy = {a:.3f}{flag}")

# %% [markdown]
# ## 4.3 Candidate 1 - choosing the forecast horizon
#
# One underlying question ("how full will this production be?") answered as a
# regression for planners who need a number and as a three-band classification
# for operational triage. This satisfies the brief's requirement for both a
# classification and a regression task from a single coherent problem.
#
# The horizon is a **scoping decision, not a tuning knob**, so we make it here
# with evidence. At one week ahead the problem is nearly solved by persistence
# alone and a model adds nothing - but a one-week forecast is also operationally
# useless, because marketing buys and TKTS allocations cannot be changed at that
# notice. As the horizon lengthens, persistence decays and the model's
# contribution grows.

# %%
rows = []
for h in config.HORIZONS:
    feats_h, num_h = model_features(h), numeric_features(h)
    assert_no_leakage(feats_h)
    tr_h = splits["train"].dropna(subset=["target_band", f"capacity_lag{h}"])
    te_h = splits["test"].dropna(subset=["target_band", f"capacity_lag{h}"])

    clf_h = build_pipe(LogisticRegression(max_iter=2000, random_state=config.SEED), num_h)
    clf_h.fit(tr_h[feats_h], tr_h["target_band"])
    acc_h = (clf_h.predict(te_h[feats_h]) == te_h["target_band"]).mean()

    reg_h = build_pipe(LinearRegression(), num_h)
    reg_h.fit(tr_h[feats_h], tr_h["target_capacity"])
    pred_h = reg_h.predict(te_h[feats_h])

    rows.append({
        "horizon_weeks": h,
        "n_test": len(te_h),
        "majority_acc": te_h["target_band"].value_counts(normalize=True).max(),
        "persistence_acc": (band_of(te_h[f"capacity_lag{h}"]) == te_h["target_band"]).mean(),
        "logreg_acc": acc_h,
        "persistence_MAE": mean_absolute_error(te_h["target_capacity"], te_h[f"capacity_lag{h}"]),
        "linreg_MAE": mean_absolute_error(te_h["target_capacity"], pred_h),
    })

horizon_table = pd.DataFrame(rows).set_index("horizon_weeks")
horizon_table["acc_lift_vs_persistence"] = (
    horizon_table["logreg_acc"] - horizon_table["persistence_acc"]
)
horizon_table["MAE_gain_vs_persistence_pct"] = 100 * (
    1 - horizon_table["linreg_MAE"] / horizon_table["persistence_MAE"]
)
print(horizon_table.round(3).to_string())

# %% [markdown]
# The table settles the choice. At **h = 1** persistence scores 0.73 and the
# model cannot beat it: there is nothing to learn that last week's number does
# not already say. At **h = 4** persistence has decayed to 0.61 while the model
# holds 0.62 and the regression cuts MAE by around 6% - a modest but real gain,
# on the horizon at which the decision is genuinely made.
#
# We therefore scope Candidate 1 at a **four-week horizon** and carry the
# one-week case forward only as a reference point.

# %%
assert horizon_table.loc[4, "acc_lift_vs_persistence"] > 0
assert horizon_table.loc[1, "acc_lift_vs_persistence"] < horizon_table.loc[4, "acc_lift_vs_persistence"]

# %% [markdown]
# ### Band definition
#
# Bands are the empirical terciles of capacity computed **on the training years
# only**, so the cut points carry no information from the test period.

# %%
train_cuts = splits["train"]["capacity_pct"].quantile([1/3, 2/3]).round(0).values
print(f"tercile cuts from training data : {train_cuts}")
print(f"cuts stored in config           : {list(config.CAPACITY_BAND_CUTS)}")
assert np.allclose(train_cuts, config.CAPACITY_BAND_CUTS, atol=2.0)
print(f"\nband distribution (%):\n"
      f"{(df['target_band'].value_counts(normalize=True).sort_index() * 100).round(1).to_string()}")

# %% [markdown]
# ### Framing A - regression on capacity, four weeks out

# %%
y_te = te["target_capacity"]
persist_pred = te[f"capacity_lag{H}"].values
mean_pred = np.full(len(y_te), tr["target_capacity"].mean())

reg = build_pipe(LinearRegression(), numeric_features(H))
reg.fit(tr[FEATS], tr["target_capacity"])
model_pred = reg.predict(te[FEATS])

reg_results = pd.DataFrame([
    {"model": "Mean baseline", "pred": mean_pred},
    {"model": "Persistence baseline (capacity 4 weeks ago)", "pred": persist_pred},
    {"model": "Linear regression (untuned)", "pred": model_pred},
]).assign(
    MAE=lambda d: d["pred"].map(lambda p: mean_absolute_error(y_te, p)),
    RMSE=lambda d: d["pred"].map(lambda p: float(np.sqrt(((y_te - p) ** 2).mean()))),
    R2=lambda d: d["pred"].map(lambda p: r2_score(y_te, p)),
).drop(columns="pred").set_index("model").round(3)
print(reg_results.to_string())

gain = 100 * (1 - reg_results.loc["Linear regression (untuned)", "MAE"]
              / reg_results.loc["Persistence baseline (capacity 4 weeks ago)", "MAE"])
print(f"\nMAE improvement over persistence: {gain:+.1f}%")

# %% [markdown]
# Two points we will carry into the report. First, the mean baseline is far
# worse than persistence, which confirms that most of the predictable structure
# is *within* a production's own trajectory rather than across productions.
# Second, the gain over persistence is single-digit. A regression result on this
# target is meaningless without the persistence number printed beside it, so we
# will report both throughout Phase 2.

# %% [markdown]
# ### Framing B - three demand bands, four weeks out

# %%
dummy = DummyClassifier(strategy="most_frequent", random_state=config.SEED)
dummy.fit(tr[FEATS], tr["target_band"])
dummy_acc = (dummy.predict(te[FEATS]) == te["target_band"]).mean()
persist_acc = (band_of(te[f"capacity_lag{H}"]) == te["target_band"]).mean()

clf = build_pipe(LogisticRegression(max_iter=2000, random_state=config.SEED),
                 numeric_features(H))
clf.fit(tr[FEATS], tr["target_band"])
pred = clf.predict(te[FEATS])
acc = (pred == te["target_band"]).mean()

print(f"majority-class baseline   : {dummy_acc:.3f}")
print(f"persistence baseline      : {persist_acc:.3f}")
print(f"logistic regression       : {acc:.3f}")
print(f"balanced accuracy         : {balanced_accuracy_score(te['target_band'], pred):.3f}")
print(f"\n{classification_report(te['target_band'], pred, digits=3)}")

cm = confusion_matrix(te["target_band"], pred, labels=config.CAPACITY_BAND_LABELS)
print("confusion matrix (rows = actual):")
print(pd.DataFrame(cm, index=config.CAPACITY_BAND_LABELS,
                   columns=config.CAPACITY_BAND_LABELS).to_string())

# %% [markdown]
# The confusion matrix shows where the difficulty sits: `Low` and `High` are
# separated reliably, while `Mid` - productions sitting near the tercile
# boundaries - is where most errors land. That is an artefact of cutting a
# continuous quantity into bands, not a modelling failure, and it is a good
# argument for reporting the regression alongside the classifier.

# %%
unseen = splits["test_unseen_productions"].dropna(subset=["target_band", f"capacity_lag{H}"])
acc_unseen = (clf.predict(unseen[FEATS]) == unseen["target_band"]).mean()
print(f"accuracy on {len(unseen):,} rows from productions unseen in training: {acc_unseen:.3f}")
print(f"accuracy on the full test period                                   : {acc:.3f}")

# %% [markdown]
# ## 4.4 Candidate 2 - closure risk within 8 weeks
#
# Genuinely imbalanced, and right-censored: for the final 8 weeks of the data we
# cannot know whether a running production was about to close, so those rows are
# labelled `NaN` rather than defaulting to "no". Accuracy is the wrong headline
# metric here, and we show why.

# %%
lab = df.dropna(subset=["target_closes_soon"])
print(f"labelled rows          : {len(lab):,} of {len(df):,}")
print(f"right-censored, dropped: {int(df['is_right_censored'].sum()):,}")
print(f"positive rate          : {lab['target_closes_soon'].mean():.3f}")

cl_tr = splits["train"].dropna(subset=["target_closes_soon", f"capacity_lag{H}"])
cl_te = splits["test"].dropna(subset=["target_closes_soon", f"capacity_lag{H}"])

cl = build_pipe(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED),
    numeric_features(H),
)
cl.fit(cl_tr[FEATS], cl_tr["target_closes_soon"])
proba = cl.predict_proba(cl_te[FEATS])[:, 1]
pred_cl = (proba >= 0.5).astype(int)
y_cl = cl_te["target_closes_soon"].astype(int)
pr_auc = average_precision_score(y_cl, proba)

print(f"\naccuracy          : {(pred_cl == y_cl).mean():.3f}   "
      f"(always-predict-open baseline: {1 - y_cl.mean():.3f})")
print(f"balanced accuracy : {balanced_accuracy_score(y_cl, pred_cl):.3f}")
print(f"ROC-AUC           : {roc_auc_score(y_cl, proba):.3f}")
print(f"PR-AUC            : {pr_auc:.3f}   (no-skill floor = {y_cl.mean():.3f})")
minority_recall = classification_report(y_cl, pred_cl, output_dict=True)["1"]["recall"]
print(f"\n{classification_report(y_cl, pred_cl, digits=3, target_names=['stays open', 'closes <8w'])}")

# %% [markdown]
# This is the clearest illustration in the project of why accuracy misleads on
# imbalanced targets. An "always predict stays open" rule scores about 0.79 -
# higher than our model - while catching **zero** of the closures the model
# exists to find. Our classifier trades some accuracy for recall above 0.75 on
# the minority class, and PR-AUC against the no-skill floor is the honest
# summary of that trade.

# %% [markdown]
# ## 4.5 Candidate 3 - opening-month gross (cold start)
#
# The hardest of the three, because the lag features that carry most of the
# signal in Candidate 1 do not exist yet. We probe it to quantify *how much*
# harder, then record the result rather than forcing it.

# %%
cold = df[df["is_opening_month"]].copy()
COLD_NUMERIC = ["theatre_seats", "n_shows_running", "iso_week", "month", "year",
                "run_week_index"]
COLD_FEATURES = COLD_NUMERIC + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
assert_no_leakage(COLD_FEATURES)

cold_tr = cold[cold["date"] <= config.TRAIN_END]
cold_te = cold[cold["date"] > config.VAL_END]
print(f"opening-month rows: {len(cold):,} (train {len(cold_tr):,}, test {len(cold_te):,})")

cold_pipe = build_pipe(LinearRegression(), COLD_NUMERIC)
cold_pipe.fit(cold_tr[COLD_FEATURES], np.log(cold_tr["gross"]))
cold_pred = cold_pipe.predict(cold_te[COLD_FEATURES])
cold_true = np.log(cold_te["gross"])
cold_r2 = r2_score(cold_true, cold_pred)

dummy_r = DummyRegressor(strategy="mean").fit(cold_tr[COLD_FEATURES], np.log(cold_tr["gross"]))
print(f"\nR2, mean baseline : {r2_score(cold_true, dummy_r.predict(cold_te[COLD_FEATURES])):.3f}")
print(f"R2, linear model  : {cold_r2:.3f}   (target = log weekly gross)")
print(f"MAE (log gross)   : {mean_absolute_error(cold_true, cold_pred):.3f}")

# %% [markdown]
# The model explains under a third of the variance in opening-month gross, and
# essentially all of what it does capture is theatre size and seasonality. The
# information that would actually drive a new show's opening - advance sales,
# star casting, marketing spend, critical reception - is simply not in this
# dataset. Documenting that boundary is a more useful Phase 1 result than
# forcing a weak model.

# %%
# ---- Figure 12: candidate problem panel ------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(config.FIG_WIDTH, 2.7))

ax = axes[0]
ax.plot(horizon_table.index, horizon_table["persistence_acc"], marker="o", ms=4,
        color=config.PALETTE["accent"], label="persistence")
ax.plot(horizon_table.index, horizon_table["logreg_acc"], marker="s", ms=4,
        color=config.PALETTE["primary"], label="logistic regression")
ax.axhline(horizon_table["majority_acc"].mean(), color=config.PALETTE["muted"],
           ls=":", lw=1.1, label="majority class")
ax.axvline(H, color="black", ls="--", lw=0.9)
ax.annotate("chosen\nhorizon", xy=(H, 0.42), fontsize=6.5, ha="center")
ax.set_xticks(list(config.HORIZONS))
ax.set_xlabel("forecast horizon (weeks)"); ax.set_ylabel("test accuracy")
ax.set_ylim(0.25, 0.80)
ax.set_title("(a) Candidate 1\nsignal decays with horizon", fontsize=8)
ax.legend(fontsize=6.5, loc="lower left")

ax = axes[1]
bins = [0, 4, 8, 16, 26, 52, 80]
sub = lab[lab["run_week_index"] <= 80]
by_runweek = sub.groupby(pd.cut(sub["run_week_index"], bins=bins),
                         observed=True)["target_closes_soon"].mean() * 100
ax.bar(range(len(by_runweek)), by_runweek.values, color=config.PALETTE["primary"], alpha=0.85)
ax.set_xticks(range(len(by_runweek)))
ax.set_xticklabels(["1-4", "5-8", "9-16", "17-26", "27-52", "53-80"], fontsize=6.5, rotation=30)
ax.axhline(lab["target_closes_soon"].mean() * 100, color=config.PALETTE["accent"],
           ls="--", lw=1.1, label="overall rate")
ax.set_xlabel("week of run"); ax.set_ylabel("% closing within 8 weeks")
ax.set_title("(b) Candidate 2\nclosure risk by run stage", fontsize=8)
ax.legend(fontsize=6.5)

ax = axes[2]
ax.hist(df["target_capacity"].dropna(), bins=45, color=config.PALETTE["muted"], alpha=0.8)
lo, hi = config.CAPACITY_BAND_CUTS
for c in (lo, hi):
    ax.axvline(c, color=config.PALETTE["accent"], lw=1.2)
ymax = ax.get_ylim()[1]
# Shade the three bands so the labels do not have to carry the distinction.
for x0, x1, lbl, y in [(10, lo, "Low", 0.90), (lo, hi, "Mid", 0.78), (hi, 100, "High", 0.90)]:
    ax.annotate(lbl, xy=((x0 + x1) / 2, ymax * y), fontsize=7.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
ax.set_xlim(10, 101)
ax.set_xlabel("capacity utilisation (%)"); ax.set_ylabel("rows")
ax.set_title("(c) Three demand bands\n(training terciles)", fontsize=8)

fig.tight_layout()
print(save_fig(fig, "fig12_candidates"))

# %% [markdown]
# ## 4.6 Scoping checklist

# %%
checklist = pd.DataFrame([
    {"criterion": "Derivable target",
     "C1 demand (4 weeks)": "Yes - capacity is a column",
     "C2 closure risk": "Yes - from the last observed week of a run",
     "C3 opening gross": "Yes - gross in run weeks 1-4"},
    {"criterion": "Realistic features (no leakage)",
     "C1 demand (4 weeks)": "Yes - lags start at t-4; gate enforced in code",
     "C2 closure risk": "Yes - run state and lagged demand",
     "C3 opening gross": "Yes, but no lags exist yet"},
    {"criterion": "Sufficient signal",
     "C1 demand (4 weeks)": f"Yes - {acc:.2f} vs {dummy_acc:.2f} majority, {persist_acc:.2f} persistence",
     "C2 closure risk": f"Moderate - PR-AUC {pr_auc:.2f} vs {y_cl.mean():.2f} floor",
     "C3 opening gross": f"Weak - R2 {cold_r2:.2f}"},
    {"criterion": "Manageable imbalance",
     "C1 demand (4 weeks)": "Balanced by construction (terciles)",
     "C2 closure risk": f"Imbalanced ({lab['target_closes_soon'].mean():.0%}); class weights + PR-AUC",
     "C3 opening gross": "N/A - regression"},
    {"criterion": "Clear stakeholder",
     "C1 demand (4 weeks)": "Theatre GM - staffing, discounting",
     "C2 closure risk": "Producer - capital reallocation",
     "C3 opening gross": "Marketing - pro-forma"},
]).set_index("criterion")
print(checklist.to_string())

# %% [markdown]
# ## 4.7 Recommendation for Phase 2
#
# **Candidate 1 becomes the primary problem, scoped at a four-week horizon.**
# Its dual framing supplies both the classification and the regression task the
# brief requires from a single coherent question, the target is balanced by
# construction, and the stakeholder decision is concrete and time-bound.
#
# **Candidate 2 becomes the secondary problem.** It contributes what Candidate 1
# lacks: a genuinely imbalanced, right-censored target that forces the
# imbalance-aware metrics and class-weighting discussed in the literature review.
#
# **Candidate 3 is recorded as explored and deprioritised**, with the reason
# stated: the drivers of a new production's opening are not observable in this
# dataset.
#
# ### Expected Phase 2 performance, stated in advance
#
# We expect a tuned Candidate 1 classifier to land in the **low-to-mid 60s
# percent** against a 35% majority baseline and a 61% persistence baseline. That
# is a real and useful lift, and it is where the ceiling honestly sits: demand
# four weeks out is dominated by a production's own recent trajectory, and the
# residual variance is driven by factors this dataset does not observe -
# marketing spend, reviews, cast changes, tourism and weather.
#
# A model reporting 90%+ on this target would indicate leakage, not skill; the
# leaked variant in Section 4.2 is what that failure looks like.

# %%
summary = pd.DataFrame([
    {"candidate": f"C1 regression (capacity, {H}w ahead)",
     "baseline": f"persistence MAE {reg_results.loc['Persistence baseline (capacity 4 weeks ago)','MAE']:.2f}",
     "probe": f"MAE {reg_results.loc['Linear regression (untuned)','MAE']:.2f} ({gain:+.0f}%)",
     "verdict": "primary"},
    {"candidate": f"C1 classification (3 bands, {H}w ahead)",
     "baseline": f"majority {dummy_acc:.3f} / persistence {persist_acc:.3f}",
     "probe": f"accuracy {acc:.3f}",
     "verdict": "primary"},
    {"candidate": "C2 closure within 8 weeks",
     "baseline": f"PR-AUC floor {y_cl.mean():.3f}",
     "probe": f"PR-AUC {pr_auc:.3f}, minority recall {minority_recall:.2f}",
     "verdict": "secondary"},
    {"candidate": "C3 opening-month gross",
     "baseline": "R2 0.000",
     "probe": f"R2 {cold_r2:.3f}",
     "verdict": "deprioritised"},
]).set_index("candidate")
print(summary.to_string())

# %%
assert 0.50 < acc < 0.75, f"probe accuracy {acc:.3f} outside the plausible range"
assert acc > dummy_acc + 0.10, "probe should clearly beat the majority baseline"
assert acc > persist_acc, "at the chosen horizon the model should beat persistence"
assert pr_auc > 2 * y_cl.mean(), "closure model should clearly beat the no-skill floor"
print("\nCandidate-problem assertions passed.")
