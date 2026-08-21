"""
Feature engineering for the Broadway weekly panel.

Leakage policy
--------------
Statistics.Capacity is not an independent measurement. It is an identity:

    capacity_pct == attendance / (theatre_seats * performances) * 100

Any same-week attendance, gross or gross-potential measure therefore
reconstructs the target almost exactly. This is the single largest leakage
risk in this dataset and the reason a careless model can score above 95% here.
Every builder below is annotated LEAKAGE-SAFE or SAME-WEEK-ONLY, and
assert_no_leakage() is the gate that model code must pass.

The chronological + production-grouped split in make_splits() is this dataset's
analogue of the customer_unique_id grouping warning in the project brief: a long
run such as Phantom of the Opera contributes hundreds of rows, so a random split
would put a production's own neighbouring weeks on both sides of the divide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    FORECAST_HORIZON_WEEKS,
    HORIZONS,
    CAPACITY_BAND_CUTS,
    CAPACITY_BAND_LABELS,
    CLOSURE_HORIZON_WEEKS,
    SAME_WEEK_FORBIDDEN,
    TRAIN_END,
    VAL_END,
)

LAGS = (1, 2, 3, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------------------
# Same-week derived quantities (EDA / dashboard only, never model inputs)
# ---------------------------------------------------------------------------
def add_same_week_derived(df: pd.DataFrame) -> pd.DataFrame:
    """SAME-WEEK-ONLY. Descriptive quantities for EDA and the dashboard.

    These are explicitly listed in config.SAME_WEEK_FORBIDDEN and must never be
    passed to a model that predicts capacity or gross.
    """
    out = df.copy()
    out["avg_ticket_price"] = out["gross"] / out["attendance"].replace(0, np.nan)
    out["seats_per_perf"] = out["attendance"] / out["performances"]
    return out


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. Known from the calendar alone, arbitrarily far in advance."""
    out = df.copy()
    out["is_holiday_week"] = out["iso_week"].isin([51, 52, 53])
    out["is_summer"] = out["month"].isin([7, 8])
    out["is_jan_slump"] = out["iso_week"].isin([4, 5, 6])
    out["is_tony_season"] = out["month"].eq(6)
    return out


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
def add_run_state(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. How far into its run a production is, as of this week.

    Uses only the production's own past, never its eventual run length.
    """
    out = df.sort_values(["production_id", "date"]).copy()
    grp = out.groupby("production_id", sort=False)
    out["run_week_index"] = grp.cumcount() + 1
    out["weeks_since_open"] = (
        (out["date"] - grp["date"].transform("min")).dt.days // 7
    )
    out["is_opening_month"] = out["run_week_index"] <= 4
    return out.sort_values(["date", "show_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------
def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. Prior-week values within the same production.

    A lag is only valid when the previous observation really is the previous
    calendar week; after a break in the run the lag is set to NaN rather than
    silently reaching across the gap.
    """
    out = df.sort_values(["production_id", "date"]).copy()
    grp = out.groupby("production_id", sort=False)

    for lag in LAGS:
        contiguous = grp["date"].diff(lag).dt.days.eq(7 * lag)
        for src, name in [
            ("capacity_pct", "capacity_lag"),
            ("gross", "gross_lag"),
            ("gross_potential_pct", "gp_lag"),
        ]:
            out[f"{name}{lag}"] = grp[src].shift(lag).where(contiguous)

    # Rolling summaries are horizon-specific: a model forecasting h weeks ahead
    # may only look at weeks t-h .. t-h-3, so each horizon gets its own window.
    for h in HORIZONS:
        window = [f"capacity_lag{i}" for i in range(h, h + 4)]
        out[f"cap_h{h}_roll_mean"] = out[window].mean(axis=1)
        out[f"cap_h{h}_roll_std"] = out[window].std(axis=1)
        out[f"cap_h{h}_trend"] = out[window[0]] - out[window[-1]]

    return out.sort_values(["date", "show_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------------
def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. State of the whole Broadway market in the *previous* week.

    n_shows_running is same-week but is a scheduling fact known before the week
    opens, not an outcome, so it is admissible.
    """
    out = df.copy()
    weekly = (
        out.groupby("date")
        .agg(n_shows_running=("production_id", "nunique"), market_gross=("gross", "sum"))
        .sort_index()
    )
    weekly["market_gross_lag1"] = weekly["market_gross"].shift(1)
    weekly["market_capacity_lag1"] = (
        out.groupby("date")["capacity_pct"].median().sort_index().shift(1)
    )
    return out.merge(
        weekly[["n_shows_running", "market_gross_lag1", "market_capacity_lag1"]],
        left_on="date",
        right_index=True,
        how="left",
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate 1 (upcoming-week demand) and Candidate 2 (closure risk) targets.

    Candidate 1 predicts a row's own capacity from weeks t-h and earlier, where
    h is the forecast horizon. A general manager commits marketing spend and
    TKTS allocation roughly a month before the week plays, so h = 4 is the point
    at which the decision is actually taken; numeric_features(h) enforces the
    corresponding lag window.

    Because the target is week t's own capacity, every same-week column in
    SAME_WEEK_FORBIDDEN is a genuine leak rather than a modelling preference:
    attendance, gross and performances for week t are all outcomes of week t and
    are unknown when the decision is made.
    """
    out = df.sort_values(["production_id", "date"]).copy()
    grp = out.groupby("production_id", sort=False)

    # -- Candidate 1: this week's capacity, regression and 3-band classification
    out["target_capacity"] = out["capacity_pct"]
    out["target_band"] = pd.cut(
        out["target_capacity"],
        bins=[-np.inf, CAPACITY_BAND_CUTS[0], CAPACITY_BAND_CUTS[1], np.inf],
        labels=CAPACITY_BAND_LABELS,
    )
    # Only rows with a genuine preceding week are predictable at all.
    out["is_predictable"] = out["capacity_lag1"].notna()

    # -- Candidate 2: does the production close within the horizon?
    # Right-censored: if the data ends before we could have observed a closure,
    # the label is unknowable and must stay NaN rather than default to "no".
    run_end = grp["date"].transform("max")
    weeks_to_close = (run_end - out["date"]).dt.days // 7
    data_end = out["date"].max()
    observable = ((data_end - out["date"]).dt.days // 7) >= CLOSURE_HORIZON_WEEKS
    out["weeks_to_close"] = weeks_to_close
    out["is_right_censored"] = ~observable
    out["target_closes_soon"] = (
        (weeks_to_close < CLOSURE_HORIZON_WEEKS).where(observable).astype("float")
    )

    return out.sort_values(["date", "show_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every builder in dependency order."""
    out = add_same_week_derived(df)
    out = add_calendar(out)
    out = add_run_state(out)
    out = add_lags(out)
    out = add_market_context(out)
    out = add_targets(out)
    return out


# Feature sets a model is allowed to see. Kept here rather than in a notebook so
# there is a single auditable list.
#
# Features that do not depend on the forecast horizon: calendar facts, run
# state, and static theatre attributes are all knowable arbitrarily far ahead.
HORIZON_FREE_NUMERIC = [
    "run_week_index", "weeks_since_open",
    "theatre_seats", "n_shows_running",
    "iso_week", "month", "year",
]
BOOLEAN_FEATURES = [
    "is_holiday_week", "is_summer", "is_jan_slump", "is_tony_season",
    "is_opening_month",
]
CATEGORICAL_FEATURES = ["show_type", "theatre"]


def numeric_features(horizon: int = FORECAST_HORIZON_WEEKS) -> list[str]:
    """Numeric predictors admissible when forecasting `horizon` weeks ahead.

    Only weeks t-horizon and earlier are visible, so the lag block starts at
    `horizon`, never at 1. Calling this with the wrong horizon is the easiest
    way to leak, which is why the lag list is generated rather than hand-typed.
    """
    lags = [f"capacity_lag{i}" for i in range(horizon, horizon + 4)]
    lags += [f"gp_lag{horizon}", f"gp_lag{horizon + 1}"]
    rolls = [f"cap_h{horizon}_roll_mean", f"cap_h{horizon}_roll_std",
             f"cap_h{horizon}_trend"]
    return lags + rolls + HORIZON_FREE_NUMERIC


def model_features(horizon: int = FORECAST_HORIZON_WEEKS) -> list[str]:
    """Full predictor list (numeric + boolean + categorical) for a horizon."""
    return numeric_features(horizon) + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


# Convenience aliases at the project's default horizon.
NUMERIC_FEATURES = numeric_features()
MODEL_FEATURES = model_features()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def assert_no_leakage(feature_names) -> None:
    """Refuse any feature set that contains a same-week outcome measure.

    Called by every modelling cell. See the module docstring for why capacity is
    mechanically reconstructible from the forbidden columns.
    """
    offenders = sorted(set(feature_names) & SAME_WEEK_FORBIDDEN)
    if offenders:
        raise ValueError(
            "Same-week outcome columns cannot be used as predictors because "
            "capacity_pct is an identity over them: " + ", ".join(offenders)
        )


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological split, then drop productions that straddle train and test.

    Chronology alone is not enough: a production running from 2012 to 2016 would
    place its own adjacent weeks in both train and test, so the test score would
    partly measure memorisation of that specific run.
    """
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END]

    straddlers = set(train["production_id"]) & set(test["production_id"])
    test_clean = test[~test["production_id"].isin(straddlers)]

    return {
        "train": train,
        "val": val,
        "test": test,
        "test_unseen_productions": test_clean,
        "n_straddling_productions": len(straddlers),
    }
