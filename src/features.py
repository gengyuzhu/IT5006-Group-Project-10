"""
Feature engineering for the Olist order table.

Leakage policy
--------------
Prediction time for every candidate problem is **the moment the order is
placed**. That single decision determines what is admissible:

* `order_delivered_customer_date` and `order_delivered_carrier_date` *define*
  the delivery outcome, so they can never be predictors of it.
* `review_score` and its timestamps are written after the customer has lived
  through the order.
* `order_approved_at` and `shipping_limit_date` are only known once fulfilment
  has started, which is after checkout. The brief calls approval "usually safe";
  we exclude it anyway, because a model meant to run at checkout would not have
  it, and the cost of including it is unmeasurable optimism.
* `order_estimated_delivery_date` **is** admissible: the customer is shown that
  promise at checkout, so it exists before the outcome and is in fact the single
  most useful feature available.

`assert_no_leakage()` is the gate every modelling cell must pass, checked
against `config.POST_OUTCOME_FORBIDDEN`.

The split in `make_splits()` is chronological *and* grouped on
`customer_unique_id`. The brief warns that `customer_id` is regenerated per
order and is therefore not a person; splitting on rows would let the same buyer
appear on both sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DELIVERY_OUTCOME_FEATURES,
    LEAKAGE_REGIMES,
    LOW_REVIEW_MAX_SCORE,
    REPEAT_HORIZON_DAYS,
    TRAIN_END,
    VAL_END,
)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. Everything here is known the instant the order is placed."""
    out = df.copy()
    ts = out["order_purchase_timestamp"]
    out["purchase_year"] = ts.dt.year
    out["purchase_month"] = ts.dt.month
    out["purchase_dow"] = ts.dt.dayofweek
    out["purchase_hour"] = ts.dt.hour
    out["purchase_week"] = ts.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["purchase_dow"].isin([5, 6])
    out["is_business_hours"] = out["purchase_hour"].between(9, 18)
    # Black Friday falls in late November and is the platform's largest spike.
    out["is_nov_peak"] = (out["purchase_month"] == 11) & (ts.dt.day >= 20)
    return out


# ---------------------------------------------------------------------------
# Order composition
# ---------------------------------------------------------------------------
def add_basket(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. What the customer put in the basket and what it cost."""
    out = df.copy()
    price = out["total_price"].replace(0, np.nan)
    out["freight_ratio"] = out["total_freight"] / price
    out["price_per_item"] = out["total_price"] / out["n_items"].replace(0, np.nan)
    out["weight_per_item"] = out["total_weight_g"] / out["n_items"].replace(0, np.nan)
    out["is_multi_seller"] = out["n_sellers"] > 1
    out["log_price"] = np.log1p(out["total_price"])
    out["log_weight"] = np.log1p(out["total_weight_g"])
    out["log_distance"] = np.log1p(out["distance_km"])
    return out


# ---------------------------------------------------------------------------
# The promise made at checkout
# ---------------------------------------------------------------------------
def add_promise(df: pd.DataFrame) -> pd.DataFrame:
    """LEAKAGE-SAFE. The delivery window quoted to the customer at checkout.

    `estimated_days` is shown on the order confirmation, so it precedes the
    outcome. It also encodes whatever Olist's own routing model knew about the
    shipment, which is why it dominates the feature importance.
    """
    out = df.copy()
    out["estimated_days"] = out["estimated_days"].astype(float)
    out["long_promise"] = out["estimated_days"] > out["estimated_days"].median()
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate 1 (delivery, dual framing), 2 (review) and 3 (repeat purchase).

    Targets are NaN wherever the outcome is genuinely unobservable - an order
    that never arrived, a customer who left no review, or an order too close to
    the end of the data for a 90-day repeat window to have closed. None of these
    is defaulted to a negative label, which would manufacture signal.
    """
    out = df.copy()

    # -- Candidate 1: delivery performance, framed twice
    out["target_delivery_days"] = out["delivery_days"]
    out["target_is_late"] = out["is_late"]

    # -- Candidate 2: low satisfaction
    out["target_low_review"] = np.where(
        out["review_score"].notna(),
        (out["review_score"] <= LOW_REVIEW_MAX_SCORE).astype(float),
        np.nan,
    )

    # -- Candidate 3: does this buyer come back within the horizon?
    ts = out["order_purchase_timestamp"]
    nxt = (
        out.sort_values(["customer_unique_id", "order_purchase_timestamp"])
        .groupby("customer_unique_id")["order_purchase_timestamp"]
        .shift(-1)
        .reindex(out.index)
    )
    gap_days = (nxt - ts).dt.total_seconds() / 86400
    observable = (out["order_purchase_timestamp"].max() - ts).dt.days >= REPEAT_HORIZON_DAYS
    out["days_to_next_order"] = gap_days
    out["repeat_is_censored"] = ~observable
    out["target_repeat"] = np.where(
        observable, (gap_days <= REPEAT_HORIZON_DAYS).fillna(False).astype(float), np.nan
    )
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every builder in dependency order."""
    out = add_calendar(df)
    out = add_basket(out)
    out = add_promise(out)
    out = add_targets(out)
    return out


# ---------------------------------------------------------------------------
# Feature sets a model is allowed to see
# ---------------------------------------------------------------------------
# Deliberately free of exact duplicates. The first draft carried total_price
# *and* log_price *and* total_payment (which is price + freight), plus
# distance_km beside log_distance and purchase_week beside purchase_month.
# Ordinary least squares on that design matrix is rank-deficient and its
# coefficients explode; keeping one representative of each quantity is what
# makes a plain linear baseline meaningful at all.
NUMERIC_FEATURES = [
    "estimated_days",
    "n_items", "n_sellers",
    "log_price", "price_per_item",
    "total_freight", "freight_ratio",
    "log_weight", "max_volume_cm3", "total_photos",
    "log_distance",
    "max_installments",
    "purchase_month", "purchase_dow", "purchase_hour",
]
BOOLEAN_FEATURES = [
    "is_weekend", "is_business_hours", "is_nov_peak",
    "is_multi_seller", "same_state", "long_promise",
]
CATEGORICAL_FEATURES = [
    "customer_state", "seller_region", "lead_category", "lead_payment_type",
]

# purchase_year is deliberately absent: the split is chronological, so the test
# period contains a year the model never saw during training and a linear term
# on it extrapolates blindly.

MODEL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# The at-delivery regime may additionally see how the delivery actually went.
REVIEW_MODEL_FEATURES = MODEL_FEATURES + DELIVERY_OUTCOME_FEATURES


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def assert_no_leakage(feature_names, regime: str = "at_checkout") -> None:
    """Refuse any feature set containing a column unavailable at `regime`.

    `regime` is the moment the model would actually run - see the module
    docstring. Passing the wrong regime is itself the mistake this guard exists
    to catch, so an unknown name is an error rather than a silent pass.
    """
    if regime not in LEAKAGE_REGIMES:
        raise KeyError(
            f"Unknown leakage regime {regime!r}; expected one of "
            f"{sorted(LEAKAGE_REGIMES)}"
        )
    offenders = sorted(set(feature_names) & LEAKAGE_REGIMES[regime])
    if offenders:
        raise ValueError(
            f"At {regime.replace('_', ' ')} these columns are not yet "
            "observable, so they cannot be predictors: " + ", ".join(offenders)
        )


def make_splits(df: pd.DataFrame) -> dict[str, object]:
    """Chronological split, then remove customers who straddle train and test.

    Chronology alone is not enough. 2,997 buyers placed more than one order, so
    a purely time-based cut can still put one of their orders in training and
    another in test; the model would then be graded partly on a person it has
    already seen. This is the concrete form of the brief's customer_id versus
    customer_unique_id warning.
    """
    ts = df["order_purchase_timestamp"]
    train = df[ts <= TRAIN_END]
    val = df[(ts > TRAIN_END) & (ts <= VAL_END)]
    test = df[ts > VAL_END]

    straddlers = set(train["customer_unique_id"]) & set(test["customer_unique_id"])
    test_clean = test[~test["customer_unique_id"].isin(straddlers)]

    return {
        "train": train,
        "val": val,
        "test": test,
        "test_unseen_customers": test_clean,
        "n_straddling_customers": len(straddlers),
    }
