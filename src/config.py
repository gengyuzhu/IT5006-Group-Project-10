"""
Project-wide configuration for the IT5006 Phase 1 Olist analysis.

Everything a notebook, the dashboard or the report might quote as "a decision we
made" lives here, so there is exactly one authoritative copy of each constant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "olist"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# The nine tables named in the project brief.
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Row counts asserted on load, so a changed source file fails loudly rather than
# silently altering every number in the report.
EXPECTED_ROWS = {
    "orders": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "customers": 99_441,
    "sellers": 3_095,
    "products": 32_951,
    "geolocation": 1_000_163,
    "category_translation": 71,
}

# --------------------------------------------------------------------------
# Analysis window
# --------------------------------------------------------------------------
# The raw file spans 2016-09 to 2018-10, but the first four months are a
# pilot period with a handful of orders per month, and the final weeks are
# truncated mid-cycle. We model the fully-operational window and report the
# ramp-up separately as a coverage exhibit.
ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-08-31")

# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2018-04-30")
VAL_END = pd.Timestamp("2018-06-30")
# Test = 2018-07-01 .. ANALYSIS_END

# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------
# Candidate 1, framing B: an order is late if it reaches the customer after the
# delivery date promised at checkout.
# Candidate 2: a review of 1 or 2 stars counts as low satisfaction.
LOW_REVIEW_MAX_SCORE = 2
# Candidate 3: did the customer place another order within this many days?
REPEAT_HORIZON_DAYS = 30

# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
# Leakage is not one rule but one rule *per prediction moment*, because what is
# already known depends on when the model runs.
#
#   at_checkout - the order has just been placed. Nothing about fulfilment or
#                 the customer's eventual opinion exists yet. Used by the
#                 delivery candidates and by repeat-purchase.
#   at_delivery - the parcel has arrived and we are predicting the review that
#                 has not been written yet. The delivery outcome is now history
#                 and is admissible; the brief itself lists "delivery
#                 performance" among the features for this problem. Only the
#                 review and its timestamps remain forbidden.
#
# assert_no_leakage(features, regime) enforces the right set.
_REVIEW_COLS = {
    "review_score", "review_creation_date", "review_answer_timestamp",
    "is_low_review", "target_low_review",
}
_DELIVERY_COLS = {
    "order_delivered_customer_date", "order_delivered_carrier_date",
    "delivery_days", "is_late", "delay_vs_estimate_days",
    "target_delivery_days", "target_is_late",
}
_FULFILMENT_COLS = {
    # only known once fulfilment has started, i.e. after checkout
    "order_approved_at", "approval_hours", "shipping_limit_date", "order_status",
}

LEAKAGE_REGIMES = {
    "at_checkout": _REVIEW_COLS | _DELIVERY_COLS | _FULFILMENT_COLS,
    "at_delivery": _REVIEW_COLS | {"order_status"},
}

# Columns describing how delivery actually went. Forbidden at checkout,
# admissible once the parcel has arrived.
DELIVERY_OUTCOME_FEATURES = ["delivery_days", "delay_vs_estimate_days", "is_late"]

# --------------------------------------------------------------------------
# Plot theme
# --------------------------------------------------------------------------
FIG_WIDTH = 7.0          # inches; matches \linewidth at 1in margins on A4
FIG_HEIGHT = 4.0
PALETTE = {
    "primary": "#2F6F9F",
    "accent": "#C1666B",
    "gold": "#C08B2E",
    "green": "#5B8C5A",
    "muted": "#8C8C8C",
    "grid": "#DDDDDD",
    "late": "#C1666B",
    "ontime": "#2F6F9F",
}

# Brazilian regions, used to collapse 27 states into an interpretable grouping.
STATE_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North",
    "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast",
    "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast",
    "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West",
    "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}
