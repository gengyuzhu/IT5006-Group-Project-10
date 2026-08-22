"""
Build the slim artifact the deployed dashboard reads.

The nine raw Olist CSVs total 121 MB, which is too much to carry in a repository
and too slow to join on every cold start of a hosted app. This script runs the
same `data_load` + `features` pipeline the notebooks and the report use, keeps
only the columns the dashboard actually plots, downcasts them, and writes a
4 MB parquet file.

    python src/build_dashboard_data.py

Re-run it whenever the cleaning or feature code changes, so the dashboard cannot
drift away from the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from data_load import load_analysis
from features import build_features

# Only what the five dashboard tabs plot. Deliberately excludes the 32-character
# hash identifiers, which alone accounted for three quarters of the file size.
DASHBOARD_COLUMNS = [
    "order_purchase_timestamp", "order_status",
    "customer_state", "customer_region", "seller_region", "lead_seller_state",
    "lead_category", "lead_payment_type",
    "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
    "max_installments", "total_weight_g", "distance_km", "same_state",
    "delivery_days", "is_late", "estimated_days", "delay_vs_estimate_days",
    "review_score",
    "purchase_month", "purchase_dow", "purchase_hour", "is_weekend",
    "target_low_review", "target_repeat", "repeat_is_censored",
]
CATEGORICAL = [
    "customer_state", "customer_region", "seller_region", "lead_seller_state",
    "lead_category", "lead_payment_type", "order_status",
]
FLOAT32 = [
    "total_price", "total_freight", "freight_ratio", "total_weight_g",
    "distance_km", "delivery_days", "is_late", "estimated_days",
    "delay_vs_estimate_days", "target_low_review", "target_repeat",
    "n_items", "n_sellers", "max_installments", "purchase_month",
    "purchase_dow", "purchase_hour", "review_score",
]

OUTPUT = config.DATA_PROCESSED / "orders_dashboard.parquet"


def main() -> None:
    df = build_features(load_analysis())

    missing = [c for c in DASHBOARD_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature pipeline no longer produces: {missing}")

    slim = df[DASHBOARD_COLUMNS].copy()
    for col in CATEGORICAL:
        slim[col] = slim[col].astype("category")
    for col in FLOAT32:
        slim[col] = slim[col].astype("float32")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(OUTPUT, compression="zstd", index=False)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT}")
    print(f"  {len(slim):,} rows x {slim.shape[1]} columns, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
