"""
Single entry point for the Broadway weekly grosses data.

Every notebook, the dashboard and the report read the data through this module,
so the numbers quoted in the PDF and the numbers rendered in the dashboard can
never drift apart.

Pipeline
--------
    load_raw()       read CSV, parse dates, assert the raw data contract
    clean()          apply the documented cleaning decisions (CLEANING_DECISIONS)
    load_analysis()  load_raw -> clean -> restrict to the analysis window
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (
    ANALYSIS_END,
    ANALYSIS_START,
    DATA_RAW,
    EXPECTED_RAW_COLS,
    EXPECTED_RAW_ROWS,
    NATURAL_KEY,
    RUN_BLOCK_GAP_WEEKS,
)

# ---------------------------------------------------------------------------
# The cleaning contract. Section 2.2 of the report is generated from this dict,
# so editing a decision here updates the report's audit table too.
# ---------------------------------------------------------------------------
CLEANING_DECISIONS: dict[str, dict[str, str]] = {
    "coverage_gaps": {
        "issue": "Two large holes in the weekly series: 29 weeks missing between "
                 "1990-08-26 and 1991-03-24, and 42 weeks between 1991-05-26 and "
                 "1992-03-22, plus a 3-week and a 1-week gap.",
        "evidence": "1,281 distinct week-ending dates span 1,357 calendar weeks.",
        "decision": "Restrict the main analysis to 1996-01-01 onwards; report the "
                    "1990-95 sparsity as a separate data-quality exhibit.",
    },
    "performances_zero": {
        "issue": "Statistics.Performances == 0 while Statistics.Attendance > 0, "
                 "which is structurally impossible.",
        "evidence": "2,309 rows (7.9%), concentrated in the earlier years.",
        "decision": "Recode 0 to NaN and raise the boolean flag perf_missing. "
                    "Never impute; rows are excluded from any statistic that "
                    "divides by performances.",
    },
    "gross_potential_zero": {
        "issue": "Statistics.Gross Potential == 0, i.e. missing values coded as "
                 "zero rather than as blanks.",
        "evidence": "1,918 rows (6.6%); 837 of them fall in 1996 alone and a "
                    "further 71 in 2008, so the zeros cluster by reporting era "
                    "rather than by show.",
        "decision": "Recode 0 to NaN and raise the boolean flag gp_missing.",
    },
    "capacity_censored": {
        "issue": "Statistics.Capacity is capped at 100 and cannot express the "
                 "standing-room sales that push Gross Potential above 100%.",
        "evidence": "1,433 rows have Gross Potential > 100 while Capacity <= 100.",
        "decision": "Treat 100 as right-censored; flag with capacity_censored and "
                    "state the censoring wherever capacity is modelled.",
    },
    "nominal_dollars": {
        "issue": "Statistics.Gross is in nominal USD across 26 years, so any "
                 "level comparison across time confounds real demand with "
                 "inflation and with ticket-price growth.",
        "evidence": "Median average ticket price rises from $24.4 (1990) to "
                    "$93.1 (2015).",
        "decision": "Base the analysis on scale-free measures (Capacity %, Gross "
                    "Potential %) and carry Date.Year as an explicit feature; no "
                    "external deflator is used in the main analysis.",
    },
    "title_case_artifacts": {
        "issue": "Show titles were upper-cased word-by-word, turning possessive "
                 "apostrophes into 'S (for example, A Doll'S House).",
        "evidence": "42 of the 820 distinct titles are affected.",
        "decision": "Normalise 'S to 's into show_name; keep the untouched "
                    "original in show_name_raw.",
    },
    "multi_theatre_shows": {
        "issue": "The same title can run in more than one theatre (transfers and "
                 "revivals), so Show.Name alone is not a production identifier.",
        "evidence": "27 titles appear in more than one theatre.",
        "decision": "Identify a production by (show, theatre, run_block), where "
                    "run_block increments after any absence longer than "
                    f"{RUN_BLOCK_GAP_WEEKS} weeks.",
    },
}

_APOSTROPHE_S = re.compile(r"'S\b")


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------
def load_raw(path=DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV and assert the data contract we documented in Phase 1.

    Raises AssertionError if the file we were given ever changes shape, which
    protects every downstream number in the report.
    """
    df = pd.read_csv(path)

    assert df.shape == (EXPECTED_RAW_ROWS, EXPECTED_RAW_COLS), (
        f"Expected {EXPECTED_RAW_ROWS} x {EXPECTED_RAW_COLS}, got {df.shape}"
    )
    assert not df.isna().any().any(), "Raw file is documented as having no NaNs"
    assert not df.duplicated().any(), "Raw file is documented as having no dupes"
    assert not df.duplicated(subset=NATURAL_KEY).any(), (
        f"{NATURAL_KEY} is documented as the natural key"
    )

    df["date"] = pd.to_datetime(df["Date.Full"], format="%m/%d/%Y")
    assert (df["date"].dt.dayofweek == 6).all(), (
        "Every observation should be a week ending on a Sunday"
    )
    return df


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CLEANING_DECISIONS and derive the production identifier.

    Returns a tidy frame with snake_case analysis columns alongside the original
    Statistics.* columns, so an auditor can always trace a value back.
    """
    out = df.copy()

    # -- title normalisation -------------------------------------------------
    out["show_name_raw"] = out["Show.Name"]
    out["show_name"] = out["Show.Name"].str.replace(_APOSTROPHE_S, "'s", regex=True)
    out["theatre"] = out["Show.Theatre"]
    out["show_type"] = out["Show.Type"]

    # -- zero-coded missingness ---------------------------------------------
    out["perf_missing"] = out["Statistics.Performances"].eq(0)
    out["gp_missing"] = out["Statistics.Gross Potential"].eq(0)
    out["capacity_censored"] = out["Statistics.Capacity"].eq(100)

    out["performances"] = out["Statistics.Performances"].replace(0, np.nan)
    out["gross_potential_pct"] = out["Statistics.Gross Potential"].replace(0, np.nan)
    out["capacity_pct"] = out["Statistics.Capacity"].astype(float)
    out["attendance"] = out["Statistics.Attendance"].astype(float)
    out["gross"] = out["Statistics.Gross"].astype(float)

    # -- calendar ------------------------------------------------------------
    iso = out["date"].dt.isocalendar()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["iso_week"] = iso["week"].astype(int)

    # -- production identity -------------------------------------------------
    out = out.sort_values(["show_name", "theatre", "date"]).reset_index(drop=True)
    gap_weeks = out.groupby(["show_name", "theatre"], sort=False)["date"].diff().dt.days.div(7)
    new_block = gap_weeks.isna() | (gap_weeks > RUN_BLOCK_GAP_WEEKS)
    out["run_block"] = (
        new_block.groupby([out["show_name"], out["theatre"]]).cumsum().astype(int)
    )
    out["production_id"] = (
        out["show_name"] + " @ " + out["theatre"] + " #" + out["run_block"].astype(str)
    )

    return out.sort_values(["date", "show_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Theatre seat counts
# ---------------------------------------------------------------------------
def theatre_seat_counts(df: pd.DataFrame) -> pd.Series:
    """Recover each theatre's seat count by inverting the capacity identity.

    Statistics.Capacity is the percentage of seats sold, so

        seats == attendance / performances / (capacity_pct / 100)

    Taking the per-theatre median of that quantity recovers seat counts that
    match the published figures closely (Majestic 1608 vs 1645 actual, Imperial
    1419 vs 1417 actual). This both validates our reading of the column and
    gives us a legitimate, leakage-free static feature.
    """
    usable = df[(df["performances"] > 0) & (df["capacity_pct"] > 0)]
    implied = (
        usable["attendance"] / usable["performances"] / (usable["capacity_pct"] / 100)
    )
    return (
        implied.groupby(usable["theatre"])
        .median()
        .round()
        .astype(int)
        .rename("theatre_seats")
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def load_analysis(path=DATA_RAW) -> pd.DataFrame:
    """Load, clean, attach theatre seat counts, restrict to the analysis window."""
    df = clean(load_raw(path))
    df = df.merge(
        theatre_seat_counts(df), left_on="theatre", right_index=True, how="left"
    )
    mask = df["date"].between(ANALYSIS_START, ANALYSIS_END)
    return df.loc[mask].reset_index(drop=True)


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column audit table: distinct values and zero-coded missingness."""
    rows = []
    for col in ["attendance", "capacity_pct", "gross", "gross_potential_pct", "performances"]:
        s = df[col]
        rows.append(
            {
                "column": col,
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(100 * s.isna().mean(), 2),
                "n_unique": int(s.nunique()),
                "min": round(float(s.min()), 1),
                "median": round(float(s.median()), 1),
                "max": round(float(s.max()), 1),
            }
        )
    return pd.DataFrame(rows)
