"""
Project-wide configuration for the IT5006 Phase 1 Broadway analysis.

Everything that a downstream notebook, the dashboard, or the report might need
to quote as a "decision we made" lives here, so that there is exactly one
authoritative copy of each constant.
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
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "broadway_clean.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Raw-file expectations (asserted in data_load.load_raw)
# --------------------------------------------------------------------------
EXPECTED_RAW_ROWS = 29_167
EXPECTED_RAW_COLS = 12
NATURAL_KEY = ["Date.Full", "Show.Name", "Show.Theatre"]

# --------------------------------------------------------------------------
# Analysis window
# --------------------------------------------------------------------------
# The raw file starts in Aug 1990 but has two very large coverage holes
# (29 weeks in 1990-91 and 42 weeks in 1991-92) plus a thin ramp-up through
# 1995. We restrict the main analysis to the fully-covered era and report the
# early sparsity separately as a data-quality exhibit.
ANALYSIS_START = pd.Timestamp("1996-01-01")
ANALYSIS_END = pd.Timestamp("2016-08-14")

# --------------------------------------------------------------------------
# Chronological split (used by the Phase 1 baseline probes and by Phase 2)
# --------------------------------------------------------------------------
TRAIN_END = pd.Timestamp("2013-12-31")
VAL_END = pd.Timestamp("2014-12-31")
# Test = everything after VAL_END, i.e. 2015-01-01 .. 2016-08-14

# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------
# Demand bands for the classification framing of Candidate 1. Cuts are the
# empirical terciles of Statistics.Capacity over the analysis window; they are
# recomputed and asserted in notebook 04 so the report never quotes a stale
# number.
CAPACITY_BAND_CUTS = (73.0, 89.0)
CAPACITY_BAND_LABELS = ["Low", "Mid", "High"]

# How far ahead Candidate 1 forecasts. A one-week horizon is nearly free -
# persistence alone scores 0.73 - but it is also operationally useless: a
# marketing buy or a TKTS allocation cannot be changed with a week's notice.
# Four weeks is the horizon at which the decision is actually made, and it is
# where a model earns its keep, because persistence has decayed to 0.61 by then.
# Features for horizon h use weeks t-h back to t-h-3 only.
FORECAST_HORIZON_WEEKS = 4
HORIZONS = (1, 2, 4)

# Candidate 2: does this production close within N weeks?
CLOSURE_HORIZON_WEEKS = 8

# A production is treated as a new run block if it disappears from the
# listings for more than this many weeks and then returns.
RUN_BLOCK_GAP_WEEKS = 4

# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------
# Statistics.Capacity is a deterministic function of attendance, performances
# and the theatre's seat count:
#     Capacity% == Attendance / (theatre_seats * Performances) * 100
# so any same-week attendance/gross measure trivially reconstructs the target.
# features.assert_no_leakage() refuses to let these reach a model.
SAME_WEEK_FORBIDDEN = {
    "Statistics.Attendance",
    "Statistics.Gross",
    "Statistics.Gross Potential",
    "Statistics.Capacity",
    "Statistics.Performances",
    "avg_ticket_price",
    "seats_per_perf",
}

# --------------------------------------------------------------------------
# Real-world events visible in the data (used to annotate temporal figures)
# --------------------------------------------------------------------------
EVENTS = [
    (pd.Timestamp("2001-09-16"), "9/11"),
    (pd.Timestamp("2007-11-18"), "Stagehand strike"),
    (pd.Timestamp("2008-10-05"), "Global financial crisis"),
]

# --------------------------------------------------------------------------
# Plot theme
# --------------------------------------------------------------------------
FIG_WIDTH = 7.0          # inches, matches \linewidth at 1in margins on A4
FIG_HEIGHT = 4.0
PALETTE = {
    "Musical": "#2F6F9F",
    "Play": "#C1666B",
    "Special": "#7A9E6F",
    "primary": "#2F6F9F",
    "accent": "#C1666B",
    "muted": "#8C8C8C",
    "grid": "#DDDDDD",
}
