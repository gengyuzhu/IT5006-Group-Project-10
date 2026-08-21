# %% [markdown]
# # 01 - Data Audit
#
# Establishes the data contract for `broadway_clean.csv` and documents every
# cleaning decision before any analysis is done.
#
# The audit answers four questions:
#
# 1. **What is the grain?** One row per (week, show, theatre).
# 2. **What do the columns actually mean?** In particular, is `Statistics.Capacity`
#    a seat count or a percentage? We settle this from the data itself.
# 3. **Where is the data missing?** The file has no `NaN` at all, which is
#    suspicious rather than reassuring - missingness is coded as zero.
# 4. **Which window is safe to analyse?**
#
# Every claim below is asserted, so re-running this notebook top-to-bottom is a
# proof that the numbers quoted in the report still hold.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data_load import (
    CLEANING_DECISIONS,
    clean,
    load_raw,
    quality_summary,
    theatre_seat_counts,
)
from viz import save_fig, use_report_style

use_report_style()
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

# %% [markdown]
# ## 1.1 Raw data contract
#
# `load_raw` asserts the shape, the absence of nulls and duplicates, the natural
# key, and that every observation is a week ending on a Sunday. If any of these
# ever fail, everything downstream is suspect.

# %%
raw = load_raw()
print(f"rows x cols          : {raw.shape[0]:,} x {raw.shape[1]}")
print(f"date range           : {raw['date'].min().date()} -> {raw['date'].max().date()}")
print(f"distinct week-endings: {raw['date'].nunique():,}")
print(f"distinct shows       : {raw['Show.Name'].nunique():,}")
print(f"distinct theatres    : {raw['Show.Theatre'].nunique():,}")
print(f"nulls                : {int(raw.isna().sum().sum())}")
print(f"duplicate rows       : {int(raw.duplicated().sum())}")
raw.head()

# %% [markdown]
# ## 1.2 What does `Statistics.Capacity` mean?
#
# The column is named "Capacity" but ranges only from 10 to 100, so it cannot be
# a seat count. If it is instead the **percentage of seats sold**, then
#
# $$\text{seats} = \frac{\text{attendance}}{\text{performances}} \Big/ \frac{\text{capacity}}{100}$$
#
# should recover a *constant* seat count for each theatre. It does - and the
# recovered values match the published seat counts closely. This is the evidence
# that fixes the column's meaning, and it also hands us a legitimate static
# feature (`theatre_seats`) that involves no leakage.

# %%
cleaned = clean(raw)
seats = theatre_seat_counts(cleaned)

# A handful of published seat counts, for external validation.
PUBLISHED = {
    "Majestic": 1645, "Imperial": 1417, "Gershwin": 1933, "Broadway": 1761,
    "Minskoff": 1710, "Palace": 1743, "Ambassador": 1125, "Lunt-Fontanne": 1509,
}
check = pd.DataFrame(
    {"recovered": seats.reindex(PUBLISHED), "published": pd.Series(PUBLISHED)}
)
check["abs_error_pct"] = (
    100 * (check["recovered"] - check["published"]).abs() / check["published"]
).round(1)
print(check)
print(f"\nmedian absolute error: {check['abs_error_pct'].median():.1f}%")
assert check["abs_error_pct"].median() < 6, "Capacity-as-percentage reading failed"

# %% [markdown]
# ## 1.3 Missingness is coded as zero
#
# The file contains no `NaN`, but two columns use `0` as a missing marker and a
# third is censored at its maximum.

# %%
n = len(cleaned)
flags = {
    "Performances == 0 (but attendance > 0)": cleaned["perf_missing"].sum(),
    "Gross Potential == 0": cleaned["gp_missing"].sum(),
    "Capacity == 100 (right-censored)": cleaned["capacity_censored"].sum(),
    "Gross Potential > 100 (premium pricing)": (cleaned["gross_potential_pct"] > 100).sum(),
}
audit = pd.DataFrame(
    {"n_rows": pd.Series(flags)}
).assign(pct=lambda d: (100 * d["n_rows"] / n).round(1))
print(audit)

# Performances == 0 is structurally impossible: those weeks still sold tickets.
impossible = cleaned[cleaned["perf_missing"] & (cleaned["attendance"] > 0)]
print(f"\nweeks with 0 performances but positive attendance: {len(impossible):,}")
assert len(impossible) == int(cleaned["perf_missing"].sum())

# %%
# When do the zero-coded values occur? They cluster by reporting era, not by show,
# which is what tells us they are a data-collection artefact.
zero_by_year = (
    cleaned.groupby("year")[["perf_missing", "gp_missing"]].mean().mul(100).round(1)
)
print(zero_by_year.loc[1994:2010])

# %% [markdown]
# ## 1.4 Coverage gaps
#
# 1,281 distinct weeks are present, but the span from Aug 1990 to Aug 2016 covers
# roughly 1,357 calendar weeks. The shortfall is concentrated in two large holes
# at the start of the series.

# %%
dates = pd.Series(sorted(cleaned["date"].unique()))
gaps = dates.diff().dt.days
big = pd.DataFrame({"from": dates.shift(1), "to": dates, "days": gaps})
big = big[big["days"] > 7]
big["weeks_missing"] = (big["days"] // 7 - 1).astype(int)
print(big.to_string(index=False))
print(f"\ntotal weeks missing: {int(big['weeks_missing'].sum())}")

# %%
per_year = cleaned.groupby("year").agg(
    rows=("date", "size"), weeks=("date", "nunique"), shows=("show_name", "nunique")
)
print(per_year.head(8))
print("...")
print(per_year.tail(4))

# %% [markdown]
# **Decision.** Restrict the main analysis to `1996-01-01` onward. 1990-1995 is
# not merely gappy but structurally thinner (a handful of shows per week versus
# ~25 later), so including it would mix a reporting artefact into every temporal
# trend. The early period is retained in Figure 1 as a data-quality exhibit.

# %%
# ---- Figure 1: data quality panel -----------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(config.FIG_WIDTH, 5.0))

ax = axes[0]
weekly_rows = cleaned.groupby("date").size()
full_index = pd.date_range(dates.min(), dates.max(), freq="W-SUN")
coverage = weekly_rows.reindex(full_index).fillna(0)
ax.fill_between(coverage.index, coverage.values, color=config.PALETTE["primary"], alpha=0.75, lw=0)
for _, row in big.iterrows():
    ax.axvspan(row["from"], row["to"], color=config.PALETTE["accent"], alpha=0.30, lw=0)
ax.axvline(config.ANALYSIS_START, color="black", ls="--", lw=1.0)
ax.annotate(
    "analysis window starts 1996",
    xy=(config.ANALYSIS_START, ax.get_ylim()[1] * 0.82),
    xytext=(6, 0), textcoords="offset points", fontsize=7.5,
)
ax.set_title("(a) Weekly coverage: productions reported per week")
ax.set_ylabel("productions reported")
ax.set_xlabel("")

ax = axes[1]
share = cleaned.groupby("year")[["perf_missing", "gp_missing", "capacity_censored"]].mean().mul(100)
ax.plot(share.index, share["gp_missing"], label="Gross Potential coded 0", color=config.PALETTE["accent"])
ax.plot(share.index, share["perf_missing"], label="Performances coded 0", color=config.PALETTE["primary"])
ax.plot(share.index, share["capacity_censored"], label="Capacity censored at 100%",
        color=config.PALETTE["muted"], ls="--")
ax.set_title("(b) Zero-coded missingness and censoring, by year")
ax.set_ylabel("% of rows")
ax.set_xlabel("year")
ax.legend(loc="upper right")

fig.tight_layout()
print(save_fig(fig, "fig01_data_quality"))

# %%
# ---- Figure 4: distributions, with the zero spikes left visible ------------
fig, axes = plt.subplots(1, 5, figsize=(config.FIG_WIDTH, 2.1))
specs = [
    ("Statistics.Attendance", "Attendance", None),
    ("Statistics.Capacity", "Capacity (%)", None),
    ("Statistics.Gross", "Gross (USD m)", 1e6),
    ("Statistics.Gross Potential", "Gross potential (%)", None),
    ("Statistics.Performances", "Performances", None),
]
for ax, (col, label, scale) in zip(axes, specs):
    v = cleaned[col].astype(float)
    if scale:
        v = v / scale
    ax.hist(v, bins=40, color=config.PALETTE["primary"], alpha=0.85)
    zeros = int((cleaned[col] == 0).sum())
    if zeros:
        ax.axvline(0, color=config.PALETTE["accent"], lw=1.6)
        ax.set_title(f"{label}\n{zeros:,} zeros", fontsize=8)
    else:
        ax.set_title(label, fontsize=8)
    ax.set_yticks([])
fig.suptitle("Raw distributions before cleaning (red line marks the zero spike)", fontsize=9)
fig.tight_layout()
print(save_fig(fig, "fig04_distributions"))

# %% [markdown]
# ## 1.5 Cleaning contract
#
# The full set of decisions lives in `src/data_load.py::CLEANING_DECISIONS`, so
# the report table and the code cannot drift apart.

# %%
contract = pd.DataFrame(CLEANING_DECISIONS).T
for name, row in contract.iterrows():
    print(f"\n### {name}")
    print(f"  issue    : {row['issue']}")
    print(f"  evidence : {row['evidence']}")
    print(f"  decision : {row['decision']}")

# %%
from data_load import load_analysis

analysis = load_analysis()
print(f"analysis window rows : {len(analysis):,}")
print(f"window               : {analysis['date'].min().date()} -> {analysis['date'].max().date()}")
print(f"productions          : {analysis['production_id'].nunique():,}")
print(f"theatres             : {analysis['theatre'].nunique()}")
print()
print(quality_summary(analysis).to_string(index=False))

# %%
# Final contract assertions - these are what make the notebook a proof.
assert len(raw) == config.EXPECTED_RAW_ROWS
assert cleaned["perf_missing"].sum() == 2309
assert cleaned["gp_missing"].sum() == 1918
assert int(big["weeks_missing"].sum()) == 75
assert analysis["date"].min() >= config.ANALYSIS_START
assert not analysis.duplicated(subset=["date", "show_name", "theatre"]).any()
print("All data-contract assertions passed.")
