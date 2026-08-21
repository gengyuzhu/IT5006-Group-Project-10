# %% [markdown]
# # 02 - Temporal Patterns and Seasonality
#
# Broadway is a weekly business: contracts, reporting and closing notices all run
# on a Tuesday-to-Sunday cycle. This notebook establishes the three temporal
# structures that any forecasting model has to respect:
#
# 1. a **long-run growth trend** in nominal revenue that is largely a price
#    effect, not a demand effect;
# 2. a **strong annual seasonality** in attendance;
# 3. **rare exogenous shocks** that no feature in this dataset can predict.
#
# The third point matters for how we set expectations in Phase 2: some of the
# variance in this data is genuinely irreducible.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
from data_load import clean, load_analysis, load_raw
from features import add_same_week_derived
from viz import annotate_events, save_fig, use_report_style

use_report_style()
pd.set_option("display.width", 160)

full = add_same_week_derived(clean(load_raw()))   # 1990-2016, for the long view
df = add_same_week_derived(load_analysis())       # 1996-2016, analysis window
print(f"full: {len(full):,} rows | analysis: {len(df):,} rows")

# %% [markdown]
# ## 2.1 The market as a whole

# %%
market = full.groupby("date").agg(
    gross=("gross", "sum"),
    attendance=("attendance", "sum"),
    productions=("production_id", "nunique"),
    median_capacity=("capacity_pct", "median"),
)
print(market.describe().round(1))

# %% [markdown]
# ### Exogenous shocks are visible and severe
#
# Two weeks in the series are unlike any others, and both have a documented
# external cause.

# %%
def window(a, b):
    return market.loc[a:b, ["gross", "productions"]].assign(
        gross=lambda d: (d["gross"] / 1e6).round(2)
    ).rename(columns={"gross": "market_gross_$m"})

print("--- September 2001 ---")
print(window("2001-08-26", "2001-10-07").to_string())
print("\n--- November 2007 stagehand strike ---")
print(window("2007-10-28", "2007-12-09").to_string())

# %%
# Quantify the two shocks against the preceding week.
for label, shock_wk, base_wk in [
    ("9/11", "2001-09-16", "2001-09-09"),
    ("Stagehand strike", "2007-11-18", "2007-11-04"),
]:
    g0, g1 = market.loc[base_wk, "gross"], market.loc[shock_wk, "gross"]
    p0, p1 = market.loc[base_wk, "productions"], market.loc[shock_wk, "productions"]
    print(
        f"{label:18s} gross ${g0/1e6:5.1f}m -> ${g1/1e6:5.1f}m ({100*(g1-g0)/g0:+.0f}%)"
        f" | productions {p0:.0f} -> {p1:.0f}"
    )

# %%
# ---- Figure 2: market gross with events ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(config.FIG_WIDTH, 5.0), sharex=True)

ax = axes[0]
ax.plot(market.index, market["gross"] / 1e6, color=config.PALETTE["primary"], lw=0.8)
roll = market["gross"].rolling(52, min_periods=26).mean() / 1e6
ax.plot(market.index, roll, color="black", lw=1.4, label="52-week moving average")
ax.set_ylabel("weekly gross (USD m)")
ax.set_title("(a) Total Broadway weekly gross, 1990-2016")
ax.legend(loc="upper left")
annotate_events(ax, config.EVENTS)

ax = axes[1]
ax.plot(market.index, market["productions"], color=config.PALETTE["muted"], lw=0.8)
ax.set_ylabel("productions running")
ax.set_xlabel("week ending")
ax.set_title("(b) Number of productions reporting each week")
annotate_events(ax, config.EVENTS)
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
print(save_fig(fig, "fig02_market_events"))

# %% [markdown]
# ## 2.2 Seasonality
#
# Attendance follows a pronounced annual cycle. Because `capacity_pct` is
# scale-free it is the right lens here: it removes both theatre size and ticket
# price, leaving demand.

# %%
by_week = df.groupby("iso_week")["capacity_pct"].agg(["median", "count"])
print("highest-demand ISO weeks:")
print(by_week.sort_values("median", ascending=False).head(5))
print("\nlowest-demand ISO weeks:")
print(by_week[by_week["count"] > 100].sort_values("median").head(5))

by_month = df.groupby("month")["capacity_pct"].median().round(1)
print("\nmedian capacity by month:")
print(by_month.to_string())

# %%
# ---- Figure 6: seasonality heatmap -----------------------------------------
pivot = (
    df.pivot_table(index="iso_week", columns="year", values="capacity_pct", aggfunc="median")
    .reindex(range(1, 53))
)
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.6))
im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", origin="lower",
               extent=[pivot.columns.min() - 0.5, pivot.columns.max() + 0.5, 0.5, 52.5],
               vmin=60, vmax=100)
ax.set_xlabel("year")
ax.set_ylabel("ISO week of year")
ax.set_title("Median capacity utilisation by week of year (1996-2016)")
ax.set_yticks([1, 9, 18, 26, 35, 44, 52])
for wk, lbl in [(52, "holidays"), (36, "Sept trough")]:
    ax.axhline(wk, color="black", lw=0.7, ls=":")
    ax.annotate(lbl, xy=(pivot.columns.min() + 0.3, wk - 1.6), fontsize=7)
cb = fig.colorbar(im, ax=ax, pad=0.015)
cb.set_label("median capacity (%)", fontsize=8)
fig.tight_layout()
print(save_fig(fig, "fig06_seasonality"))

# %% [markdown]
# ## 2.3 The inflation confound
#
# Nominal gross nearly quadruples across the series, but that is not four times
# the demand. Average ticket price rises from roughly \$24 to \$93 while median
# capacity utilisation is essentially flat. Any model trained on dollar levels
# would spend its capacity learning the price trend.
#
# This is why the analysis is built on `capacity_pct` and `gross_potential_pct`
# rather than on dollars, and why `year` is carried as an explicit feature
# instead of being deflated with an external price index.

# %%
yearly = df.groupby("year").agg(
    median_ticket_price=("avg_ticket_price", "median"),
    median_capacity=("capacity_pct", "median"),
    total_gross=("gross", "sum"),
    total_attendance=("attendance", "sum"),
)
print(yearly.round(1).to_string())
r = yearly["median_ticket_price"].corr(yearly["median_capacity"])
print(f"\ncorr(median ticket price, median capacity) across years = {r:.3f}")

# %%
# ---- Figure 11: price versus demand ----------------------------------------
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.0))
ax.plot(yearly.index, yearly["median_ticket_price"], color=config.PALETTE["accent"],
        marker="o", ms=3, label="median ticket price (nominal USD)")
ax.set_ylabel("USD", color=config.PALETTE["accent"])
ax.set_xlabel("year")
ax.tick_params(axis="y", colors=config.PALETTE["accent"])

ax2 = ax.twinx()
ax2.plot(yearly.index, yearly["median_capacity"], color=config.PALETTE["primary"],
         marker="s", ms=3, label="median capacity utilisation")
ax2.set_ylabel("capacity (%)", color=config.PALETTE["primary"])
ax2.set_ylim(60, 100)
ax2.tick_params(axis="y", colors=config.PALETTE["primary"])
ax2.grid(False)

ax.set_title("Nominal revenue growth is a price effect, not a demand effect")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=7.5,
          framealpha=0.9, facecolor="white", edgecolor="none")
fig.tight_layout()
print(save_fig(fig, "fig11_price_vs_demand"))

# %% [markdown]
# ### Aggregation granularity matters
#
# The monthly view hides the very pattern it is supposed to show: December's
# median capacity (78%) is identical to September's, the weakest month of the
# year. The holiday peak is a **two-week** phenomenon confined to ISO weeks 52-53,
# and averaging it together with the slow first three weeks of December cancels
# it out entirely.
#
# This is a concrete argument for engineering calendar features at ISO-week
# resolution rather than month resolution, and we adopt week-level flags in
# `src/features.py` accordingly.

# %%
wk_peak = by_week.loc[[52, 53], "median"].mean()
wk_trough = by_week.loc[[36, 44], "median"].mean()
print(f"ISO week 52-53 median capacity : {wk_peak:.1f}%")
print(f"ISO week 36/44 median capacity : {wk_trough:.1f}%")
print(f"week-level contrast            : {wk_peak - wk_trough:.1f} pp")
print(f"month-level contrast (Dec-Sep)  : {by_month.loc[12] - by_month.loc[9]:.1f} pp  <- signal destroyed")

# %% [markdown]
# ## 2.4 Takeaways
#
# * Between 1996 and 2016 the median ticket price rises from \$44.6 to \$93.1
#   (+109%) while median capacity utilisation moves only from 82% to 85%. The
#   growth in nominal gross is overwhelmingly a price effect, so dollar-valued
#   targets would force a model to spend its capacity learning inflation.
# * Demand is strongly seasonal, but only at week resolution: ISO weeks 52-53
#   reach ~90% capacity while weeks 36 and 44 sit at 73-74%. The same contrast
#   measured monthly is zero. Calendar features must be built at week level.
# * Two weeks are dominated by exogenous shocks: the week of 9/11 (market gross
#   down 54%) and the November 2007 stagehand strike (only 8 of 34 productions
#   able to play). No feature available to us predicts either. We keep them in
#   the data and flag them, rather than deleting inconvenient rows.

# %%
assert wk_peak - wk_trough > 10, "expected a strong week-level seasonal contrast"
assert abs(by_month.loc[12] - by_month.loc[9]) < 1, (
    "monthly aggregation is expected to cancel the holiday peak"
)
assert market.loc["2007-11-18", "productions"] == 8
print("Temporal assertions passed.")
