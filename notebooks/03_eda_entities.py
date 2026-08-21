# %% [markdown]
# # 03 - Shows, Theatres and Production Lifecycles
#
# The panel has three nested entities: **theatres** (fixed physical capacity),
# **shows** (the creative product), and **productions** (a show's run in a
# particular theatre). This notebook characterises each, and establishes the
# two structural facts that shape our candidate problems:
#
# * run lengths are extremely heavy-tailed, and
# * the target variable is mechanically linked to several other columns.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data_load import clean, load_analysis, load_raw, theatre_seat_counts
from features import add_same_week_derived
from viz import save_fig, use_report_style

use_report_style()
pd.set_option("display.width", 170)

df = add_same_week_derived(load_analysis())
cleaned_full = clean(load_raw())
print(f"{len(df):,} rows | {df['production_id'].nunique():,} productions | "
      f"{df['show_name'].nunique():,} shows | {df['theatre'].nunique()} theatres")

# %% [markdown]
# ## 3.1 Show types are heavily imbalanced
#
# Musicals dominate both the row count and the revenue. `Special` is a residual
# category of concerts and limited engagements with barely 1% of rows - too thin
# to model as its own class, and a good illustration of why accuracy alone is a
# poor metric on this data.

# %%
type_summary = df.groupby("show_type").agg(
    rows=("date", "size"),
    productions=("production_id", "nunique"),
    median_capacity=("capacity_pct", "median"),
    median_weekly_gross=("gross", "median"),
    total_gross_bn=("gross", lambda s: s.sum() / 1e9),
)
type_summary["row_share_pct"] = (100 * type_summary["rows"] / len(df)).round(1)
print(type_summary.round(2).to_string())

# %%
# ---- Figure 7: composition over time + demand by type ----------------------
fig, axes = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 2.9))

comp = (
    df.pivot_table(index="year", columns="show_type", values="date", aggfunc="size")
    .fillna(0)
)
comp = comp.div(comp.sum(axis=1), axis=0) * 100
axes[0].stackplot(
    comp.index, [comp[c] for c in ["Musical", "Play", "Special"]],
    labels=["Musical", "Play", "Special"],
    colors=[config.PALETTE[c] for c in ["Musical", "Play", "Special"]], alpha=0.9,
)
axes[0].set_title("(a) Share of weekly listings by type")
axes[0].set_ylabel("% of rows"); axes[0].set_xlabel("year"); axes[0].set_ylim(0, 100)
axes[0].legend(loc="lower left", ncol=3, fontsize=7)

data = [df.loc[df["show_type"] == t, "capacity_pct"].dropna() for t in ["Musical", "Play", "Special"]]
parts = axes[1].violinplot(data, showmedians=True, widths=0.85)
for pc, t in zip(parts["bodies"], ["Musical", "Play", "Special"]):
    pc.set_facecolor(config.PALETTE[t]); pc.set_alpha(0.75)
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    parts[key].set_color("black"); parts[key].set_linewidth(0.9)
axes[1].set_xticks([1, 2, 3]); axes[1].set_xticklabels(["Musical", "Play", "Special"])
axes[1].set_ylabel("capacity (%)")
axes[1].set_title("(b) Capacity utilisation by type")

fig.tight_layout()
print(save_fig(fig, "fig07_show_types"))

# %% [markdown]
# ## 3.2 Theatres
#
# Recovering seat counts from the capacity identity (Notebook 01) lets us treat
# theatre size as a genuine numeric feature rather than a 56-level categorical.

# %%
seats_full = theatre_seat_counts(cleaned_full)
theatre = df.groupby("theatre").agg(
    weeks=("date", "size"),
    productions=("production_id", "nunique"),
    median_capacity=("capacity_pct", "median"),
    total_gross_m=("gross", lambda s: s.sum() / 1e6),
).join(seats_full)
print(theatre.sort_values("total_gross_m", ascending=False).head(10).round(1).to_string())

# %%
# ---- Figure 5: validating the capacity identity ----------------------------
usable = cleaned_full[(cleaned_full["performances"] > 0) & (cleaned_full["capacity_pct"] > 0)].copy()
usable["implied_seats"] = (
    usable["attendance"] / usable["performances"] / (usable["capacity_pct"] / 100)
)
top = usable["theatre"].value_counts().head(12).index.tolist()
top = sorted(top, key=lambda t: seats_full[t])

fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.2))
ax.boxplot([usable.loc[usable["theatre"] == t, "implied_seats"].values for t in top],
           showfliers=False, widths=0.6,
           medianprops=dict(color=config.PALETTE["primary"], lw=1.6))
PUBLISHED = {
    "Majestic": 1645, "Imperial": 1417, "Gershwin": 1933, "Broadway": 1761,
    "Minskoff": 1710, "Palace": 1743, "Ambassador": 1125, "Lunt-Fontanne": 1509,
    "Nederlander": 1232, "Marquis": 1611, "St. James": 1710, "Shubert": 1460,
}
xs = [i + 1 for i, t in enumerate(top) if t in PUBLISHED]
ys = [PUBLISHED[t] for t in top if t in PUBLISHED]
ax.scatter(xs, ys, marker="D", s=26, color=config.PALETTE["accent"], zorder=5,
           label="published seat count")
ax.set_xticks(range(1, len(top) + 1))
ax.set_xticklabels(top, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("implied seats")
ax.set_title("Seats implied by attendance / performances / capacity, per theatre")
ax.legend(loc="upper left")
fig.tight_layout()
print(save_fig(fig, "fig05_capacity_validation"))

# %%
# ---- Figure 8: theatre profile ---------------------------------------------
fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.4))
t = theatre.dropna(subset=["theatre_seats"])
sc = ax.scatter(t["theatre_seats"], t["median_capacity"], s=t["total_gross_m"] / 6 + 12,
                c=t["productions"], cmap="viridis", alpha=0.75, edgecolor="white", lw=0.5)
for name in ["Majestic", "Gershwin", "Minskoff", "Ambassador", "Lyceum", "Booth"]:
    if name in t.index:
        ax.annotate(name, (t.loc[name, "theatre_seats"], t.loc[name, "median_capacity"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("theatre seats (recovered)")
ax.set_ylabel("median capacity utilisation (%)")
ax.set_title("Theatre profile: size, demand, cumulative gross (bubble) and productions (colour)")
fig.colorbar(sc, ax=ax, label="distinct productions hosted", pad=0.015)
fig.tight_layout()
print(save_fig(fig, "fig08_theatres"))

# %% [markdown]
# ## 3.3 Production lifecycles
#
# Run length is the most skewed quantity in the dataset. Half of all productions
# last 15 weeks or fewer; the longest, *The Phantom of the Opera* at the Majestic,
# accounts for 995 weeks on its own. Thirty productions were still running when
# the data ends, so their true run length is **right-censored** - treating their
# last observed week as a closure would bias any lifetime estimate downwards.

# %%
runs = df.groupby(["production_id", "show_type"]).agg(
    weeks=("date", "size"), start=("date", "min"), end=("date", "max"),
    total_gross_m=("gross", lambda s: s.sum() / 1e6),
).reset_index()
data_end = df["date"].max()
runs["censored"] = runs["end"] >= data_end - pd.Timedelta(weeks=1)

print(runs["weeks"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(1).to_string())
print(f"\nright-censored productions: {int(runs['censored'].sum())} of {len(runs)}")
print(f"runs of 4 weeks or fewer  : {int((runs['weeks'] <= 4).sum())}")
print("\nlongest runs:")
print(runs.nlargest(6, "weeks")[["production_id", "weeks", "total_gross_m"]].round(1).to_string(index=False))

# %%
def kaplan_meier(durations, events):
    """Kaplan-Meier survival estimate.

    durations: observed weeks; events: 1 if the run actually ended, 0 if the
    observation was censored by the end of the data.
    """
    order = np.argsort(durations)
    d, e = np.asarray(durations)[order], np.asarray(events)[order]
    times, surv, s, at_risk = [0], [1.0], 1.0, len(d)
    for t in np.unique(d):
        n_events = int(e[d == t].sum())
        n_at_t = int((d == t).sum())
        if n_events:
            s *= 1 - n_events / at_risk
            times.append(t); surv.append(s)
        at_risk -= n_at_t
        if at_risk <= 0:
            break
    return np.array(times), np.array(surv)


fig, ax = plt.subplots(figsize=(config.FIG_WIDTH, 3.2))
for t in ["Musical", "Play"]:
    sub = runs[runs["show_type"] == t]
    x, y = kaplan_meier(sub["weeks"].values, (~sub["censored"]).astype(int).values)
    ax.step(x, y, where="post", color=config.PALETTE[t], label=f"{t} (n={len(sub)})")
ax.axhline(0.5, color=config.PALETTE["muted"], ls=":", lw=0.9)
ax.set_xlim(0, 200); ax.set_ylim(0, 1)
ax.set_xlabel("weeks since opening"); ax.set_ylabel("P(still running)")
ax.set_title("Kaplan-Meier survival of productions (right-censored runs handled)")
ax.legend()
for t in ["Musical", "Play"]:
    sub = runs[runs["show_type"] == t]
    x, y = kaplan_meier(sub["weeks"].values, (~sub["censored"]).astype(int).values)
    med = x[np.argmax(y <= 0.5)] if (y <= 0.5).any() else np.nan
    print(f"median survival, {t}: {med:.0f} weeks")
fig.tight_layout()
print(save_fig(fig, "fig09_survival"))

# %% [markdown]
# ## 3.4 Relationships, and the leakage they imply
#
# `capacity_pct` is not an independent measurement. It is an identity over
# attendance, performances and seat count. The correlation matrix makes the
# consequence concrete: attendance and gross are near-perfect proxies for the
# target *within the same week*, which is exactly why they are banned from the
# feature set in `src/features.py`.

# %%
num = df[["attendance", "capacity_pct", "gross", "gross_potential_pct",
          "performances", "avg_ticket_price", "theatre_seats"]]
corr = num.corr()
print(corr.round(3).to_string())

# Demonstrate the identity directly.
chk = df.dropna(subset=["performances", "theatre_seats"]).copy()
chk["capacity_reconstructed"] = (
    100 * chk["attendance"] / (chk["theatre_seats"] * chk["performances"])
)
err = (chk["capacity_reconstructed"] - chk["capacity_pct"]).abs()
print(f"\nReconstructing capacity from same-week columns: "
      f"median abs error {err.median():.2f} pp, 90th pct {err.quantile(0.9):.2f} pp")

# %%
# ---- Figure 10: correlations + the premium-pricing tail ---------------------
fig, axes = plt.subplots(1, 2, figsize=(config.FIG_WIDTH, 3.1))

im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels = ["attend", "capacity", "gross", "gross pot", "perf", "ticket $", "seats"]
axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=7)
for i in range(len(labels)):
    for j in range(len(labels)):
        axes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                     color="white" if abs(corr.values[i, j]) > 0.6 else "black")
axes[0].set_title("(a) Same-week correlations")
axes[0].grid(False)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

sub = df.dropna(subset=["gross_potential_pct"]).sample(6000, random_state=config.SEED)
premium = sub["gross_potential_pct"] > 100
axes[1].scatter(sub.loc[~premium, "capacity_pct"], sub.loc[~premium, "gross_potential_pct"],
                s=3, alpha=0.25, color=config.PALETTE["primary"], lw=0, label="standard pricing")
axes[1].scatter(sub.loc[premium, "capacity_pct"], sub.loc[premium, "gross_potential_pct"],
                s=4, alpha=0.5, color=config.PALETTE["accent"], lw=0, label="premium (> 100%)")
axes[1].axhline(100, color="black", lw=0.8, ls="--")
axes[1].axvline(100, color="black", lw=0.8, ls=":")
axes[1].set_xlabel("capacity (%)"); axes[1].set_ylabel("gross potential (%)")
axes[1].set_title("(b) Capacity is censored at 100%,\ngross potential is not")
axes[1].legend(loc="upper left", fontsize=7)

fig.tight_layout()
print(save_fig(fig, "fig10_relationships"))

# %% [markdown]
# ## 3.5 Takeaways
#
# * The class balance is 71% Musical / 28% Play / 1% Special, so a classifier
#   that always answers "Musical" scores 71% while being useless.
# * Theatre seat counts recovered from the capacity identity span roughly
#   500-1,900 seats and are stable within a theatre, giving us a clean numeric
#   feature in place of a 56-level categorical.
# * Run length has median 15 weeks but a 995-week maximum, and 30 runs are
#   right-censored. Any lifetime model must handle censoring explicitly.
# * Same-week attendance and gross reconstruct `capacity_pct` to within a
#   fraction of a percentage point. This is the dominant leakage risk in the
#   dataset and drives the feature policy used from here on.

# %%
assert err.median() < 1.5, "capacity identity should reconstruct almost exactly"
assert type_summary.loc["Musical", "row_share_pct"] > 65
print("Entity assertions passed.")
