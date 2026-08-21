"""
Broadway Weekly Grosses - IT5006 Phase 1 exploratory dashboard.

Run locally:
    streamlit run dashboard/app.py

Every view reads through src/data_load.py, the same entry point the notebooks
and the report use, so no figure here can disagree with the PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import theme

st.set_page_config(
    page_title="Broadway Weekly Grosses | IT5006",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.register_template()
st.markdown(theme.CSS, unsafe_allow_html=True)

PLOT_CFG = {"displayModeBar": False, "responsive": True}

# Published seat counts, used to validate the recovered capacity identity.
PUBLISHED_SEATS = {
    "Majestic": 1645, "Imperial": 1417, "Gershwin": 1933, "Broadway": 1761,
    "Minskoff": 1710, "Palace": 1743, "Ambassador": 1125, "Lunt-Fontanne": 1509,
    "Nederlander": 1232, "Marquis": 1611, "St. James": 1710, "Shubert": 1460,
}


# ===========================================================================
# Data
# ===========================================================================
@st.cache_data(show_spinner="Loading Broadway data…")
def get_data() -> pd.DataFrame:
    """Load, validate and clean the full 1990-2016 history.

    Raises are surfaced to the user rather than swallowed: if the data contract
    in data_load.load_raw() fails, every number on this page would be wrong.
    """
    from data_load import clean, load_raw, theatre_seat_counts
    from features import add_same_week_derived

    df = add_same_week_derived(clean(load_raw()))
    return df.merge(
        theatre_seat_counts(df), left_on="theatre", right_index=True, how="left"
    )


try:
    DF = get_data()
    from config import ANALYSIS_START, EVENTS
    from data_load import CLEANING_DECISIONS
except FileNotFoundError:
    st.error(
        "**Data file not found.** Expected `data/raw/broadway_clean.csv` relative "
        "to the project root. Clone the full repository and run this app from "
        "the project root: `streamlit run dashboard/app.py`."
    )
    st.stop()
except AssertionError as exc:
    st.error(
        f"**The data failed its integrity contract:** {exc}\n\n"
        "`src/data_load.py` asserts the expected shape, natural key and date "
        "convention. A failure here means the source file has changed and no "
        "figure on this page can be trusted."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - last-resort guard for a public app
    st.error(f"**Could not start the dashboard:** `{type(exc).__name__}: {exc}`")
    st.stop()


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.markdown("### 🎭 Broadway Explorer")
    st.caption("IT5006 Phase 1 · Group 10")

    with st.expander("ℹ️  How to use this dashboard", expanded=False):
        st.markdown(
            "1. **Filter** with the controls below — every tab responds.\n"
            "2. **Overview** — market-level revenue, attendance and shocks.\n"
            "3. **Productions** — compare shows on calendar time *or* aligned "
            "by week of run.\n"
            "4. **Venues** — theatre size against how full it actually gets.\n"
            "5. **Seasonality** — when demand peaks, and why month-level "
            "aggregation hides it.\n"
            "6. **Data quality** — the defects we found and what we did.\n\n"
            "Charts are interactive: drag to zoom, double-click to reset, "
            "hover for values."
        )

    st.markdown("#### Filters")
    y0, y1 = int(DF["year"].min()), int(DF["year"].max())
    years = st.slider(
        "Season range", y0, y1, (int(ANALYSIS_START.year), y1),
        help="The report's analysis window starts in 1996. Earlier years are "
             "sparse and contain 75 missing weeks.",
    )
    types = st.multiselect(
        "Show type", ["Musical", "Play", "Special"],
        default=["Musical", "Play", "Special"],
    )
    theatres = st.multiselect(
        "Theatre", sorted(DF["theatre"].unique()), default=[],
        help="Leave empty to include all 56 theatres.",
    )

    mask = DF["year"].between(*years) & DF["show_type"].isin(types)
    if theatres:
        mask &= DF["theatre"].isin(theatres)
    d = DF[mask]

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(d):,}")
    c2.metric("Productions", f"{d['production_id'].nunique():,}")

    with st.expander("⚠️  Data caveats", expanded=False):
        st.markdown(
            f"- **Capacity** is the *percentage of seats sold*, censored at 100%.\n"
            f"- **Gross Potential = 0** marks a missing value, not a zero "
            f"({int(DF['gp_missing'].sum()):,} rows), shown blank here.\n"
            f"- **Performances = 0** is structurally impossible and also marks "
            f"missing data ({int(DF['perf_missing'].sum()):,} rows).\n"
            f"- **Gross is nominal USD**, not inflation-adjusted: cross-decade "
            f"comparisons reflect ticket prices as much as demand.\n"
            f"- 1990–95 has 75 missing weeks in total."
        )

    st.markdown("---")
    st.caption(
        "Data: Broadway weekly grosses (CORGIS Dataset Project, from The "
        "Broadway League), distributed via Canvas as `broadway_clean.csv`."
    )

# ===========================================================================
# Masthead
# ===========================================================================
st.markdown(
    theme.masthead(
        "IT5006 · Phase 1 · Exploratory Data Analysis",
        "Broadway Weekly Grosses, 1990–2016",
        f"{len(DF):,} weekly box-office records · one row per "
        f"<b>week × show × theatre</b> · {DF['show_name'].nunique():,} shows across "
        f"{DF['theatre'].nunique()} Manhattan theatres. All views read through the "
        f"same cleaning pipeline as the report, so the numbers cannot diverge.",
    ),
    unsafe_allow_html=True,
)

if d.empty:
    st.warning(
        "**No rows match the current filters.** Widen the season range or add "
        "a show type in the sidebar."
    )
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Productions", "Venues", "Seasonality", "Data quality"]
)


# ===========================================================================
# Tab 1 — Overview
# ===========================================================================
with tab1:
    weekly = (
        d.groupby("date")
        .agg(gross=("gross", "sum"), attendance=("attendance", "sum"),
             productions=("production_id", "nunique"),
             capacity=("capacity_pct", "median"))
        .reset_index()
    )

    st.markdown(
        theme.kpi_row([
            ("Total gross", f"${d['gross'].sum() / 1e9:.2f}bn",
             f"{years[0]}–{years[1]}", theme.GOLD),
            ("Total attendance", f"{d['attendance'].sum() / 1e6:.1f}m",
             "tickets sold", theme.PRIMARY),
            ("Median capacity", f"{d['capacity_pct'].median():.0f}%",
             "of seats sold per week", theme.GREEN),
            ("Median ticket", f"${d['avg_ticket_price'].median():.0f}",
             "nominal, not deflated", theme.ACCENT),
        ]),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1], gap="medium")
    with right:
        metric = st.radio(
            "Series",
            ["Weekly gross", "Attendance", "Productions running", "Median capacity"],
            label_visibility="collapsed",
        )
        show_ma = st.checkbox("52-week average", value=True)
        show_ev = st.checkbox("Mark shocks", value=True)

    col, label, colour, fmt = {
        "Weekly gross": ("gross", "weekly gross (USD)", theme.GOLD, "$,.0f"),
        "Attendance": ("attendance", "tickets sold", theme.PRIMARY, ",.0f"),
        "Productions running": ("productions", "productions", theme.MUTED, ",.0f"),
        "Median capacity": ("capacity", "capacity (%)", theme.GREEN, ".0f"),
    }[metric]

    with left:
        fig = go.Figure()
        r, g, b = (int(colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        fig.add_scatter(
            x=weekly["date"], y=weekly[col], mode="lines", name=metric,
            line=dict(color=colour, width=1.1),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.16)",
            hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:{fmt}}}<extra></extra>",
        )
        if show_ma and len(weekly) > 60:
            fig.add_scatter(
                x=weekly["date"], y=weekly[col].rolling(52, min_periods=13).mean(),
                mode="lines", name="52-week average",
                line=dict(color=theme.INK, width=2.2), hoverinfo="skip",
            )
        if show_ev:
            # Explicit shape + annotation: plotly's annotated add_vline averages
            # its x endpoints, which raises on Timestamp input.
            for i, (date, name) in enumerate(EVENTS):
                if weekly["date"].min() <= date <= weekly["date"].max():
                    fig.add_shape(type="line", x0=date, x1=date, yref="paper",
                                  y0=0, y1=1,
                                  line=dict(color=theme.ACCENT, width=1, dash="dot"))
                    fig.add_annotation(
                        x=date, yref="paper", y=1.02 - 0.09 * (i % 3),
                        text=name, showarrow=False, xanchor="left", xshift=4,
                        font=dict(size=10, color=theme.ACCENT),
                    )
        fig.update_layout(title=f"{metric} by week", yaxis_title=label, height=380)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "What to look for",
            "The week ending <b>16 Sep 2001</b> shows the 9/11 collapse — market "
            "gross roughly halved, from &#36;7.48m to &#36;3.47m — while the number of "
            "productions barely moved. The <b>Nov 2007 stagehands' strike</b> is the "
            "opposite signature: productions able to play fell from 33 to 8 and "
            "gross collapsed to &#36;4.6m. Neither shock is predictable from anything "
            "in this dataset, which is a useful reminder that a real share of the "
            "variance here is irreducible."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown(theme.section("Highest-grossing productions",
                                  "Cumulative gross within the current filters."),
                    unsafe_allow_html=True)
        top = (d.groupby(["show_name", "show_type"])["gross"].sum()
               .reset_index().nlargest(10, "gross").sort_values("gross"))
        fig = px.bar(top, x="gross", y="show_name", orientation="h",
                     color="show_type", color_discrete_map=theme.TYPE_COLOURS,
                     labels={"gross": "cumulative gross (USD)", "show_name": "",
                             "show_type": ""})
        fig.update_layout(height=330, legend=dict(y=1.06))
        fig.update_traces(hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    with c2:
        st.markdown(theme.section("Strongest weeks",
                                  "Ranked by total market gross."),
                    unsafe_allow_html=True)
        best = weekly.nlargest(10, "gross")[
            ["date", "gross", "productions", "capacity"]
        ].copy()
        best["date"] = best["date"].dt.strftime("%d %b %Y")
        st.dataframe(
            best.rename(columns={"date": "Week ending", "gross": "Market gross",
                                 "productions": "Shows", "capacity": "Median cap."}),
            hide_index=True, use_container_width=True, height=330,
            column_config={
                "Market gross": st.column_config.NumberColumn(format="$%d"),
                "Median cap.": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )


# ===========================================================================
# Tab 2 — Productions
# ===========================================================================
with tab2:
    totals = (
        d.groupby("show_name")
        .agg(total_gross=("gross", "sum"), weeks=("date", "size"),
             median_capacity=("capacity_pct", "median"),
             median_price=("avg_ticket_price", "median"))
        .sort_values("total_gross", ascending=False)
    )

    st.markdown(
        theme.section(
            "Compare productions",
            "Calendar time shows <i>when</i> a show ran; run-aligned time removes "
            "the era and compares <i>lifecycles</i> — the fairer view when a 1997 "
            "show is set against a 2014 one."),
        unsafe_allow_html=True,
    )

    csel, cmeas = st.columns([3, 2], gap="medium")
    with csel:
        picked = st.multiselect(
            "Productions", totals.index.tolist(),
            default=totals.head(4).index.tolist(),
            label_visibility="collapsed",
        )
    with cmeas:
        measure = st.selectbox(
            "Measure",
            ["capacity_pct", "gross", "attendance", "avg_ticket_price"],
            format_func=lambda c: {
                "capacity_pct": "Capacity utilisation (%)",
                "gross": "Weekly gross (USD)",
                "attendance": "Weekly attendance",
                "avg_ticket_price": "Average ticket price (USD)",
            }[c],
            label_visibility="collapsed",
        )

    if not picked:
        st.info("Select at least one production above to draw the comparison.")
    else:
        sub = d[d["show_name"].isin(picked)].sort_values("date")
        ylab = {"capacity_pct": "capacity (%)", "gross": "gross (USD)",
                "attendance": "attendance", "avg_ticket_price": "USD"}[measure]

        a, b = st.columns(2, gap="medium")
        with a:
            fig = px.line(sub, x="date", y=measure, color="show_name",
                          labels={"date": "week ending", "show_name": ""})
            fig.update_traces(line=dict(width=1.4))
            fig.update_layout(title="By calendar date", yaxis_title=ylab, height=360)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
        with b:
            run = sub.copy()
            run["run_week"] = run.groupby("show_name").cumcount() + 1
            fig = px.line(run, x="run_week", y=measure, color="show_name",
                          labels={"run_week": "week of run", "show_name": ""})
            fig.update_traces(line=dict(width=1.4))
            fig.update_layout(title="Aligned by week of run", yaxis_title=ylab,
                              height=360)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

        st.dataframe(
            totals.loc[picked].reset_index().rename(columns={
                "show_name": "Production", "total_gross": "Total gross",
                "weeks": "Weeks run", "median_capacity": "Median capacity",
                "median_price": "Median ticket"}),
            hide_index=True, use_container_width=True,
            column_config={
                "Total gross": st.column_config.NumberColumn(format="$%d"),
                "Median capacity": st.column_config.NumberColumn(format="%.0f%%"),
                "Median ticket": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(theme.section("How long do productions last?",
                              "Run length is the most skewed quantity in the data."),
                unsafe_allow_html=True)

    runs = (d.groupby(["production_id", "show_type"])
            .agg(weeks=("date", "size")).reset_index())
    g1, g2 = st.columns([2, 1], gap="medium")
    with g1:
        fig = px.histogram(runs[runs["weeks"] <= 150], x="weeks", nbins=50,
                           color="show_type", color_discrete_map=theme.TYPE_COLOURS,
                           labels={"weeks": "weeks run", "show_type": ""})
        fig.add_vline(x=float(runs["weeks"].median()), line=dict(
            color=theme.INK, width=1.5, dash="dash"))
        fig.update_layout(title="Distribution of run length (truncated at 150 weeks)",
                          yaxis_title="productions", height=320, barmode="stack")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with g2:
        st.markdown(
            theme.note(
                "Heavy tail",
                f"Median run is <b>{runs['weeks'].median():.0f} weeks</b>, but the "
                f"longest is <b>{runs['weeks'].max():.0f}</b> "
                "(<i>The Phantom of the Opera</i> at the Majestic). A handful of "
                "productions dominate the panel, so any model must be evaluated "
                "with productions — not rows — held out."
            ),
            unsafe_allow_html=True,
        )


# ===========================================================================
# Tab 3 — Venues
# ===========================================================================
with tab3:
    th = (
        d.groupby("theatre")
        .agg(weeks=("date", "size"), productions=("production_id", "nunique"),
             median_capacity=("capacity_pct", "median"),
             total_gross=("gross", "sum"), seats=("theatre_seats", "median"))
        .dropna(subset=["seats"]).reset_index()
    )

    st.markdown(
        theme.section("Theatre size against how full it gets",
                      "Bubble area is cumulative gross; colour is the number of "
                      "distinct productions the venue hosted."),
        unsafe_allow_html=True,
    )

    a, b = st.columns([3, 2], gap="medium")
    with a:
        fig = px.scatter(
            th, x="seats", y="median_capacity", size="total_gross",
            color="productions", hover_name="theatre", size_max=44,
            color_continuous_scale=theme.SEQUENTIAL,
            labels={"seats": "theatre seats (recovered)",
                    "median_capacity": "median capacity utilisation (%)",
                    "productions": "productions"},
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="white")))
        fig.update_layout(height=400, title="Bigger houses are not fuller houses",
                          coloraxis_colorbar=dict(thickness=10, len=0.7))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with b:
        rank = th.nlargest(12, "total_gross").sort_values("total_gross")
        fig = px.bar(rank, x="total_gross", y="theatre", orientation="h",
                     labels={"total_gross": "cumulative gross (USD)", "theatre": ""})
        fig.update_traces(marker_color=theme.GOLD,
                          hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
        fig.update_layout(height=400, title="Highest-grossing venues")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "Seat counts are not in the raw file",
            "They are <b>recovered</b> by inverting the identity "
            "<code>capacity = attendance / (seats × performances)</code>. The result "
            "reproduces published seat counts to a median error of <b>1.7%</b> — see "
            "the <b>Data quality</b> tab for the validation. This is the evidence "
            "that <code>Statistics.Capacity</code> is a percentage, not a seat count."
        ),
        unsafe_allow_html=True,
    )

    st.dataframe(
        th.sort_values("total_gross", ascending=False).rename(columns={
            "theatre": "Theatre", "seats": "Seats", "weeks": "Weeks",
            "productions": "Productions", "median_capacity": "Median capacity",
            "total_gross": "Total gross"}),
        hide_index=True, use_container_width=True, height=300,
        column_config={
            "Total gross": st.column_config.NumberColumn(format="$%d"),
            "Median capacity": st.column_config.NumberColumn(format="%.0f%%"),
            "Seats": st.column_config.NumberColumn(format="%d"),
        },
    )


# ===========================================================================
# Tab 4 — Seasonality
# ===========================================================================
with tab4:
    st.markdown(
        theme.section("When does Broadway fill up?",
                      "Capacity utilisation is scale-free — it removes both theatre "
                      "size and ticket price, leaving demand."),
        unsafe_allow_html=True,
    )

    a, b = st.columns([3, 2], gap="medium")
    with a:
        pivot = (d.pivot_table(index="iso_week", columns="year",
                               values="capacity_pct", aggfunc="median")
                 .reindex(range(1, 53)))
        fig = px.imshow(pivot, aspect="auto", origin="lower",
                        color_continuous_scale=theme.DIVERGING, zmin=60, zmax=100,
                        labels=dict(x="year", y="ISO week", color="capacity %"))
        fig.update_layout(height=420, title="Median capacity by week of year",
                          coloraxis_colorbar=dict(thickness=10, len=0.7))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with b:
        wk = d.groupby("iso_week")["capacity_pct"].median().reset_index()
        fig = px.bar(wk, x="iso_week", y="capacity_pct",
                     labels={"iso_week": "ISO week", "capacity_pct": "median capacity %"})
        fig.update_traces(marker_color=theme.PRIMARY,
                          hovertemplate="week %{x}<br>%{y:.0f}%<extra></extra>")
        fig.update_yaxes(range=[50, 100])
        for w, lab, col in [(52, "holidays", theme.ACCENT), (36, "Sept trough", theme.MUTED)]:
            fig.add_vline(x=w, line=dict(color=col, width=1, dash="dot"))
        fig.update_layout(height=420, title="Seasonal profile across the year")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        theme.section("Why we build calendar features by week, not by month",
                      "The same seasonal signal, measured two ways."),
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="medium")
    by_wk = d.groupby("iso_week")["capacity_pct"].median()
    by_mo = d.groupby("month")["capacity_pct"].median()
    with a:
        pk, tr = by_wk.loc[[52, 53]].mean(), by_wk.loc[[36, 44]].mean()
        fig = go.Figure(go.Bar(
            x=["Weeks 52–53<br>(holidays)", "Weeks 36 & 44<br>(troughs)"],
            y=[pk, tr], marker_color=[theme.ACCENT, theme.PRIMARY],
            text=[f"{pk:.0f}%", f"{tr:.0f}%"], textposition="outside"))
        fig.update_yaxes(range=[0, 105], title="median capacity %")
        fig.update_layout(height=300,
                          title=f"By ISO week — a {pk - tr:.0f} point swing")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with b:
        fig = go.Figure(go.Bar(
            x=["December", "September"],
            y=[by_mo.loc[12], by_mo.loc[9]],
            marker_color=[theme.MUTED, theme.MUTED],
            text=[f"{by_mo.loc[12]:.0f}%", f"{by_mo.loc[9]:.0f}%"],
            textposition="outside"))
        fig.update_yaxes(range=[0, 105], title="median capacity %")
        fig.update_layout(
            height=300,
            title=f"By month — a {by_mo.loc[12] - by_mo.loc[9]:.0f} point swing")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "Aggregation granularity destroys the signal",
            "The holiday peak is a <b>two-week</b> phenomenon. Averaged across the "
            "whole of December it is cancelled by the slow first three weeks of the "
            "month, leaving December and September with an <b>identical</b> median. "
            "A model given month-level calendar features would see no seasonality "
            "at all — which is why ours are built at ISO-week resolution."
        ),
        unsafe_allow_html=True,
    )


# ===========================================================================
# Tab 5 — Data quality
# ===========================================================================
with tab5:
    st.markdown(
        theme.section("A file with zero nulls is not a clean file",
                      "The raw CSV contains no missing values at all. That is the "
                      "problem: missingness has been encoded as zero."),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.kpi_row([
            ("Performances = 0", f"{int(DF['perf_missing'].sum()):,}",
             "impossible: attendance > 0", theme.ACCENT),
            ("Gross Potential = 0", f"{int(DF['gp_missing'].sum()):,}",
             "96% of all 1996 rows", theme.ACCENT),
            ("Capacity at 100%", f"{int(DF['capacity_censored'].sum()):,}",
             "right-censored", theme.GOLD),
            ("Weeks missing", "75", "mostly before 1993", theme.MUTED),
        ]),
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="medium")
    with a:
        q = (DF[DF["year"].between(*years)]
             .groupby("year")[["gp_missing", "perf_missing", "capacity_censored"]]
             .mean().mul(100).reset_index())
        fig = go.Figure()
        for col, name, colour in [
            ("gp_missing", "Gross Potential coded 0", theme.ACCENT),
            ("perf_missing", "Performances coded 0", theme.PRIMARY),
            ("capacity_censored", "Capacity censored at 100%", theme.GOLD),
        ]:
            fig.add_scatter(x=q["year"], y=q[col], mode="lines+markers", name=name,
                            line=dict(color=colour, width=2), marker=dict(size=5))
        fig.update_layout(height=360, yaxis_title="% of rows",
                          title="Missingness is an era effect, not a show effect")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)
    with b:
        usable = DF[(DF["performances"] > 0) & (DF["capacity_pct"] > 0)].copy()
        usable["implied"] = (usable["attendance"] / usable["performances"]
                             / (usable["capacity_pct"] / 100))
        keep = [t for t in PUBLISHED_SEATS if t in set(usable["theatre"])]
        vv = usable[usable["theatre"].isin(keep)]
        order = sorted(keep, key=lambda t: PUBLISHED_SEATS[t])
        fig = px.box(vv, x="theatre", y="implied", category_orders={"theatre": order},
                     points=False, labels={"theatre": "", "implied": "implied seats"})
        fig.update_traces(marker_color=theme.PRIMARY, line=dict(color=theme.PRIMARY))
        fig.add_scatter(x=order, y=[PUBLISHED_SEATS[t] for t in order],
                        mode="markers", name="published seat count",
                        marker=dict(symbol="diamond", size=9, color=theme.ACCENT))
        fig.update_layout(height=360,
                          title="Recovered seat counts vs published figures")
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CFG)

    st.markdown(
        theme.note(
            "Why this matters for modelling",
            "Because capacity is an <b>identity</b> over attendance, performances "
            "and seat count, the same-week columns reconstruct it to within "
            "<b>0.48 percentage points</b>. Handing a model same-week attendance "
            "lifts our band-classification accuracy from 0.638 to 0.902 — a "
            "fictitious result. <code>features.assert_no_leakage()</code> rejects "
            "any feature list containing those columns, and every modelling cell "
            "must pass it."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(theme.section("Cleaning decisions",
                              "Implemented as <code>CLEANING_DECISIONS</code> in "
                              "<code>src/data_load.py</code>, so code and "
                              "documentation cannot drift apart."),
                unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(CLEANING_DECISIONS).T.reset_index(names="Issue").rename(
            columns={"issue": "What we found", "evidence": "Evidence",
                     "decision": "Decision"}),
        hide_index=True, use_container_width=True,
    )

st.markdown("---")
st.caption(
    "IT5006 Fundamentals of Data Analytics · Group Project Phase 1 · "
    "National University of Singapore. Built with Streamlit; all views read "
    "through `src/data_load.py`. Source: "
    "https://github.com/gengyuzhu/IT5006-Group-Project-10"
)
