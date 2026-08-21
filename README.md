# IT5006 Phase 1 — Forecasting Weekly Demand for Broadway Productions

**Group 10** · Pan Pinyou · Li Yudan · Zhu Gengyu
National University of Singapore · IT5006 Fundamentals of Data Analytics

Literature review, exploratory data analysis, and an interactive dashboard over
26 years of Broadway weekly box-office data.

**Deliverables**

| Item | Location |
|---|---|
| Report (PDF) | [`report/main.pdf`](report/main.pdf) — 5 body pages + references + appendix |
| Notebooks | [`notebooks/*.ipynb`](notebooks/) — four, executed with outputs |
| Dashboard | **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)** — source in [`dashboard/`](dashboard/) |
| Reusable code | [`src/`](src/) |

---

## Quick start

The base Anaconda install on the original machine is broken (`numpy 2.2.6`
alongside `pandas`/`scipy`/`matplotlib` compiled against numpy 1.x, so
`import pandas` raises `ValueError: numpy.dtype size changed`). **Always work
inside the virtual environment**, never the base conda env.

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

Two dependency files, and they do **not** overlap — `requirements-dev.txt`
pulls the runtime set in with `-r requirements.txt` rather than restating it, so
no package version is written twice:

| File | Adds | Installed by |
|---|---|---|
| `requirements.txt` | streamlit, pandas, numpy, plotly | Streamlit Cloud — exactly what `dashboard/app.py` imports |
| `requirements-dev.txt` | `-r requirements.txt` **+** scikit-learn, scipy, matplotlib, Jupyter | you, for the notebooks and figures |

The split exists because Streamlit Cloud installs `requirements.txt` and nothing
else. Merging the two would put the whole analysis stack back into the
deployment — which is what broke it: a pinned `pyarrow` the project never
imported aborted the build outright. Four packages instead of ~130 makes the
build fast and leaves almost nothing that can fail. A guard test confirms the
dashboard's entire import graph runs with every dev-only package made
unimportable.

Then, in order:

```bash
python src/report_stats.py
```

```bash
streamlit run dashboard/app.py
```

---

## What the analysis found

The dataset is `broadway_clean.csv`: 29,167 rows, one per **week × show ×
theatre**, spanning 1990-08-26 to 2016-08-14 across 820 shows and 56 theatres.

Five findings shape everything downstream:

1. **A file with zero nulls is not a clean file.** 2,309 rows (7.9%) report zero
   performances alongside positive attendance — structurally impossible — and
   1,918 rows (6.6%) code gross potential as zero, including 96% of all 1996
   rows. Both cluster by *reporting era*, not by show.
2. **`Statistics.Capacity` is an undocumented percentage, not a seat count.**
   Inverting `capacity = attendance / (seats × performances)` recovers seat
   counts matching published figures to a median error of **1.7%** (the
   Lunt-Fontanne recovers exactly its published 1,509 seats).
3. **Nominal revenue growth is a price effect.** Median ticket price rises 109%
   between 1996 and 2015 while capacity utilisation moves from 82% to 85%.
4. **Seasonality exists only at week resolution.** ISO weeks 52–53 hit 90%
   capacity against 73% in weeks 36 and 44 — a 17-point swing. Measured
   *monthly*, that contrast is exactly **zero**, because the holiday peak is a
   two-week phenomenon cancelled by the slow first three weeks of December.
5. **Leakage is the central methodological risk.** Capacity is an identity over
   attendance, performances and seat count, and reconstructs from same-week
   columns to **0.48 percentage points**.

### The leakage guard

Because of finding 5, `features.assert_no_leakage()` rejects any feature list
containing a same-week outcome column, and every modelling cell must pass it.
The cost of ignoring it is measured directly in notebook 04:

| Feature set | Test accuracy |
|---|---|
| Leakage-safe (what we report) | **0.638** |
| \+ same-week attendance | 0.875 |
| \+ same-week attendance and performances | 0.902 |

Those last two numbers are not achievements; they are what this dataset's
failure mode looks like.

---

## Candidate problems

Three were explored against the brief's scoping checklist.

**Candidate 1 — demand for a running production, 4 weeks ahead (primary).**
One question framed twice: regression on capacity utilisation, and
classification into three bands at training-set terciles. The horizon was chosen
with evidence rather than tuned — capacity has lag-1 autocorrelation 0.84, so at
one week ahead persistence scores 0.728 and a model adds nothing, but a one-week
forecast cannot change a marketing commitment either:

| Horizon | Persistence acc. | Model acc. | Persistence MAE | Model MAE |
|---|---|---|---|---|
| 1 week | **0.728** | 0.724 | 5.52 | 5.33 |
| 2 weeks | 0.634 | **0.656** | 7.21 | 6.52 |
| 4 weeks | 0.607 | **0.638** | 8.01 | 7.36 |

At four weeks the classifier reaches **0.638** against a 0.315 majority baseline
and 0.607 persistence baseline; the regression reaches MAE 7.36 / R² 0.546
against 8.01 / 0.399.

**Candidate 2 — closure risk within 8 weeks (secondary).** 22.1% positive, with
208 rows right-censored and labelled missing rather than defaulted to "no".
PR-AUC **0.698** against a 0.204 no-skill floor, ROC-AUC 0.887, minority recall
0.806. Its accuracy of 0.813 is *below* the 0.796 scored by always predicting
"stays open" — a rule that catches none of the closures it exists to find.

**Candidate 3 — opening-month gross (deprioritised).** R² 0.292. Without lag
features little remains, and the drivers the literature identifies for a new
show (advance sales, casting, marketing spend, reviews) are absent from this
dataset. Documenting the boundary beat forcing a weak model.

We expect a tuned Phase 2 classifier in the **low-to-mid 60s**. A result above
90% on this target would indicate leakage, not skill.

---

## Repository layout

```
├── data/raw/broadway_clean.csv        read-only source
├── notebooks/
│   ├── 01_data_audit.ipynb            data contract, quality audit, capacity validation
│   ├── 02_eda_temporal.ipynb          trend, seasonality, exogenous shocks
│   ├── 03_eda_entities.ipynb          shows, theatres, survival curves, correlations
│   └── 04_candidate_problems.ipynb    horizon selection, leakage demo, three candidates
├── src/
│   ├── config.py                      seeds, paths, windows, band cuts, plot theme
│   ├── data_load.py                   single data entry point + CLEANING_DECISIONS
│   ├── features.py                    leakage-annotated feature builders + the guard
│   ├── viz.py                         shared plot style, vector PDF export
│   └── report_stats.py                regenerates every number quoted in the report
├── dashboard/
│   ├── app.py                         Streamlit, five linked views
│   └── theme.py                       plotly template + CSS design system
├── .streamlit/config.toml             widget theme matched to theme.py
├── requirements.txt                   dashboard runtime (what Cloud installs)
├── requirements-dev.txt               + analysis stack for the notebooks
└── report/
    ├── main.pdf                       the submitted report
    └── figures/                       11 vector PDFs, generated by the notebooks
```


## Reproducibility

- `SEED = 42` everywhere; both requirements files pin exact versions on
  Python <= 3.12, the interpreter every reported number was produced on.
- `data_load.load_raw()` asserts the raw data contract (29,167 × 12, natural key
  uniqueness, all dates Sunday). Downstream numbers are protected by it.
- Every figure is generated into `report/figures/` as vector PDF by the
  notebooks — none is pasted in by hand.
- `python src/report_stats.py` recomputes every numeric claim in the report;
  diff it against the manuscript before submitting.
- All 19 bibliography entries were verified against publisher records
  (DOI, volume, issue, pages) rather than quoted from memory.

## Dashboard

Five linked views, all reading through `src/data_load.py`:

| Tab | What it shows |
|---|---|
| **Overview** | Market gross / attendance / productions with the three exogenous shocks annotated |
| **Productions** | Compare shows on calendar time *or* aligned by week of run; run-length distribution |
| **Venues** | Recovered seat count against utilisation; highest-grossing houses |
| **Seasonality** | Week × year heatmap, plus a side-by-side showing month-level aggregation destroying the signal |
| **Data quality** | Zero-coded missingness over time, the seat-count validation, and the cleaning contract |

### Deploying

Live at **[it5006-group-project-10.streamlit.app](https://it5006-group-project-10.streamlit.app)**,
built from `main` / `dashboard/app.py` on **Python 3.12**.

> **Keep the Python version at 3.12 or lower** (*App settings → General*).
> `numpy==1.26.4` publishes wheels only up to `cp312`. On Python 3.13+ pip finds
> no wheel and tries to compile numpy from source; the build dies and the app
> then hangs on *"Your app is in the oven"* indefinitely, with the real error
> buried in the build log rather than shown in the app. `requirements.txt`
> carries environment markers that fall back to a wheel-backed numpy on newer
> interpreters, but 3.12 is what the reported numbers were validated against —
> the deployment log confirms it installs the exact pins
> (`streamlit==1.39.0`, `pandas==2.2.3`, `numpy==1.26.4`, `plotly==5.24.1`).

## Data source

Broadway weekly grosses, distributed for IT5006 via Canvas as
`broadway_clean.csv`; originally compiled by the CORGIS Dataset Project from
data published by The Broadway League.
