"""
forecast_attendance.py
Tests how well different models forecast AMC U.S. attendance four quarters ahead.

How the test works: for each quarter from 2022 Q1 to the latest, pretend it is four
quarters earlier, give each model only the data available then, ask it to predict the
quarter, and compare with what really happened.

Run from the project folder (amc_quarterly.csv and movies_clean.csv must be in it):
    pip install pandas numpy matplotlib statsmodels
    python forecast_attendance.py

Models marked with * were added AFTER seeing the first round of test results
(the original regression kept over-predicting once attendance settled below 2019 levels).
The "release:" models add the release calendar (from the TMDB movie data of the
movie-profitability project) to the post-pandemic regression.

This product uses the TMDB API but is not endorsed or certified by TMDB.
Independent project using publicly reported figures. Not affiliated with AMC Entertainment.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

warnings.filterwarnings("ignore")

# ---------- Choices made for this project ----------
CSV_FILE = "amc_quarterly.csv"
MOVIES_FILE = "movies_clean.csv"      # from the movie-profitability project (TMDB data)
FIRST_TEST = "2022 Q1"          # test every quarter from here on
LATER_TEST_START = "2023 Q1"    # second accuracy figure, once models have seen some post-pandemic quarters
HORIZON = 4                     # forecast made 4 quarters before the quarter being predicted
PANDEMIC = ["2020 Q1", "2020 Q2", "2020 Q3", "2020 Q4",
            "2021 Q1", "2021 Q2", "2021 Q3", "2021 Q4"]   # flagged in the regression, removed for the smoothing model
POST_PANDEMIC_START = "2022 Q1"  # used only by the models marked *
CARMIKE_CODE = {"before": 0.0, "closing_quarter": 11 / 92, "after": 1.0}   # Carmike deal closed Dec 21, 2016
BIG_BUDGET = 100_000_000        # a "big-budget" film
CHART_FOLDER = "charts"

# regression versions: name -> (include time trend, include post-pandemic marker)
REGRESSIONS = {
    "regression": (False, False),
    "regression + trend": (True, False),
    "regression + post-pandemic marker*": (False, True),
    "regression + post-pandemic marker + trend*": (True, True),
}
BASE_MODEL = "regression + post-pandemic marker*"

# release-calendar features added to the post-pandemic regression (no trend)
FEATURE_SETS = {
    "big-budget films": ["big100"],
    "franchise films": ["franchise"],
    "total budget": ["budget_bn"],
    "big-budget + franchise": ["big100", "franchise"],
}
# with and without the films flagged as likely streaming / limited releases
FILM_VERSIONS = {"theatrical only": "theatrical", "incl. streaming": "all"}
# ----------------------------------------------------


def load_release_features(quarters, path=MOVIES_FILE):
    """Per quarter: number of $100M+ films, number of franchise films, total budget ($ billions)."""
    m = pd.read_csv(path)
    m["release_date"] = pd.to_datetime(m["release_date"])
    m["quarter"] = m["release_date"].dt.year.astype(str) + " Q" + m["release_date"].dt.quarter.astype(str)
    out = pd.DataFrame(index=quarters)
    for version, key in (("theatrical", "theatrical"), ("all", "all")):
        sub = m if version == "all" else m[m["release_type"] == "Theatrical"]
        out[f"big100_{key}"] = sub[sub["budget"] >= BIG_BUDGET].groupby("quarter").size()
        out[f"franchise_{key}"] = sub[sub["in_franchise"] == "Franchise"].groupby("quarter").size()
        out[f"budget_bn_{key}"] = sub.groupby("quarter")["budget"].sum() / 1e9
    return out.fillna(0.0)


def load_data(path=CSV_FILE):
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = raw.copy()
    for col in [c for c in raw.columns if c.startswith("us_")]:
        cleaned = raw[col].str.strip().str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(cleaned.replace({"-": np.nan, "": np.nan}), errors="coerce")
    df["attendance"] = df["us_attendance_millions"]
    df["guests_per_screen"] = df["attendance"] * 1_000_000 / df["us_screens"]
    df["screens"] = df["us_screens"]
    df["q"] = df["quarter"].str[-1].astype(int)
    df["idx"] = np.arange(len(df))                      # calendar position (keeps the 2020 Q2 gap)
    df["pandemic"] = df["quarter"].isin(PANDEMIC).astype(float)
    start_idx = df.index[df["quarter"] == POST_PANDEMIC_START][0]
    df["post_pandemic"] = (df["idx"] >= start_idx).astype(float)
    df["carmike"] = df["carmike_period"].map(CARMIKE_CODE)
    feats = load_release_features(df["quarter"])
    df = df.join(feats, on="quarter")
    return df.reset_index(drop=True)


# ---------- Models ----------
def design_matrix(rows, with_trend, with_post, extras=()):
    cols = [np.ones(len(rows))]
    cols += [(rows["q"] == k).astype(float).to_numpy() for k in (2, 3, 4)]
    cols += [rows["pandemic"].to_numpy(), rows["carmike"].to_numpy()]
    if with_post:
        cols.append(rows["post_pandemic"].to_numpy())
    if with_trend:
        cols.append(rows["idx"].to_numpy().astype(float))
    for name in extras:
        cols.append(rows[name].to_numpy().astype(float))
    return np.column_stack(cols)


def fit_regression(train, target, with_trend, with_post, extras=()):
    """Regression on logs: quarter of year + pandemic marker + Carmike marker (+ extras)."""
    train = train.dropna(subset=[target])
    X = design_matrix(train, with_trend, with_post, extras)
    y = np.log(train[target].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)   # a marker with no training examples gets effect 0
    return beta


def predict_regression(beta, row, with_trend, with_post, extras=()):
    x = design_matrix(pd.DataFrame([row]), with_trend, with_post, extras)
    return float(np.exp(x @ beta)[0])


def forecast_smoothing(train, target, h):
    """Seasonal smoothing (ETS). Pandemic quarters removed and the series joined up.
    Settings are chosen automatically at each step from a small set, by AIC."""
    series = train.loc[~train["quarter"].isin(PANDEMIC), target].dropna().to_numpy()
    best, best_aic = None, np.inf
    for error in ("add", "mul"):
        for trend, damped in ((None, False), ("add", False), ("add", True)):
            for seasonal in ("add", "mul"):
                try:
                    res = ETSModel(series, error=error, trend=trend, damped_trend=damped,
                                   seasonal=seasonal, seasonal_periods=4).fit(disp=False)
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best, best_aic = res, res.aic
                except Exception:
                    continue
    if best is None:
        return np.nan
    return float(best.forecast(h)[-1])


def release_models():
    """name -> list of feature columns, for every release-calendar variant."""
    models = {}
    for fname, bases in FEATURE_SETS.items():
        for vname, key in FILM_VERSIONS.items():
            models[f"release: {fname} ({vname})"] = [f"{b}_{key}" for b in bases]
    return models


# ---------- Rolling test ----------
def run_tests(df):
    first = df.index[df["quarter"] == FIRST_TEST][0]
    rel = release_models()
    rows = []
    for t in range(first, len(df)):
        target_row = df.iloc[t]
        origin = t - HORIZON
        train = df.iloc[: origin + 1]
        screens_known = df.loc[origin, "screens"]            # last known screen count at forecast time
        out = {"quarter": target_row["quarter"], "actual": target_row["attendance"]}

        # steps ahead inside the joined-up series (pandemic quarters skipped)
        last_ok = train.loc[~train["quarter"].isin(PANDEMIC)].index.max()
        h_join = sum(1 for k in range(last_ok + 1, t + 1) if df.loc[k, "quarter"] not in PANDEMIC)

        for target, label in (("attendance", "attendance"), ("guests_per_screen", "per-screen")):
            to_attendance = (lambda v: v) if target == "attendance" else (lambda v: v * screens_known / 1_000_000)

            out[f"{label} | same quarter last year"] = to_attendance(df.loc[t - 4, target])

            for name, (with_trend, with_post) in REGRESSIONS.items():
                beta = fit_regression(train, target, with_trend, with_post)
                out[f"{label} | {name}"] = to_attendance(predict_regression(beta, target_row, with_trend, with_post))

            for name, extras in rel.items():
                beta = fit_regression(train, target, False, True, extras)
                out[f"{label} | {name}"] = to_attendance(predict_regression(beta, target_row, False, True, extras))

            out[f"{label} | smoothing"] = to_attendance(forecast_smoothing(train, target, h_join))
        rows.append(out)
    return pd.DataFrame(rows)


def accuracy_table(results, start=None, release=False):
    sub = results if start is None else results[results["quarter"] >= start]
    table = []
    for col in [c for c in results.columns if " | " in c]:
        if col == "per-screen | same quarter last year":
            continue            # identical to the attendance version (screens held at last known count)
        if ("release:" in col) != release:
            continue
        err = sub[col] - sub["actual"]
        name = col.replace("attendance | same quarter last year", "(either target) | same quarter last year")
        table.append({
            "model": name,
            "avg_miss_millions": err.abs().mean(),
            "avg_miss_pct": (err.abs() / sub["actual"]).mean() * 100,
            "avg_bias_millions": err.mean(),
        })
    return pd.DataFrame(table).round(1).sort_values("avg_miss_pct").reset_index(drop=True)


def release_comparison(results):
    """Average miss of each release model minus the base model (negative = better), in percentage points."""
    def miss(col, start):
        sub = results if start is None else results[results["quarter"] >= start]
        return ((sub[col] - sub["actual"]).abs() / sub["actual"]).mean() * 100

    rows = []
    for name in release_models():
        row = {"release feature": name.replace("release: ", "")}
        for label in ("attendance", "per-screen"):
            for start, tag in ((None, "all"), (LATER_TEST_START, "from 2023")):
                diff = miss(f"{label} | {name}", start) - miss(f"{label} | {BASE_MODEL}", start)
                row[f"{label}, {tag}"] = round(diff, 1)
        rows.append(row)
    base = {"release feature": "(base: post-pandemic regression, miss %)"}
    for label in ("attendance", "per-screen"):
        for start, tag in ((None, "all"), (LATER_TEST_START, "from 2023")):
            base[f"{label}, {tag}"] = round(miss(f"{label} | {BASE_MODEL}", start), 1)
    return pd.DataFrame([base] + rows)


def describe_full_fit(df):
    """What each regression learned from all the data (for the write-up)."""
    for name, (with_trend, with_post) in REGRESSIONS.items():
        beta = fit_regression(df, "attendance", with_trend, with_post)
        names = ["Q2 vs Q1", "Q3 vs Q1", "Q4 vs Q1", "pandemic", "Carmike"]
        names += ["post-pandemic era"] if with_post else []
        names += ["per quarter trend"] if with_trend else []
        print(f"\n{name}: effect on attendance (fitted on all data)")
        for n, b in zip(names, beta[1:]):
            print(f"  {n:18s} {np.expm1(b) * 100:+6.1f}%")

    print("\nRelease features (added to the post-pandemic regression): effect on attendance of one more unit")
    for name, extras in release_models().items():
        beta = fit_regression(df, "attendance", False, True, extras)
        effects = ", ".join(f"{e.rsplit('_', 1)[0]} {np.expm1(b) * 100:+.1f}%" for e, b in zip(extras, beta[-len(extras):]))
        print(f"  {name:55s} {effects}")


def make_chart(results):
    os.makedirs(CHART_FOLDER, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    x = np.arange(len(results))
    styles = {
        "same quarter last year": ("#999999", "--"),
        "smoothing": ("#e67e22", "-"),
        "regression + trend": ("#8e24aa", "-"),
        "regression + post-pandemic marker + trend*": ("#8e24aa", ":"),
        "regression": ("#2e7d32", "-"),
        "regression + post-pandemic marker*": ("#2e7d32", ":"),
    }
    for ax, label, title in ((axes[0], "attendance", "Forecasting attendance directly"),
                             (axes[1], "per-screen", "Forecasting guests per screen, converted to attendance")):
        ax.plot(x, results["actual"], color="black", linewidth=2.5, marker="o", markersize=4, label="actual")
        for name, (color, ls) in styles.items():
            ax.plot(x, results[f"{label} | {name}"], color=color, linestyle=ls, linewidth=1.5, label=name)
        ax.set_title(title + " (each forecast made 4 quarters ahead)", loc="left", fontsize=11)
        ax.set_ylabel("Millions of guests")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=7.5, loc="upper right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results["quarter"], rotation=70, fontsize=8)
    fig.text(0.01, 0.005, "* added after seeing the first round of results. Dotted lines = versions with the post-pandemic marker. "
             "Vertical axis capped at 100 million.", fontsize=8, color="#555555")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    path = os.path.join(CHART_FOLDER, "forecast_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_release_chart(results):
    """Average miss from 2023 Q1 for the base regression and each release-feature version."""
    sub = results[results["quarter"] >= LATER_TEST_START]
    names = [BASE_MODEL] + list(release_models())
    labels = ["base (no release data)"] + [n.replace("release: ", "") for n in release_models()]
    fig, ax = plt.subplots(figsize=(11, 6))
    ypos = np.arange(len(names))
    for offset, label, color in ((-0.2, "attendance", "#1f4e79"), (0.2, "per-screen", "#e67e22")):
        vals = [((sub[f"{label} | {n}"] - sub["actual"]).abs() / sub["actual"]).mean() * 100 for n in names]
        ax.barh(ypos + offset, vals, height=0.38, color=color, label=f"forecasting {label}")
    for label, ls in (("same quarter last year", "--"), ("smoothing", ":")):
        v = ((sub[f"attendance | {label}"] - sub["actual"]).abs() / sub["actual"]).mean() * 100
        ax.axvline(v, color="#555555", linestyle=ls, linewidth=1.2, label=f"{label} ({v:.1f}%)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Average miss, % (4 quarters ahead, tests from {LATER_TEST_START}); lower is better")
    ax.set_title("Does the release calendar help the post-pandemic regression?", loc="left", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(CHART_FOLDER, "release_features_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    df = load_data()
    results = run_tests(df)
    results.round(2).to_csv("forecast_results.csv", index=False)

    print(f"Tests: {len(results)} quarters, {results['quarter'].iloc[0]} to {results['quarter'].iloc[-1]}, "
          f"each forecast {HORIZON} quarters ahead")
    print("* = added after seeing the first round of results\n")
    print(f"=== ACCURACY, ALL TESTS (from {FIRST_TEST}) ===")
    print(accuracy_table(results).to_string(index=False))
    print(f"\n=== ACCURACY, FROM {LATER_TEST_START} ===")
    print(accuracy_table(results, LATER_TEST_START).to_string(index=False))

    print("\n=== RELEASE FEATURES vs THE BASE POST-PANDEMIC REGRESSION ===")
    print("(differences in percentage points of average miss; negative = the release data helped)")
    print(release_comparison(results).to_string(index=False))

    describe_full_fit(df)
    print("\nSaved charts:", make_chart(results), "and", make_release_chart(results), "| saved table: forecast_results.csv")


if __name__ == "__main__":
    main()
