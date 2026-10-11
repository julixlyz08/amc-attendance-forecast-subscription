"""
compare_candidates.py
Compares the candidate forecasts that could feed the subscription model, using the
four-quarters-ahead test results saved by forecast_attendance.py (forecast_results.csv).

Candidates:
  A  seasonal smoothing (attendance directly)
  B  same quarter last year
  C  blend: the average of A and B
  D1 post-pandemic regression + big-budget films, theatrical only (attendance directly)
  D2 the same regression, forecasting guests per screen

Run after forecast_attendance.py:
    python compare_candidates.py

This product uses the TMDB API but is not endorsed or certified by TMDB.
Independent project using publicly reported figures. Not affiliated with AMC Entertainment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FILE = "forecast_results.csv"
LATER_START = "2023 Q1"
CHART_FOLDER = "charts"


def build_candidates(r):
    c = pd.DataFrame({"quarter": r["quarter"], "actual": r["actual"]})
    c["A smoothing"] = r["attendance | smoothing"]
    c["B same quarter last year"] = r["attendance | same quarter last year"]
    c["C blend of A and B"] = (c["A smoothing"] + c["B same quarter last year"]) / 2
    c["D1 regression + big-budget films"] = r["attendance | release: big-budget films (theatrical only)"]
    c["D2 same, per-screen"] = r["per-screen | release: big-budget films (theatrical only)"]
    return c


def scoreboard(c, start=None):
    sub = c if start is None else c[c["quarter"] >= start]
    base_err = (sub["B same quarter last year"] - sub["actual"]).abs()
    rows = []
    for col in [x for x in c.columns if x not in ("quarter", "actual")]:
        err = sub[col] - sub["actual"]
        rows.append({
            "candidate": col,
            "avg_miss_pct": (err.abs() / sub["actual"]).mean() * 100,
            "avg_miss_millions": err.abs().mean(),
            "worst_miss_pct": (err.abs() / sub["actual"]).max() * 100,
            "avg_bias_millions": err.mean(),
            "quarters_closer_than_B": int((err.abs() < base_err).sum()),
            "tests": len(sub),
        })
    return pd.DataFrame(rows).round(1)


def make_chart(c):
    sub = c[c["quarter"] >= LATER_START]
    os.makedirs(CHART_FOLDER, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(sub))
    colors = {"A smoothing": "#e67e22", "B same quarter last year": "#999999",
              "C blend of A and B": "#1f4e79", "D1 regression + big-budget films": "#2e7d32",
              "D2 same, per-screen": "#8e24aa"}
    for col, color in colors.items():
        pct = (sub[col] - sub["actual"]) / sub["actual"] * 100
        ax.plot(x, pct, marker="o", markersize=4, linewidth=1.6, color=color, label=col)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["quarter"], rotation=70, fontsize=8)
    ax.set_ylabel("Forecast minus actual, % of actual")
    ax.set_title("How far off was each candidate? (4 quarters ahead; above 0 = too high)", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(CHART_FOLDER, "candidate_errors.png")
    plt.savefig(path, dpi=150)
    return path


def main():
    r = pd.read_csv(RESULTS_FILE)
    c = build_candidates(r)
    print(f"=== ALL TESTS (from {c['quarter'].iloc[0]}) ===")
    print(scoreboard(c).to_string(index=False))
    print(f"\n=== FROM {LATER_START} ===")
    later = scoreboard(c, LATER_START)
    print(later.to_string(index=False))
    later.to_csv("candidate_scoreboard.csv", index=False)
    print("\nSaved chart:", make_chart(c), "| saved table: candidate_scoreboard.csv")


if __name__ == "__main__":
    main()
