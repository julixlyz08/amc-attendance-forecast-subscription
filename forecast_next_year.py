"""
forecast_next_year.py
Forecasts U.S. attendance for the next four quarters and builds five scenarios around it.

Forecast = average of seasonal smoothing and "same quarter last year" (the blend).
Scenarios come from how far actual attendance landed from the blend in past tests
(four quarters ahead, tests from 2023 Q1):
    low / high          = 10th and 90th percentile of past misses (8 in 10 quarters landed inside)
    stress low / high   = the worst and best misses seen

Run after forecast_attendance.py (needs forecast_results.csv):
    python forecast_next_year.py

Independent project using publicly reported figures. Not affiliated with AMC Entertainment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_attendance import CSV_FILE, load_data, forecast_smoothing

RESULTS_FILE = "forecast_results.csv"
TEST_START = "2023 Q1"
FUTURE = ["2026 Q3", "2026 Q4", "2027 Q1", "2027 Q2"]
CHART_FOLDER = "charts"


def past_miss_percentages(path=RESULTS_FILE, start=TEST_START):
    """How far actual attendance was from the blend, as % of the blend."""
    r = pd.read_csv(path)
    r = r[r["quarter"] >= start]
    blend = (r["attendance | smoothing"] + r["attendance | same quarter last year"]) / 2
    return (r["actual"] - blend) / blend * 100


def main():
    df = load_data(CSV_FILE)
    last_quarter = df["quarter"].iloc[-1]

    # Blend forecast: smoothing (1 to 4 quarters ahead) averaged with the same quarter a year earlier
    smoothing = [forecast_smoothing(df, "attendance", h) for h in range(1, 5)]
    year_ago = [df.loc[df["quarter"] == q, "attendance"].iloc[0] for q in ("2025 Q3", "2025 Q4", "2026 Q1", "2026 Q2")]
    base = np.array([(s + y) / 2 for s, y in zip(smoothing, year_ago)])

    miss = past_miss_percentages()
    edges = {
        "stress low": miss.min(),
        "low": np.percentile(miss, 10),
        "base": 0.0,
        "high": np.percentile(miss, 90),
        "stress high": miss.max(),
    }
    print(f"Data ends {last_quarter}. Past misses used: {len(miss)} tests from {TEST_START}")
    print("Scenario adjustments (% of the base forecast):",
          {k: round(v, 1) for k, v in edges.items()})

    table = pd.DataFrame({"quarter": FUTURE})
    for name, pct in edges.items():
        table[name] = base * (1 + pct / 100)
    total = table.drop(columns="quarter").sum()
    table = pd.concat([table, pd.DataFrame([["Full year"] + total.tolist()], columns=table.columns)], ignore_index=True)
    table.round(1).to_csv("attendance_scenarios.csv", index=False)
    print("\nU.S. attendance, millions of guests:")
    print(table.round(1).to_string(index=False))

    # Chart
    os.makedirs(CHART_FOLDER, exist_ok=True)
    hist = df[df["quarter"] >= "2022 Q1"]
    labels = list(hist["quarter"]) + FUTURE
    xh = np.arange(len(hist))
    xf = np.arange(len(hist), len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(xh, hist["attendance"], color="black", marker="o", markersize=4, linewidth=2, label="actual")
    # connect the last actual point to the forecast so the line reads continuously
    xs = np.concatenate([[xh[-1]], xf])
    ax.fill_between(xs, np.concatenate([[hist["attendance"].iloc[-1]], table["stress low"][:4]]),
                    np.concatenate([[hist["attendance"].iloc[-1]], table["stress high"][:4]]),
                    color="#1f4e79", alpha=0.12, label="stress range (worst and best past misses)")
    ax.fill_between(xs, np.concatenate([[hist["attendance"].iloc[-1]], table["low"][:4]]),
                    np.concatenate([[hist["attendance"].iloc[-1]], table["high"][:4]]),
                    color="#1f4e79", alpha=0.28, label="low to high (8 in 10 past quarters)")
    ax.plot(xs, np.concatenate([[hist["attendance"].iloc[-1]], base]), color="#1f4e79", marker="o",
            markersize=4, linewidth=2, linestyle="--", label="base forecast (blend)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_ylabel("Millions of guests")
    ax.set_title("U.S. attendance: actual and forecast for the next four quarters", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    path = os.path.join(CHART_FOLDER, "attendance_scenarios.png")
    plt.savefig(path, dpi=150)
    print("\nSaved chart:", path, "| saved table: attendance_scenarios.csv")


if __name__ == "__main__":
    main()
