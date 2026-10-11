"""
explore_quarterly.py
Looks at the AMC quarterly table before any forecasting: how attendance moves over
time, how seasonal it is, and how it compares with the pre-pandemic years.

Run from the project folder:
    pip install pandas matplotlib
    python explore_quarterly.py

Independent project using publicly reported figures. Not affiliated with AMC Entertainment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILE = "amc_quarterly.csv"   # change if you rename the file
CHART_FOLDER = "charts"


def load_data(path=CSV_FILE):
    """Read the table, tolerating stray spaces, thousands commas, and '-' for no data."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = raw.copy()
    for col in [c for c in raw.columns if c.startswith("us_")]:
        cleaned = raw[col].str.strip().str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(cleaned.replace({"-": np.nan, "": np.nan}), errors="coerce")

    df["year"] = df["quarter"].str[:4].astype(int)
    df["q"] = df["quarter"].str[-1].astype(int)
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": (df["q"] - 1) * 3 + 1, "day": 1}
    )
    # Guests per screen: a way to compare quarters when the number of screens changes
    df["guests_per_screen"] = df["us_attendance_millions"] * 1_000_000 / df["us_screens"]
    return df


def main():
    df = load_data()
    print(f"{len(df)} quarters, {df['quarter'].iloc[0]} to {df['quarter'].iloc[-1]}")
    print("Quarters with no attendance figure:", df.loc[df["us_attendance_millions"].isna(), "quarter"].tolist())

    # 1. How seasonal is it? Share of each year's attendance that falls in each quarter (2015-2019)
    pre = df[(df["year"] >= 2015) & (df["year"] <= 2019)]
    yearly = pre.groupby("year")["us_attendance_millions"].transform("sum")
    share = (pre["us_attendance_millions"] / yearly * 100).groupby(pre["q"]).mean().round(1)
    print("\n=== SEASONALITY: average share of the year's attendance, 2015-2019 ===")
    print(share.rename(lambda q: f"Q{q}").to_string())

    # 2. Recovery compared with the same quarter in 2019
    base = df[df["year"] == 2019].set_index("q")
    print("\n=== ATTENDANCE VS SAME QUARTER IN 2019 (100 = same as 2019) ===")
    rows = []
    for _, r in df[df["year"] >= 2022].iterrows():
        b = base.loc[r["q"]]
        rows.append({
            "quarter": r["quarter"],
            "attendance_index": round(r["us_attendance_millions"] / b["us_attendance_millions"] * 100),
            "screens_index": round(r["us_screens"] / b["us_screens"] * 100),
            "guests_per_screen_index": round(r["guests_per_screen"] / b["guests_per_screen"] * 100),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # 3. Year over year change, post-pandemic
    df = df.sort_values("date").reset_index(drop=True)
    df["yoy_pct"] = (df["us_attendance_millions"] / df["us_attendance_millions"].shift(4) - 1) * 100
    print("\n=== YEAR-OVER-YEAR CHANGE IN ATTENDANCE, 2023 onward (%) ===")
    print(df.loc[df["year"] >= 2023, ["quarter", "us_attendance_millions", "yoy_pct"]].round(1).to_string(index=False))

    # Charts
    os.makedirs(CHART_FOLDER, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, col, title in [
        (axes[0], "us_attendance_millions", "U.S. attendance per quarter (millions of guests)"),
        (axes[1], "guests_per_screen", "Guests per screen per quarter (adjusts for the number of screens)"),
    ]:
        ax.plot(df["date"], df[col], marker="o", markersize=4, color="#1f4e79")
        ax.axvline(pd.Timestamp("2016-12-21"), color="#c0392b", linestyle="--", linewidth=1)
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="#999999", alpha=0.15)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].text(pd.Timestamp("2016-12-21"), axes[0].get_ylim()[1] * 0.97, " Carmike deal closes",
                 color="#c0392b", fontsize=9, va="top")
    axes[0].text(pd.Timestamp("2020-02-01"), axes[0].get_ylim()[1] * 0.97, " 2020-2021 pandemic years",
                 color="#555555", fontsize=9, va="top")
    axes[1].set_xlabel("Quarter (gap in 2020 Q2: theatres closed, no figures reported)")
    plt.tight_layout()
    out = os.path.join(CHART_FOLDER, "attendance_over_time.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved chart: {out}")


if __name__ == "__main__":
    main()
