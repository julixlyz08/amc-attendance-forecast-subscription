"""
subscription_model.py
An illustrative model of a movie subscription (AMC Stubs A-List style).

Questions it answers
  1. What is a member worth per year, compared with the same person without the subscription?
  2. How long until a new member pays back the cost of winning them (with cancellations)?
  3. How many members could AMC lose before the July 2026 price increase stops paying?
  4. What-if: how does the "best price" change with how members react to price?
  5. Do the answers hold under the five attendance scenarios?
  6. Which assumptions matter most?

Two ways of paying studios on member visits are always shown side by side:
  Method A: as if every member visit were a full-price ticket
  Method B: on the member's per-visit share of the subscription fee (annual fee / visits)

Run (needs assumptions.csv and attendance_scenarios.csv in the same folder):
    pip install pandas numpy matplotlib
    python subscription_model.py

This is an illustrative model built on public figures and labeled assumptions, not AMC's
real accounting. Independent project, not affiliated with AMC Entertainment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSUMPTIONS_FILE = "assumptions.csv"
SCENARIOS_FILE = "attendance_scenarios.csv"
RESULTS_FOLDER = "results"
CHART_FOLDER = "charts"
TYPES = ["light", "typical", "heavy"]
METHODS = {"A": "A: studios paid as if each visit were a full-price ticket",
           "B": "B: studios paid on the member's per-visit share of the fee"}
SCENARIOS = ["stress low", "low", "base", "high", "stress high"]
PAYBACK_HORIZON = 60        # months


# ---------- Inputs ----------
def load_assumptions(path=ASSUMPTIONS_FILE):
    a = pd.read_csv(path).set_index("key")
    return {case: a[case].dropna().astype(float).to_dict() for case in ("low", "base", "high")}


def load_attendance(path=SCENARIOS_FILE):
    t = pd.read_csv(path).set_index("quarter")
    return t.loc["Full year", SCENARIOS].astype(float).to_dict()      # millions of guests, next four quarters


def params(table, overrides=None):
    p = dict(table["base"])
    p.update(overrides or {})
    return p


# ---------- Model ----------
def type_values(p, method, attendance_m, price=None):
    """Yearly value of one member of each type: profit with the subscription minus profit without it."""
    price = p["price_now"] if price is None else price
    mix_avg = sum(p[f"share_{t}"] * p[f"visits_{t}"] for t in TYPES)
    # visits move with total attendance and with the members' share of it
    scale = (p["member_share"] * attendance_m * 1e6 / p["members"]) / mix_avg
    fee = 12 * price
    snack_profit_member = p["snack_member"] * (1 - p["snack_cost"])
    snack_profit_other = p["snack_avg"] * (1 - p["snack_cost"])
    out = {}
    for t in TYPES:
        visits = p[f"visits_{t}"] * scale
        film = p["film_share"] * (p["ticket"] if method == "A" else fee / visits)
        with_membership = fee + visits * (snack_profit_member - film - p["perks"])
        baseline_visits = visits * (1 - p["extra"])      # visits that would happen anyway
        without = baseline_visits * (p["ticket"] * (1 - p["film_share"]) + snack_profit_other)
        out[t] = with_membership - without
    return out


def average_value(p, values):
    return sum(p[f"share_{t}"] * values[t] for t in TYPES)


def program_contribution(p, method, attendance_m, price=None, members_fraction=1.0):
    """Yearly contribution of the whole program, in $ millions."""
    v = type_values(p, method, attendance_m, price)
    return p["members"] * members_fraction * average_value(p, v) / 1e6


def payback_months(value_per_year, monthly_churn, cac):
    """Months until the expected profit from a new member covers the cost of winning them."""
    monthly = value_per_year / 12
    if monthly <= 0:
        return np.nan
    total = 0.0
    for month in range(1, PAYBACK_HORIZON + 1):
        total += monthly * (1 - monthly_churn) ** (month - 1)
        if total >= cac:
            return month
    return np.nan


def five_year_value(value_per_year, monthly_churn):
    monthly = value_per_year / 12
    return sum(monthly * (1 - monthly_churn) ** (m - 1) for m in range(1, PAYBACK_HORIZON + 1))


def break_even_loss(p, method, attendance_m):
    """Share of members AMC could lose after the July 2026 increase before the program earns less than before."""
    v0 = type_values(p, method, attendance_m, p["price_before"])
    v1 = type_values(p, method, attendance_m, p["price_now"])
    avg0, avg1 = average_value(p, v0), average_value(p, v1)
    gain = sum(p[f"share_{t}"] * (v1[t] - v0[t]) for t in TYPES)       # extra profit per member from the higher fee
    proportional = np.nan if (avg0 <= 0 or avg1 <= 0) else float(np.clip(1 - avg0 / avg1, 0, 1))
    light_first = np.nan if v1["light"] <= 0 else float(np.clip(gain / v1["light"], 0, p["share_light"]))
    return proportional, light_first


def profit_by_price(p, method, attendance_m, response, prices):
    out = []
    for price in prices:
        members_fraction = max(0.0, 1 - response * ((price - p["price_now"]) / p["price_now"]) / 0.10)
        out.append(program_contribution(p, method, attendance_m, price, members_fraction))
    return np.array(out)


# ---------- Reports ----------
def report_member_value(table, attendance):
    rows = []
    for method in METHODS:
        for scenario, att in attendance.items():
            p = params(table)
            v = type_values(p, method, att)
            for t in TYPES:
                rows.append({"method": method, "scenario": scenario, "member_type": t, "value_per_year": v[t]})
            rows.append({"method": method, "scenario": scenario, "member_type": "average member", "value_per_year": average_value(p, v)})
            rows.append({"method": method, "scenario": scenario, "member_type": "whole program ($M)",
                         "value_per_year": program_contribution(p, method, att)})
    return pd.DataFrame(rows).round(1)


def report_payback(table, attendance):
    rows = []
    for method in METHODS:
        for scenario, att in attendance.items():
            p = params(table)
            v = type_values(p, method, att)
            for t in TYPES:
                rows.append({"method": method, "scenario": scenario, "member_type": t,
                             "value_per_year": round(v[t], 1),
                             "payback_months": payback_months(v[t], p[f"churn_{t}"], p["cac"]),
                             "five_year_value_per_member": round(five_year_value(v[t], p[f"churn_{t}"]), 1)})
    return pd.DataFrame(rows)


def report_price_increase(table, attendance):
    rows = []
    for method in METHODS:
        for scenario, att in attendance.items():
            prop, light = break_even_loss(params(table), method, att)
            rows.append({"method": method, "scenario": scenario,
                         "members_AMC_can_lose_pct_if_everyone_equally": prop * 100,
                         "members_AMC_can_lose_pct_if_only_light_members_leave": light * 100})
    return pd.DataFrame(rows).round(1)


def report_best_price(table, attendance, prices):
    rows = []
    for method in METHODS:
        for scenario, att in attendance.items():
            for label, response in (("low", table["low"]["price_response"]), ("base", table["base"]["price_response"]),
                                    ("high", table["high"]["price_response"])):
                profit = profit_by_price(params(table), method, att, response, prices)
                i = int(np.argmax(profit))
                rows.append({"method": method, "scenario": scenario, "price_response": label,
                             "best_price": prices[i], "program_contribution_M": profit[i],
                             "contribution_at_current_price_M": profit_by_price(params(table), method, att, response, [table["base"]["price_now"]])[0]})
    return pd.DataFrame(rows).round(1)


SENSITIVITY_PROGRAM = {"share of visits that are extra": ["extra"], "snack spend per member visit": ["snack_member"],
                       "studios' share of a ticket": ["film_share"], "snack cost": ["snack_cost"],
                       "perks cost per visit": ["perks"], "members' share of attendance": ["member_share"]}
SENSITIVITY_PAYBACK = {"share of visits that are extra": ["extra"], "snack spend per member visit": ["snack_member"],
                       "studios' share of a ticket": ["film_share"], "perks cost per visit": ["perks"],
                       "cost of winning a member": ["cac"],
                       "cancellation": ["churn_light", "churn_typical", "churn_heavy"]}


def sensitivity(table, attendance, method, which):
    att = attendance["base"]
    rows = []
    for label, keys in (SENSITIVITY_PROGRAM if which == "program" else SENSITIVITY_PAYBACK).items():
        res = {}
        for case in ("low", "high"):
            p = params(table, {k: table[case][k] for k in keys})
            if which == "program":
                res[case] = program_contribution(p, method, att)
            else:
                v = type_values(p, method, att)["typical"]
                res[case] = payback_months(v, p["churn_typical"], p["cac"])
        rows.append({"assumption": label, "at_low": res["low"], "at_high": res["high"]})
    return pd.DataFrame(rows)


# ---------- Charts ----------
def chart_member_value(table, attendance):
    p = params(table)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = TYPES + ["average member"]
    x = np.arange(len(labels))
    for offset, method, color in ((-0.2, "A", "#1f4e79"), (0.2, "B", "#e67e22")):
        v = type_values(p, method, attendance["base"])
        vals = [v[t] for t in TYPES] + [average_value(p, v)]
        bars = ax.bar(x + offset, vals, width=0.38, color=color, label=METHODS[method])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + (6 if val >= 0 else -14), f"${val:,.0f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["light (12 visits)", "typical (30)", "heavy (60)", "average member"])
    ax.set_ylabel("Yearly value to AMC, $ per member")
    ax.set_title("What is a member worth per year? (base attendance, base assumptions)", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(CHART_FOLDER, "member_value_by_type.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_sensitivity(table, attendance):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    base_p = params(table)
    for col, method in enumerate(METHODS):
        for row, which in enumerate(("program", "payback")):
            ax = axes[row, col]
            s = sensitivity(table, attendance, method, which)
            if which == "program":
                base = program_contribution(base_p, method, attendance["base"])
                unit, title = "$ millions per year", f"Program contribution, method {method}"
            else:
                v = type_values(base_p, method, attendance["base"])["typical"]
                base = payback_months(v, base_p["churn_typical"], base_p["cac"])
                unit, title = "months to pay back (typical member)", f"Payback, method {method}"
            s = s.fillna(PAYBACK_HORIZON)       # "never within 5 years" is drawn at the 60-month cap
            s["spread"] = (s["at_high"] - s["at_low"]).abs()
            s = s.sort_values("spread")
            y = np.arange(len(s))
            for i, (_, r) in enumerate(s.iterrows()):
                ax.barh(i, r["at_low"] - base, left=base, height=0.55, color="#2e7d32", alpha=0.8)
                ax.barh(i, r["at_high"] - base, left=base, height=0.55, color="#c0392b", alpha=0.8)
            ax.axvline(base, color="black", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(s["assumption"], fontsize=8)
            ax.set_xlabel(unit, fontsize=8)
            ax.set_title(f"{title} (base: {base:,.0f})", loc="left", fontsize=10)
            ax.grid(axis="x", alpha=0.25)
    fig.text(0.01, 0.005, "Green = each assumption at its low value, red = at its high value; all others at base. "
             "Payback bars are capped at 60 months.", fontsize=8, color="#555555")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    path = os.path.join(CHART_FOLDER, "sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_best_price(table, attendance, prices):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    p = params(table)
    for ax, method in zip(axes, METHODS):
        for label, color in (("low", "#2e7d32"), ("base", "#1f4e79"), ("high", "#c0392b")):
            profit = profit_by_price(p, method, attendance["base"], table[label]["price_response"], prices)
            ax.plot(prices, profit, color=color, linewidth=2,
                    label=f"{table[label]['price_response'] * 100:.0f}% of members lost per 10% price rise")
            i = int(np.argmax(profit))
            ax.plot(prices[i], profit[i], "o", color=color)
        ax.axvline(p["price_now"], color="#555555", linestyle="--", linewidth=1)
        ax.set_title(f"Method {method}", loc="left", fontsize=10)
        ax.set_xlabel("Monthly price, $")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Program contribution, $ millions per year")
    axes[0].legend(fontsize=8)
    fig.suptitle("What-if: the best price depends on how members react (dot = best price; dashed line = current $29.99)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(CHART_FOLDER, "best_price_whatif.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(CHART_FOLDER, exist_ok=True)
    table = load_assumptions()
    attendance = load_attendance()
    prices = np.round(np.arange(19.99, 60.0, 1.0), 2)      # wide enough to show where (if anywhere) the curve peaks

    value = report_member_value(table, attendance)
    payback = report_payback(table, attendance)
    increase = report_price_increase(table, attendance)
    best = report_best_price(table, attendance, prices)
    for name, df in (("member_value", value), ("payback", payback), ("price_increase", increase), ("best_price_whatif", best)):
        df.to_csv(os.path.join(RESULTS_FOLDER, f"{name}.csv"), index=False)

    print("=== 1. YEARLY VALUE OF A MEMBER ($), base attendance ===")
    print(value[value["scenario"] == "base"].pivot(index="member_type", columns="method", values="value_per_year").to_string())
    print("\n=== 1b. WHOLE PROGRAM ($ millions per year) by attendance scenario ===")
    prog = value[value["member_type"] == "whole program ($M)"].pivot(index="scenario", columns="method", values="value_per_year")
    print(prog.loc[SCENARIOS].to_string())
    print("\n=== 2. PAYBACK (months; blank = not within 5 years), base attendance ===")
    pb = payback[payback["scenario"] == "base"].pivot(index="member_type", columns="method", values="payback_months")
    print(pb.to_string())
    print("\n=== 3. HOW MANY MEMBERS COULD AMC LOSE AFTER THE JULY 2026 INCREASE? (% of members) ===")
    print(increase.pivot(index="scenario", columns="method").loc[SCENARIOS].to_string())
    print("\n=== 4. WHAT-IF: BEST PRICE, base attendance ===")
    print(best[best["scenario"] == "base"].pivot(index="price_response", columns="method", values="best_price").to_string())
    print("\n=== 6. SENSITIVITY: program contribution ($M) at low / high assumption, base attendance ===")
    for method in METHODS:
        print(f"method {method}:"); print(sensitivity(table, attendance, method, "program").round(1).to_string(index=False))
    print("\nSaved charts:", chart_member_value(table, attendance), chart_sensitivity(table, attendance),
          chart_best_price(table, attendance, prices))


if __name__ == "__main__":
    main()
