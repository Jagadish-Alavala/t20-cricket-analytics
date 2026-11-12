"""DL closed-form impact (imp), runs-above-average (raa), wickets-above-average (waa)
matching the reference table's columns, computed per batter-season on our data.

imp per ball (from the project diagram):  s_i + D(u-1, w+w_i) - D(u, w)
  s_i = runs off bat, u = balls remaining BEFORE the ball, w = wickets down BEFORE,
  w_i = wicket on the ball (0/1), D = dlxr resource curve.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E

def batter_impact_table(comp="IPL", seasons=(2020, 2021, 2022, 2023, 2024)) -> pd.DataFrame:
    df = E.load(seasons)
    d = df[df["competition"] == comp].copy()
    d["u_after"] = d["inns_balls_rem"].clip(0, 120)
    d["w_after"] = d["inns_wkts"].clip(0, 10)
    d["wkt"] = (d["out"] == True).astype(int)             # dismissal on this ball
    d["u_before"] = (d["u_after"] + 1).clip(0, 120)
    d["w_before"] = (d["w_after"] - d["wkt"]).clip(0, 10)
    # DL closed-form impact
    d["imp"] = (d["batruns"]
                + E.dlxr(d["u_after"].values, d["w_after"].values)
                - E.dlxr(d["u_before"].values, d["w_before"].values))
    # runs / wickets above average, benchmarked to league-average per-ball by over
    d["legal"] = (d["wide"] == 0).astype(int)
    exp_runs = d.groupby("over")["batruns"].transform("mean")
    exp_out = d.groupby("over")["wkt"].transform("mean")
    d["raa_ball"] = d["batruns"] - exp_runs
    d["waa_ball"] = exp_out - d["wkt"]                     # + = dismissed less than average

    g = d.groupby(["bat", "year"])
    t = g.agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"),
              imp=("imp", "sum"), raa=("raa_ball", "sum"), waa=("waa_ball", "sum"),
              inns=("p_match", "nunique")).reset_index()
    t["raapb"] = (100 * t["raa"] / t["balls"]).round(2)
    t["imppb"] = (100 * t["imp"] / t["balls"]).round(2)
    t["waapb"] = (100 * t["waa"] / t["balls"]).round(2)
    t["sr"] = (100 * t["runs"] / t["balls"]).round(2)
    t["imppm"] = (t["imp"] / t["inns"]).round(2)
    for c in ["imp", "raa", "waa"]:
        t[c] = t[c].round(2)
    return t[["bat", "year", "balls", "runs", "imp", "raa", "waa", "inns",
              "raapb", "imppb", "waapb", "sr", "imppm"]]

if __name__ == "__main__":
    t = batter_impact_table()
    t = t[t["balls"] >= 150].sort_values("imp", ascending=False).reset_index(drop=True)
    t.index += 1
    t.to_csv(E.OUT / "ipl_impact_dl_table.csv")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(t.head(25).to_string())
    print("\nrows:", len(t), "| seasons covered:", sorted(t['year'].unique()))
