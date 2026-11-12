"""Final assembled leaderboards (IPL 2024), the settled metrics only, kept as separate
currencies (no fused score, by design):
    First-innings batting  -> RAA (runs above the over-by-over league average)
    Chase batting          -> WPA (Duckworth-Lewis power-law win probability)
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E

df = E.load()
OUT = E.OUT

# ---------------- First-innings RAA (2024) ----------------
fi = df[(df.competition == "IPL") & (df.inns == 1) & (df.year == 2024)].copy()
fi["exp_over"] = fi.groupby("over")["batruns"].transform("mean")
fi["raa_ball"] = fi["batruns"] - fi["exp_over"]
a = fi.groupby("bat").agg(inns=("p_match", "nunique"), balls=("ballfaced", "sum"),
                          runs=("batruns", "sum"), RAA=("raa_ball", "sum")).reset_index()
a = a[a.balls >= 60].copy()
a["SR"] = (100 * a.runs / a.balls).round(1)
a["RAA"] = a.RAA.round(1)
a = a.sort_values("RAA", ascending=False)
a[["bat", "inns", "balls", "runs", "SR", "RAA"]].to_csv(OUT / "final_first_innings_raa_2024.csv", index=False)

# ---------------- Chase WPA (2024) via DL power-law ----------------
si = df[(df.competition == "IPL") & (df.inns == 2) & (df.year == 2024) & (df.max_balls == 120)].copy()
si["balls_rem"] = si.inns_balls_rem.clip(0, 120)
si["wkts_lost"] = si.inns_wkts.clip(0, 9)
si["needed"] = (si.target - si.inns_runs).clip(lower=0)
si["dl"] = E.dlxr(si.balls_rem.values, si.wkts_lost.values)
r = (si.dl / si.needed.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(5).clip(0.01, 5)
si["wp"] = 1 / (1 + (r / 0.96) ** (-6.5))
si = si.sort_values(["p_match", "over", "ball"])
si["prev"] = si.groupby("p_match")["wp"].shift(1).fillna(si["wp"])
si["wpa"] = si["wp"] - si["prev"]
b = si.groupby("bat").agg(inns=("p_match", "nunique"), balls=("ballfaced", "sum"),
                          runs=("batruns", "sum"), WPA=("wpa", "sum")).reset_index()
b = b[b.balls >= 40].copy()
b["SR"] = (100 * b.runs / b.balls).round(1)
b["WPA"] = b.WPA.round(2)
b = b.sort_values("WPA", ascending=False)
b[["bat", "inns", "balls", "runs", "SR", "WPA"]].to_csv(OUT / "final_chase_wpa_2024.csv", index=False)

# bowler chase WPA prevented
bw = si.groupby("bowl").agg(inns=("p_match", "nunique"), balls=("ballfaced", "sum"),
                            wpa=("wpa", "sum")).reset_index()
bw["WPA_prevented"] = (-bw.wpa).round(2)
bw = bw[bw.balls >= 40].sort_values("WPA_prevented", ascending=False)
bw[["bowl", "inns", "balls", "WPA_prevented"]].to_csv(OUT / "final_chase_wpa_bowlers_2024.csv", index=False)

print("== IPL 2024 First-innings batting, top 10 by RAA (runs above over-average) ==")
print(a[["bat", "inns", "balls", "runs", "SR", "RAA"]].head(10).to_string(index=False))
print("\n== IPL 2024 Chase batting, top 10 by WPA (wins added) ==")
print(b[["bat", "inns", "balls", "runs", "SR", "WPA"]].head(10).to_string(index=False))
print("\n== IPL 2024 Chase bowling, top 8 by WPA prevented ==")
print(bw[["bowl", "inns", "balls", "WPA_prevented"]].head(8).to_string(index=False))
