"""Actual entry points from data.
  Batters: the over and wickets-down at which they typically come to the crease (their real slot).
  Bowlers: which overs/phases they are brought on (their deployment window).
This grounds the optimizer in how teams actually use players, instead of manual inputs.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E, matchup_engine as M

d = M.load_balls()  # IPL, has over, phase, bat, bowl, inns_wkts, p_match, inns, ballfaced

# ---------------- Batter entry point ----------------
faced = d[d.ballfaced == 1].copy()
faced["ov"] = pd.to_numeric(faced["over"], errors="coerce")
first = faced.sort_values(["p_match", "inns", "ov", "ball"]).groupby(["bat", "p_match", "inns"]).first().reset_index()
bat = first.groupby("bat").agg(inns=("p_match", "nunique"),
                               entry_over=("ov", "median"),
                               entry_wkts=("inns_wkts", "median")).reset_index()
bat = bat[bat.inns >= 10].copy()
bat["slot"] = pd.cut(bat.entry_over, [0, 1.5, 4, 9, 16, 20],
                     labels=["Opener", "No.3", "Top-mid", "Middle", "Finisher"], include_lowest=True)
bat = bat.sort_values("entry_over")
bat.round(1).to_csv(E.OUT / "entry_batters.csv", index=False)

# ---------------- Bowler deployment window ----------------
d["ov"] = pd.to_numeric(d["over"], errors="coerce")
tot = d.groupby("bowl").size()
byph = d.groupby(["bowl", "phase"]).size().unstack(fill_value=0)
byph = byph.div(byph.sum(axis=1), axis=0) * 100
firstov = d.groupby(["bowl", "p_match"])["ov"].min().groupby("bowl").median().rename("first_over")
opm = (d.groupby(["bowl", "p_match"]).size() / 6).groupby("bowl").mean().rename("overs_per_match")
bowl = pd.concat([tot.rename("balls"), byph, firstov, opm], axis=1).reset_index()
bowl = bowl[bowl.balls >= 120].copy()
for c in ["PP", "MID", "DEATH"]:
    if c not in bowl: bowl[c] = 0
bowl["role"] = np.select(
    [bowl.PP >= 45, bowl.DEATH >= 35, bowl.MID >= 55],
    ["New-ball", "Death", "Middle-overs"], default="Mixed")
bowl = bowl.round(1).sort_values("first_over")
bowl.to_csv(E.OUT / "entry_bowlers.csv", index=False)

print("== Batter entry points (typical over / wickets-down when they arrive) ==")
print(bat.head(6)[["bat","inns","entry_over","entry_wkts","slot"]].to_string(index=False))
print("  ...finishers...")
print(bat.tail(6)[["bat","inns","entry_over","entry_wkts","slot"]].to_string(index=False))

print("\n== Bowler deployment (phase % of balls, typical first over, role) ==")
show = ["bowl","first_over","PP","MID","DEATH","overs_per_match","role"]
print("  New-ball bowlers:")
print(bowl[bowl.role=="New-ball"].sort_values("PP",ascending=False).head(6)[show].to_string(index=False))
print("  Death bowlers:")
print(bowl[bowl.role=="Death"].sort_values("DEATH",ascending=False).head(6)[show].to_string(index=False))
print("\nsaved entry_batters.csv, entry_bowlers.csv")
