"""Batter phase-value profiles: where in the innings each batter adds value.
Feeds the entry-point optimizer. For each batter x phase: SR vs par and dismissal risk,
shrunk for sample. 'Best phase' = the phase where the batter beats par most."""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E
import matchup_engine as M   # reuse loader + matchup()

d = M.load_balls()
prof = M.matchup(d, ["phase"])            # batter x phase, SR edge vs league par for that phase
prof = prof[prof.balls >= 40].copy()

# wide profile: SR-edge by phase per batter
w = prof.pivot_table(index="bat", columns="phase", values="sr_edge")
b = prof.pivot_table(index="bat", columns="phase", values="balls")
w = w.reindex(columns=["PP", "MID", "DEATH"]); b = b.reindex(columns=["PP", "MID", "DEATH"])
w.columns = [f"edge_{c}" for c in w.columns]; b.columns = [f"balls_{c}" for c in b.columns]
prof_w = w.join(b)
# best phase among those with enough balls
def best_phase(row):
    cand = {p: row[f"edge_{p}"] for p in ["PP", "MID", "DEATH"]
            if pd.notna(row.get(f"balls_{p}")) and row.get(f"balls_{p}", 0) >= 60 and pd.notna(row[f"edge_{p}"])}
    return max(cand, key=cand.get) if cand else np.nan
prof_w["best_phase"] = prof_w.apply(best_phase, axis=1)
prof_w = prof_w.dropna(subset=["best_phase"])
prof_w.round(1).to_csv(E.OUT / "batter_phase_profile.csv")

print("== Death-overs specialists (top SR-edge in DEATH, min 60 death balls) ==")
dth = prof_w[(prof_w.balls_DEATH >= 60)].sort_values("edge_DEATH", ascending=False)
print(dth.head(8)[["balls_DEATH", "edge_DEATH", "edge_MID", "edge_PP"]].round(1).to_string())

print("\n== Powerplay specialists (top SR-edge in PP, min 120 PP balls) ==")
pp = prof_w[(prof_w.balls_PP >= 120)].sort_values("edge_PP", ascending=False)
print(pp.head(8)[["balls_PP", "edge_PP", "edge_MID", "edge_DEATH"]].round(1).to_string())

print("\n== Middle-overs anchors who FADE at the death (deploy earlier) ==")
fade = prof_w[(prof_w.balls_MID >= 150) & (prof_w.balls_DEATH >= 60)].copy()
fade["drop"] = fade.edge_MID - fade.edge_DEATH
print(fade.sort_values("drop", ascending=False).head(6)[["edge_PP","edge_MID","edge_DEATH"]].round(1).to_string())

print("\nBest-phase distribution:", prof_w.best_phase.value_counts().to_dict())
print("saved batter_phase_profile.csv  (", len(prof_w), "batters )")
