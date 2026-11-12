"""Production RAA: runs above a STATE-CONDITIONED expected-runs baseline (not over-average).

Expected runs off a ball = LGBM E[batruns | full match state]. RAA = actual - expected,
no wicket-swing term (that term is what made DL impact noisy). Then judge it on both axes:
  validity   : AUC predicting the batting team's win (beat crude RAA 0.660?)
  reliability: split-half correlation of per-ball RAA (beat DL impact 0.345? approach SR 0.463?)
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
import engine as E
RNG = np.random.default_rng(42)

df = E.load()
fi = E.first_innings_states(df)                       # all leagues, 1st innings, per-ball state
fi = E.add_venue_baselines(fi, fi, "runs")
fi["league"] = E.encode_league(fi["competition"])
# EXOGENOUS only: the batter cannot cause the over, wickets-in-hand, venue or league.
# Deliberately excludes score / crr / dl_rem, which absorb the batter's own contribution.
B_FEATS = ["over", "balls_rem", "wkts_lost", "venue_avg", "league", "pp", "death"]

# expected runs off THIS ball given the situation (population-average scoring function; target has no outcome leak)
xb = LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=63, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1).fit(fi[B_FEATS], fi["batruns"])
fi["exp_ball"] = xb.predict(fi[B_FEATS])
fi["raa_ball"] = fi["batruns"] - fi["exp_ball"]

ipl = fi[fi.competition == "IPL"].copy()
# crude over-average baseline for comparison
ipl["exp_over"] = ipl.groupby("over")["batruns"].transform("mean")
ipl["raa_over_ball"] = ipl["batruns"] - ipl["exp_over"]

g = ipl.groupby(["bat", "p_match"])
t = g.agg(runs=("batruns", "sum"), balls=("ballfaced", "sum"),
          raa_model=("raa_ball", "sum"), raa_over=("raa_over_ball", "sum"),
          won=("won_bat", "max")).reset_index()
t = t[t.balls >= 10].copy()
t["sr"] = 100 * t.runs / t.balls

print(f"batter-innings (IPL 1st inns, >=10 balls): {len(t):,}")
print("\n== Validity: AUC predicting the batting team's WIN ==")
for c, lab in [("runs", "raw runs"), ("sr", "strike rate"),
               ("raa_over", "RAA (over-average, old)"), ("raa_model", "RAA (state model, NEW)")]:
    print(f"    {lab:28s} AUC {roc_auc_score(t.won, t[c]):.3f}")

def reliability(frame, col, min_inns=12, iters=40):
    inn = frame.groupby(["bat", "p_match"]).agg(v=(col, "sum"), b=("ballfaced", "sum")).reset_index()
    cnt = inn.groupby("bat")["p_match"].nunique(); keep = cnt[cnt >= min_inns].index
    cors = []
    for _ in range(iters):
        inn["h"] = RNG.integers(0, 2, len(inn))
        a = inn.groupby(["bat", "h"]).agg(v=("v", "sum"), b=("b", "sum")).reset_index()
        a["m"] = 100 * a.v / a.b
        p = a.pivot(index="bat", columns="h", values="m").dropna()
        p = p.loc[p.index.intersection(keep)]
        if len(p) > 10: cors.append(np.corrcoef(p[0], p[1])[0, 1])
    return float(np.mean(cors))

ipl["runs_c"] = ipl["batruns"]
print("\n== Reliability: split-half correlation of the per-ball metric ==")
print(f"    strike rate               {reliability(ipl, 'runs_c'):.3f}")
print(f"    RAA (over-average, old)   {reliability(ipl, 'raa_over_ball'):.3f}")
print(f"    RAA (state model, NEW)    {reliability(ipl, 'raa_ball'):.3f}")

# season leaderboard with the new RAA (per 100 balls), min balls
lb = ipl.groupby("bat").agg(inns=("p_match","nunique"), balls=("ballfaced","sum"),
       runs=("batruns","sum"), raa=("raa_ball","sum")).reset_index()
lb = lb[lb.balls>=200].copy(); lb["sr"]=100*lb.runs/lb.balls; lb["raa_per100"]=100*lb.raa/lb.balls
lb.sort_values("raa", ascending=False).round(2).to_csv(E.OUT/"raa_model_leaderboard.csv", index=False)
print("\nTop 12 by RAA (state-model), IPL 2020-24, min 200 balls:")
print(lb.sort_values("raa", ascending=False).head(12)[["bat","inns","balls","runs","sr","raa","raa_per100"]].round(1).to_string(index=False))
