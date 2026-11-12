"""Validity test: which first-innings batting metric best tracks WINNING?

At the batter-innings level (first innings, min 10 balls), each metric predicts whether
the batting team won the match. Higher AUC = more win-relevant = more valid.
A context metric (DL impact, RAA, SR-vs-par) earns its complexity only if it beats the
simple metric in the same class (runs for totals, SR for rates).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import engine as E

df = E.load()
d = df[(df.competition == "IPL") & (df.inns == 1)].copy()
d["u_after"] = d.inns_balls_rem.clip(0, 120); d["w_after"] = d.inns_wkts.clip(0, 10)
d["wkt"] = (d["out"] == True).astype(int)
d["u_before"] = (d.u_after + 1).clip(0, 120); d["w_before"] = (d.w_after - d.wkt).clip(0, 10)
d["imp_ball"] = d.batruns + E.dlxr(d.u_after.values, d.w_after.values) - E.dlxr(d.u_before.values, d.w_before.values)
d["exp_runs_ball"] = d.groupby("over")["batruns"].transform("mean")   # phase-average expectation

g = d.groupby(["bat", "p_match"])
t = g.agg(runs=("batruns", "sum"), balls=("ballfaced", "sum"),
          imp=("imp_ball", "sum"), exp=("exp_runs_ball", "sum"),
          won=("won_bat", "max")).reset_index()
t = t[t.balls >= 10].copy()
t["sr"] = 100 * t.runs / t.balls
t["imppb"] = 100 * t.imp / t.balls
t["raa"] = t.runs - t.exp                       # runs above phase-average
t["sr_vs_par"] = 100 * (t.runs - t.exp) / t.balls
print(f"batter-innings (1st inns, >=10 balls): {len(t):,}   base win rate {t.won.mean():.3f}")

def auc(col): return roc_auc_score(t.won, t[col])
print("\n== Single-metric AUC for predicting the batting team's WIN ==")
print("  TOTAL metrics (reward volume):")
for c in ["runs", "imp", "raa"]:
    print(f"    {c:10s} AUC {auc(c):.3f}")
print("  RATE metrics (per-ball, volume removed):")
for c in ["sr", "imppb", "sr_vs_par"]:
    print(f"    {c:10s} AUC {auc(c):.3f}")

# incremental validity: does a context metric add over raw runs?
def incr(base, extra):
    X = StandardScaler().fit_transform(t[base + extra]); y = t.won.values
    m = LogisticRegression(max_iter=1000).fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])
print("\n== Incremental AUC over raw runs ==")
print(f"    runs alone            {incr(['runs'], []):.3f}")
print(f"    runs + imp            {incr(['runs'], ['imp']):.3f}")
print(f"    runs + raa            {incr(['runs'], ['raa']):.3f}")
print(f"    runs + sr             {incr(['runs'], ['sr']):.3f}")
print(f"    runs + sr_vs_par      {incr(['runs'], ['sr_vs_par']):.3f}")
print(f"    runs + sr + imp       {incr(['runs'], ['sr','imp']):.3f}")
