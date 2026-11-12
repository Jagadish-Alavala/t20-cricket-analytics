"""Out-of-sample validation of the line/length matchup signal.

Question: does a batter's zone CONTROL profile learned on 2020-23 predict their ACTUAL
control on 2024 balls better than knowing only the league-average for that zone?
If yes, the matchup signal is real and predictive, not just descriptive.

  baseline  : P(control) = league zone par (kind x length x line), train-fit
  +batter   : baseline + batter's shrunk zone control-edge (kind x length, kind x line), train-fit
Compare AUC and log-loss on 2024. Also batter-level persistence: train edge vs 2024 edge.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import engine as E, matchup_engine as M
SHRINK = 60

d = M.load_balls()
d = d[d.control.notna() & d.length.notna() & d.line.notna()].copy()
d["ctrl"] = d.control.astype(int)
tr, te = d[d.year <= 2023].copy(), d[d.year == 2024].copy()

# league zone par (train)
par_len = tr.groupby(["kind", "length"])["ctrl"].mean().rename("par_len")
par_line = tr.groupby(["kind", "line"])["ctrl"].mean().rename("par_line")
par_kl = tr.groupby(["kind", "length", "line"])["ctrl"].mean().rename("par_kl")
gmean = tr.ctrl.mean()

# batter shrunk edges (train)
def edge(dim, par):
    g = tr.groupby(["bat", "kind", dim])["ctrl"].agg(["mean", "size"]).reset_index()
    g = g.merge(par, on=["kind", dim], how="left")
    g["edge"] = (g["mean"] - g[par.name]) * (g["size"] / (g["size"] + SHRINK))
    return g.set_index(["bat", "kind", dim])["edge"]
len_edge = edge("length", par_len)
line_edge = edge("line", par_line)

# assemble predictions on 2024
te = te.merge(par_kl.reset_index(), on=["kind", "length", "line"], how="left")
te["par_kl"] = te["par_kl"].fillna(gmean)
te["le"] = te.set_index(["bat", "kind", "length"]).index.map(len_edge).astype(float)
te["li"] = te.set_index(["bat", "kind", "line"]).index.map(line_edge).astype(float)
te["le"] = te["le"].fillna(0); te["li"] = te["li"].fillna(0)
te["p_base"] = te["par_kl"].clip(1e-3, 1 - 1e-3)
te["p_bat"] = (te["par_kl"] + te["le"] + te["li"]).clip(1e-3, 1 - 1e-3)

y = te["ctrl"].values
print(f"2024 test balls: {len(te):,}   actual control rate: {y.mean():.3f}")
print("\n== Predicting ACTUAL 2024 control ==")
print(f"  baseline (league zones only) : AUC {roc_auc_score(y, te.p_base):.4f}   logloss {log_loss(y, te.p_base):.4f}")
print(f"  + batter zone profile        : AUC {roc_auc_score(y, te.p_bat):.4f}   logloss {log_loss(y, te.p_bat):.4f}")
gain = roc_auc_score(y, te.p_bat) - roc_auc_score(y, te.p_base)
print(f"  batter zone signal adds: {gain:+.4f} AUC  ->  {'PREDICTIVE out-of-sample' if gain>0.003 else 'no real out-of-sample lift'}")

# batter-level persistence: does 2020-23 length control-edge predict 2024 edge?
def batter_edge(frame):
    g = frame.groupby(["bat", "kind", "length"])["ctrl"].agg(["mean", "size"]).reset_index().merge(par_len.reset_index(), on=["kind","length"])
    g["e"] = (g["mean"] - g["par_len"])
    w = g[g["size"] >= 20]
    return w.groupby("bat").apply(lambda x: np.average(x.e, weights=x["size"]))
e_tr, e_te = batter_edge(tr), batter_edge(te)
j = pd.concat([e_tr.rename("train"), e_te.rename("test2024")], axis=1).dropna()
n_tr = tr.groupby("bat").size(); j = j[j.index.map(n_tr).fillna(0) >= 200]
print(f"\n== Batter-level persistence (control-edge vs pace/spin lengths) ==")
print(f"  corr(2020-23 edge, 2024 edge) across {len(j)} batters: {np.corrcoef(j.train, j.test2024)[0,1]:.3f}")
