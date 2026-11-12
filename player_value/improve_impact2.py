"""Second round: since venue-rescaling hurt reliability, test two evidence-led questions.
  Q1  Is DL impact more reliable than plain strike rate? (is it worth having at all)
  Q2  Does an EMPIRICAL resource curve refit on 2020-24 (era correction, no per-venue
      noise) beat the fixed 2015-fit global curve?
Reliability = split-half correlation of the per-ball metric across a batter's own innings.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E
RNG = np.random.default_rng(42)

df = E.load()
d = df[df["competition"] == "IPL"].copy()
d["u_after"] = d["inns_balls_rem"].clip(0, 120)
d["w_after"] = d["inns_wkts"].clip(0, 10)
d["wkt"] = (d["out"] == True).astype(int)
d["u_before"] = (d["u_after"] + 1).clip(0, 120)
d["w_before"] = (d["w_after"] - d["wkt"]).clip(0, 10)

# ---- global (2015-fit) impact
d["imp_global"] = d["batruns"] + E.dlxr(d.u_after.values, d.w_after.values) - E.dlxr(d.u_before.values, d.w_before.values)

# ---- empirical resource curve from 1st-innings, current data
fi = d[d["inns"] == 1].copy()
final = fi.groupby("p_match")["inns_runs"].transform("max")
fi["rem"] = final - fi["inns_runs"]
fi["ub"] = (fi["u_after"] // 3).clip(0, 40)            # 3-ball buckets
surf = fi.groupby(["ub", "w_after"])["rem"].mean()
gmean_rem = fi["rem"].mean()
def Demp(u, w):
    ub = np.clip(u // 3, 0, 40)
    idx = list(zip(ub, np.clip(w, 0, 10)))
    return pd.Series(idx).map(surf).fillna(gmean_rem).values
d["imp_emp"] = d["batruns"] + Demp(d.u_after.values, d.w_after.values) - Demp(d.u_before.values, d.w_before.values)

def reliability(frame, col, is_ball_level=True, min_inns=12, iters=40):
    inn = frame.groupby(["bat", "p_match"]).agg(v=(col, "sum"), balls=("ballfaced", "sum")).reset_index()
    cnt = inn.groupby("bat")["p_match"].nunique(); keep = cnt[cnt >= min_inns].index
    cors = []
    for _ in range(iters):
        inn["h"] = RNG.integers(0, 2, len(inn))
        a = inn.groupby(["bat", "h"]).agg(v=("v", "sum"), b=("balls", "sum")).reset_index()
        a["m"] = 100 * a["v"] / a["b"]
        p = a.pivot(index="bat", columns="h", values="m").dropna()
        p = p.loc[p.index.intersection(keep)]
        if len(p) > 10: cors.append(np.corrcoef(p[0], p[1])[0, 1])
    return float(np.mean(cors))

# SR reliability: "v" = runs, per-ball metric = runs/balls*100 = SR
d["runs"] = d["batruns"]
print("Split-half reliability (IPL 2020-24, >=12 inns):")
print(f"  strike rate        : {reliability(d, 'runs'):.3f}")
print(f"  DL impact (global) : {reliability(d, 'imp_global'):.3f}")
print(f"  DL impact (empir.) : {reliability(d, 'imp_emp'):.3f}")

# how different are global vs empirical rankings?
pb = d.groupby("bat").agg(balls=("ballfaced","sum"), inns=("p_match","nunique"),
      g=("imp_global","sum"), e=("imp_emp","sum")).reset_index()
pb = pb[pb.inns>=12]
from scipy.stats import spearmanr
print(f"\n  Spearman(global imp, empirical imp) across batters: {spearmanr(pb.g, pb.e).correlation:.3f}")
