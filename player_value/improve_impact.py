"""Improve first-innings DL Impact by making the resource curve context-aware
(venue + season par) instead of one global 2015-fit curve.

Test of 'better' = split-half reliability: a batter's per-ball impact in a random
half of their innings should predict the other half. A metric that measures skill
(not the pitch they got) is more repeatable. We compare:
    imppb_global   : impact from the fixed global D(u,w)
    imppb_context  : impact from D rescaled to venue+season par
Higher split-half correlation = better metric.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E

RNG = np.random.default_rng(42)

def build(comp="IPL", seasons=(2020, 2021, 2022, 2023, 2024)) -> pd.DataFrame:
    df = E.load(seasons)
    d = df[df["competition"] == comp].copy()
    d["u_after"] = d["inns_balls_rem"].clip(0, 120)
    d["w_after"] = d["inns_wkts"].clip(0, 10)
    d["wkt"] = (d["out"] == True).astype(int)
    d["u_before"] = (d["u_after"] + 1).clip(0, 120)
    d["w_before"] = (d["w_after"] - d["wkt"]).clip(0, 10)
    d["D_after"] = E.dlxr(d["u_after"].values, d["w_after"].values)
    d["D_before"] = E.dlxr(d["u_before"].values, d["w_before"].values)

    # ---- global impact (current method)
    d["imp_global"] = d["batruns"] + d["D_after"] - d["D_before"]

    # ---- context par: what an average side makes here, this season
    fi = d[d["inns"] == 1]
    final = fi.groupby(["p_match", "ground", "year"])["inns_runs"].max().reset_index()
    global_par = final["inns_runs"].mean()
    ground_par = final.groupby("ground")["inns_runs"].mean()
    season_fac = final.groupby("year")["inns_runs"].mean() / global_par
    base = float(E.dlxr(np.array([120.0]), np.array([0.0]))[0])   # global start resource
    # target start-par per (ground, year); shrink small-sample grounds toward global
    n_ground = final.groupby("ground").size()
    gp = ((ground_par * n_ground + global_par * 8) / (n_ground + 8))
    d["gpar"] = d["ground"].map(gp).fillna(global_par)
    d["sfac"] = d["year"].map(season_fac).fillna(1.0)
    d["k"] = (d["gpar"] * d["sfac"]) / base                       # per-ball rescale factor
    d["imp_context"] = d["batruns"] + d["k"] * (d["D_after"] - d["D_before"])
    return d

def per_batter(d: pd.DataFrame, min_inns=12) -> pd.DataFrame:
    g = d.groupby("bat")
    t = g.agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"),
              imp_global=("imp_global", "sum"), imp_context=("imp_context", "sum"),
              inns=("p_match", "nunique")).reset_index()
    t["sr"] = 100 * t["runs"] / t["balls"]
    t["imppb_global"] = 100 * t["imp_global"] / t["balls"]
    t["imppb_context"] = 100 * t["imp_context"] / t["balls"]
    return t[t["inns"] >= min_inns].reset_index(drop=True)

def split_half_reliability(d: pd.DataFrame, col: str, min_inns=12, iters=40) -> float:
    """Randomly split each batter's innings in two, correlate per-ball impact across halves."""
    inn = d.groupby(["bat", "p_match"]).agg(imp=(col, "sum"), balls=("ballfaced", "sum")).reset_index()
    cors = []
    for _ in range(iters):
        inn["half"] = RNG.integers(0, 2, len(inn))
        agg = inn.groupby(["bat", "half"]).agg(imp=("imp", "sum"), balls=("balls", "sum")).reset_index()
        agg["imppb"] = 100 * agg["imp"] / agg["balls"]
        piv = agg.pivot(index="bat", columns="half", values="imppb").dropna()
        cnt = inn.groupby("bat")["p_match"].nunique()
        keep = cnt[cnt >= min_inns].index
        piv = piv.loc[piv.index.intersection(keep)]
        if len(piv) > 10:
            cors.append(np.corrcoef(piv[0], piv[1])[0, 1])
    return float(np.mean(cors))

if __name__ == "__main__":
    d = build()
    rel_g = split_half_reliability(d, "imp_global")
    rel_c = split_half_reliability(d, "imp_context")
    print(f"Split-half reliability (per-ball impact):")
    print(f"  global  curve : {rel_g:.3f}")
    print(f"  context curve : {rel_c:.3f}   ({'+' if rel_c>rel_g else ''}{(rel_c-rel_g):.3f})")

    t = per_batter(d)
    t["delta"] = t["imppb_context"] - t["imppb_global"]
    # cross-check: context impact should carry venue signal beyond raw strike rate
    print(f"\n  corr(imppb_global, SR)  = {np.corrcoef(t.imppb_global, t.sr)[0,1]:.3f}")
    print(f"  corr(imppb_context, SR) = {np.corrcoef(t.imppb_context, t.sr)[0,1]:.3f}")
    up = t.sort_values("delta", ascending=False).head(6)
    dn = t.sort_values("delta").head(6)
    print("\nMost UP-graded by venue/season context (were under-credited):")
    print(up[["bat", "inns", "sr", "imppb_global", "imppb_context", "delta"]].round(2).to_string(index=False))
    print("\nMost DOWN-graded (were over-credited by the global curve):")
    print(dn[["bat", "inns", "sr", "imppb_global", "imppb_context", "delta"]].round(2).to_string(index=False))
    t.round(2).to_csv(E.OUT / "impact_context_vs_global.csv", index=False)
    print("\nsaved impact_context_vs_global.csv")
