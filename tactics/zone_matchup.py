"""Line & length matchups: the two-sided, reliable layer.

Batter zone profiles are built on the RELIABLE marginals (length-by-kind, line-by-kind),
not the sparse 36-cell grid. Primary metric is CONTROL% edge (low-variance: did the batter
middle it), with SR and dismissal alongside. Then a bowler's stock line/length mix is
overlapped with a batter's zone weaknesses to project the matchup.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E
import matchup_engine as M

SHRINK = 45
d = M.load_balls()                      # has bat, bowl, kind, phase, wkt, batruns, ballfaced
d = d[d["control"].notna() & d["length"].notna() & d["line"].notna()].copy()
d["ctrl"] = d["control"]

def zone_profile(dim):
    """Batter x kind x <dim> control/SR edge vs league par, shrunk."""
    g = d.groupby(["bat", "kind", dim]).agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"),
                                            ctrl=("ctrl", "mean"), outs=("wkt", "sum")).reset_index()
    par = d.groupby(["kind", dim]).agg(ctrl_par=("ctrl", "mean"),
            sr_par=("batruns", lambda s: 100*s.sum()/len(s))).reset_index()
    g = g.merge(par, on=["kind", dim], how="left")
    g["sr"] = 100 * g.runs / g.balls
    sh = g.balls / (g.balls + SHRINK)
    g["ctrl_edge"] = ((g.ctrl*100 - g.ctrl_par*100) * sh).round(1)     # + = more in control than par
    g["sr_edge"] = ((g.sr - g.sr_par) * sh).round(1)
    return g[g.balls >= 30]

length_prof = zone_profile("length")
line_prof = zone_profile("line")
length_prof.round(2).to_csv(E.OUT / "zone_batter_length.csv", index=False)
line_prof.round(2).to_csv(E.OUT / "zone_batter_line.csv", index=False)

def show_weak(prof, dim, kind, zone, label, min_balls=60):
    s = prof[(prof.kind == kind) & (prof[dim] == zone) & (prof.balls >= min_balls)]
    print(f"\n== Weakest vs {label} ({kind}), by control-edge, min {min_balls} balls ==")
    print(s.sort_values("ctrl_edge").head(7)[["bat","balls","ctrl","sr","ctrl_edge","sr_edge"]].round(2).to_string(index=False))

show_weak(length_prof, "length", "pace", "SHORT", "the SHORT ball")
show_weak(length_prof, "length", "pace", "YORKER", "the YORKER")
show_weak(length_prof, "length", "spin", "GOOD_LENGTH", "good-length SPIN")
show_weak(line_prof, "line", "pace", "WIDE_OUTSIDE_OFFSTUMP", "the WIDE off-stump channel (pace)")

# ---------- bowler stock zones + matchup projection ----------
bowler_mix = (d.groupby(["bowl", "kind", "length"]).size()
              / d.groupby(["bowl", "kind"]).size()).rename("share").reset_index()
bowler_mix.to_csv(E.OUT / "zone_bowler_length_mix.csv", index=False)

def matchup(batter, bowler):
    """Projected control-edge of batter vs bowler: batter's length-edges weighted by the
    bowler's actual length mix (within the bowler's type). Negative = bowler exploits batter."""
    bk = d[d.bowl == bowler]["kind"].mode()
    if len(bk) == 0: return None
    kind = bk.iloc[0]
    mix = bowler_mix[(bowler_mix.bowl == bowler) & (bowler_mix.kind == kind)]
    bp = length_prof[(length_prof.bat == batter) & (length_prof.kind == kind)]
    m = mix.merge(bp[["length", "ctrl_edge", "balls"]], on="length", how="inner")
    if m.balls.sum() == 0: return None
    return round(float((m.share * m.ctrl_edge).sum() / m.share.sum()), 1), kind

print("\n== Matchup projection examples (control-edge, negative = bowler on top) ==")
for ba, bo in [("Rohit Sharma","Jasprit Bumrah"),("Virat Kohli","Rashid Khan"),
               ("MS Dhoni","Kuldeep Yadav"),("Suryakumar Yadav","Matheesha Pathirana"),
               ("KL Rahul","Jasprit Bumrah")]:
    r = matchup(ba, bo)
    if r: print(f"  {ba:20s} vs {bo:20s} ({r[1]:>4s}) : {r[0]:+.1f}")
print("\nsaved zone_batter_length.csv, zone_batter_line.csv, zone_bowler_length_mix.csv")
