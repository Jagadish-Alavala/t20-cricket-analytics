"""Matchup engine: batter vs bowler-TYPE (pace/spin, style) by phase.

Individual batter-vs-bowler is unusable (median 5 balls/pair), so matchups are measured
against bowler type, where a top batter has hundreds of balls. Signal = the batter's edge
over the league-average batter facing the same type in the same phase, shrunk for sample.
A 'negative matchup' = the batter is reliably worse than par against a type/phase.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import engine as E

SHRINK_K = 45          # balls of prior; shrinks small-sample edges toward 0
MIN_BALLS = 40         # don't report a cell below this

def load_balls(comp="IPL"):
    df = E.load()
    d = df[df.competition == comp].copy()
    ov = pd.to_numeric(d["over"], errors="coerce")
    d["phase"] = np.select([ov <= 6, ov <= 15], ["PP", "MID"], default="DEATH")
    d["kind"] = d["bowl_kind"].map({"pace bowler": "pace", "spin bowler": "spin"})
    d = d[d["kind"].notna()].copy()
    d["wkt"] = (d["out"] == True).astype(int)
    d["legal"] = (d["wide"] == 0).astype(int)
    return d

def cells(d, by):
    """Aggregate balls faced into (batter, *by) cells with SR and dismissal rate."""
    g = d.groupby(["bat", *by]).agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"),
                                    outs=("wkt", "sum")).reset_index()
    g["sr"] = 100 * g.runs / g.balls
    g["dis_rate"] = 100 * g.outs / g.balls               # dismissals per 100 balls
    return g

def league_par(d, by):
    tot = d.groupby(by).agg(balls=("ballfaced", "sum"), runs=("batruns", "sum"), outs=("wkt", "sum"))
    tot["sr_par"] = 100 * tot.runs / tot.balls
    tot["dis_par"] = 100 * tot.outs / tot.balls
    return tot[["sr_par", "dis_par"]].reset_index()

def matchup(d, by):
    c = cells(d, by).merge(league_par(d, by), on=by, how="left")
    c["sr_edge_raw"] = c.sr - c.sr_par
    c["dis_edge_raw"] = c.dis_par - c.dis_rate           # + = dismissed less than par (good)
    shrink = c.balls / (c.balls + SHRINK_K)
    c["sr_edge"] = (c.sr_edge_raw * shrink).round(1)
    c["dis_edge"] = (c.dis_edge_raw * shrink).round(2)
    c["sr"] = c.sr.round(1)
    return c[c.balls >= MIN_BALLS].copy()

if __name__ == "__main__":
    d = load_balls()
    
    # ---- overall batter vs pace/spin (all phases) ----
    mk = matchup(d, ["kind"])
    mk.round(2).to_csv(E.OUT / "matchup_batter_vs_kind.csv", index=False)
    # wide view: spin edge vs pace edge per batter (min balls each), for players with enough of both
    piv = mk.pivot_table(index="bat", columns="kind", values=["sr", "sr_edge", "balls"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna(subset=["balls_pace", "balls_spin"])
    piv = piv[(piv.balls_pace >= 120) & (piv.balls_spin >= 120)]
    
    print("== Most NEGATIVE vs SPIN (weakest spin matchup, min 120 balls vs spin) ==")
    print(piv.sort_values("sr_edge_spin").head(8)[["balls_spin", "sr_spin", "sr_edge_spin", "sr_edge_pace"]].round(1).to_string())
    print("\n== Most NEGATIVE vs PACE (weakest pace matchup) ==")
    print(piv.sort_values("sr_edge_pace").head(8)[["balls_pace", "sr_pace", "sr_edge_pace", "sr_edge_spin"]].round(1).to_string())
    print("\n== Biggest SPIN-vs-PACE swing (much better against one type) ==")
    piv["swing"] = piv.sr_edge_spin - piv.sr_edge_pace
    print("  strong vs spin / weak vs pace:")
    print(piv.sort_values("swing", ascending=False).head(5)[["sr_edge_spin","sr_edge_pace"]].round(1).to_string())
    print("  strong vs pace / weak vs spin:")
    print(piv.sort_values("swing").head(5)[["sr_edge_spin","sr_edge_pace"]].round(1).to_string())
    
    # ---- by phase (where in the innings the matchup bites) ----
    mkp = matchup(d, ["kind", "phase"])
    mkp.round(2).to_csv(E.OUT / "matchup_batter_vs_kind_phase.csv", index=False)
    print("\n== Negative spin matchups IN THE MIDDLE OVERS (min 60 balls) ==")
    mid = mkp[(mkp.kind == "spin") & (mkp.phase == "MID") & (mkp.balls >= 60)]
    print(mid.sort_values("sr_edge").head(8)[["bat", "balls", "sr", "sr_edge", "dis_edge"]].round(1).to_string(index=False))
    
    # ---- vs specific style (secondary, needs more care on sample) ----
    ms = matchup(d, ["bowl_style"])
    ms = ms[ms.balls >= 80]
    ms.round(2).to_csv(E.OUT / "matchup_batter_vs_style.csv", index=False)
    print("\nsaved: matchup_batter_vs_kind.csv, _kind_phase.csv, _style.csv")
