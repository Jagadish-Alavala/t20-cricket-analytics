"""Bowler-aware entry-point optimizer (v2).

Upgrades v1 by connecting the matchup engine to the optimizer: instead of a pace/spin
share, you pass the ACTUAL bowlers with overs left. For each candidate batter it scores
the specific matchup against each bowler (pace/spin SR-edge + line/length control-edge),
weighted by that bowler's remaining overs, plus phase fit and a safety term.

Loads only the saved CSVs (fast, no heavy recompute). Advisory, transparent, explains itself.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E
O = E.OUT

prof = pd.read_csv(O/"batter_phase_profile.csv", index_col=0)          # edge_PP/MID/DEATH
mk = pd.read_csv(O/"matchup_batter_vs_kind.csv")                        # bat,kind,sr_edge,dis_edge
zbl = pd.read_csv(O/"zone_batter_length.csv")                          # bat,kind,length,ctrl_edge
bmix = pd.read_csv(O/"zone_bowler_length_mix.csv")                     # bowl,kind,length,share
mk_sr = mk.pivot_table(index="bat", columns="kind", values="sr_edge")
mk_dis = mk.pivot_table(index="bat", columns="kind", values="dis_edge")
blen = {(r.bat, r.kind): {} for r in zbl.itertuples()}
for r in zbl.itertuples(): blen[(r.bat, r.kind)][r.length] = r.ctrl_edge
bowler_kind = bmix.groupby("bowl")["kind"].agg(lambda s: s.mode().iloc[0]).to_dict()

def _get(df, a, b):
    try:
        v = df.loc[a, b]; return float(v) if pd.notna(v) else 0.0
    except Exception:
        return 0.0

def matchup_vs_bowler(bat, bowler):
    """Projected edge of batter vs a specific bowler: pace/spin SR-edge + weighted
    line/length control-edge over that bowler's stock lengths."""
    kind = bowler_kind.get(bowler)
    if kind is None: return 0.0, 0.0, "unknown"
    sr = _get(mk_sr, bat, kind)
    mix = bmix[(bmix.bowl == bowler) & (bmix.kind == kind)]
    ledge = blen.get((bat, kind), {})
    num = den = 0.0
    for r in mix.itertuples():
        if r.length in ledge:
            num += r.share * ledge[r.length]; den += r.share
    ctrl = (num/den) if den > 0 else 0.0
    return sr, ctrl, kind

def phase_weights(over_now):
    overs = list(range(over_now+1, 21)) or [over_now]
    w = {"PP":0,"MID":0,"DEATH":0}
    for o in overs: w["PP" if o<=6 else "MID" if o<=15 else "DEATH"] += 1
    t = sum(w.values()); return {k:v/t for k,v in w.items()}

def recommend(candidates, over_now, bowlers, need, label):
    """bowlers = list of (name, overs_left). need in [0,1]: 1=attack, 0=protect wicket."""
    pw = phase_weights(over_now); tot_ov = sum(o for _, o in bowlers)
    rows = []
    for bat in candidates:
        if bat not in prof.index: continue
        pf = sum(pw[p]*_get(prof, bat, f"edge_{p}") for p in ["PP","MID","DEATH"])
        m_sr = m_ctrl = safe = 0.0; detail = []
        for bo, ov in bowlers:
            w = ov/tot_ov; sr, ctrl, kind = matchup_vs_bowler(bat, bo)
            m_sr += w*sr; m_ctrl += w*ctrl; safe += w*_get(mk_dis, bat, kind)
            detail.append((bo, sr + 2.5*ctrl))
        upside = pf + m_sr + 2.5*m_ctrl
        score = need*upside + (1-need)*safe*12
        best = max(detail, key=lambda d: d[1]); worst = min(detail, key=lambda d: d[1])
        rows.append({"bat":bat, "score":round(score,1), "phase":round(pf,1),
                     "matchup":round(m_sr+2.5*m_ctrl,1),
                     "best_vs":best[0], "tough_vs":worst[0]})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\n### {label}")
    print(f"    over {over_now} done | bowlers left: " + ", ".join(f'{b}({o})' for b,o in bowlers) + f" | aggression {need:.2f}")
    for i, r in df.iterrows():
        tag = "  <-- SEND IN" if i==0 else ""
        print(f"    {i+1}. {r.bat:18s} score {r.score:+6.1f} (phase {r.phase:+.0f}, matchup {r.matchup:+.0f})  best vs {r.best_vs}, tough vs {r.tough_vs}{tag}")
    return df

CANDS = ["Suryakumar Yadav","Tilak Varma","Tim David","Hardik Pandya","Nicholas Pooran","Heinrich Klaasen","Ravindra Jadeja"]
# Death chase, must attack: pace-heavy attack with one spinner
recommend(CANDS, over_now=15, need=0.9, label="A. Death chase, must attack; pace + one spinner left",
          bowlers=[("Jasprit Bumrah",2),("Matheesha Pathirana",2),("Yuzvendra Chahal",1)])
# Rebuild vs spin, protect wicket
recommend(CANDS, over_now=9, need=0.3, label="B. Rebuild after wickets; two spinners operating",
          bowlers=[("Yuzvendra Chahal",3),("Ravindra Jadeja",2),("Jasprit Bumrah",3)])
print("\n(Bowler-aware. Matchup = pace/spin SR-edge + line/length control-edge vs the exact bowlers, weighted by their overs.)")
