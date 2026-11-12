"""Entry-point / batting-order optimizer (advisory).

Given a mid-match state, rank the batters still to come by expected value for the passage
of play remaining, using ONLY the validated inputs:
  - phase-value profile  (SR-edge by PP/MID/DEATH)     -> phase fit
  - pace/spin matchup    (SR-edge, dismissal-edge)     -> matchup fit
The 'aggression need' (required rate vs par, or wickets in hand) tilts the score between
upside (SR) and safety (not getting out). Output is a ranked recommendation with reasons.

This is ADVISORY. The counterfactual of a real captain's decision cannot be validated, so
the tool explains its reasoning and never claims an optimum.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E

prof = pd.read_csv(E.OUT / "batter_phase_profile.csv")            # edge_PP/MID/DEATH, balls_*
mk = pd.read_csv(E.OUT / "matchup_batter_vs_kind.csv")            # bat,kind,sr_edge,dis_edge,balls
prof = prof.set_index("bat")
mk_sr = mk.pivot_table(index="bat", columns="kind", values="sr_edge")
mk_dis = mk.pivot_table(index="bat", columns="kind", values="dis_edge")

def phase_weights(over_now: int):
    """Share of the remaining overs that fall in MID vs DEATH (PP if very early)."""
    overs = list(range(over_now + 1, 21))
    if not overs: overs = [over_now]
    w = {"PP": 0, "MID": 0, "DEATH": 0}
    for o in overs:
        w["PP" if o <= 6 else "MID" if o <= 15 else "DEATH"] += 1
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}

def score_batter(bat, pw, pace_share, need):
    """need in [0,1]: 1 = must attack (high req rate / lots of wickets left),
                      0 = protect wicket (low req rate / few wickets left)."""
    if bat not in prof.index:
        return None
    # phase fit: weighted SR-edge over the phases remaining
    pf = sum(pw[p] * (prof.loc[bat].get(f"edge_{p}", np.nan) if pd.notna(prof.loc[bat].get(f"edge_{p}", np.nan)) else 0)
             for p in ["PP", "MID", "DEATH"])
    # matchup fit: SR-edge weighted by remaining pace/spin share
    sp = 1 - pace_share
    msr = pace_share * (mk_sr.loc[bat, "pace"] if bat in mk_sr.index and pd.notna(mk_sr.loc[bat].get("pace")) else 0) \
        + sp * (mk_sr.loc[bat, "spin"] if bat in mk_sr.index and pd.notna(mk_sr.loc[bat].get("spin")) else 0)
    # safety: dismissal-edge (higher = safer) weighted by matchup
    safe = pace_share * (mk_dis.loc[bat, "pace"] if bat in mk_dis.index and pd.notna(mk_dis.loc[bat].get("pace")) else 0) \
         + sp * (mk_dis.loc[bat, "spin"] if bat in mk_dis.index and pd.notna(mk_dis.loc[bat].get("spin")) else 0)
    upside = pf + msr
    score = need * upside + (1 - need) * (safe * 12)     # scale dismissal-edge to SR units
    return {"bat": bat, "score": round(score, 1), "phase_fit": round(pf, 1),
            "matchup_sr": round(msr, 1), "safety": round(safe, 2)}

def recommend(candidates, over_now, pace_share, need, label):
    pw = phase_weights(over_now)
    rows = [r for b in candidates if (r := score_batter(b, pw, pace_share, need))]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\n### {label}")
    print(f"    state: over {over_now} done, remaining phase mix {({k:round(v,2) for k,v in pw.items() if v>0})}, "
          f"pace share {pace_share:.0%}, aggression need {need:.2f}")
    for i, r in df.iterrows():
        tag = "  <-- SEND IN" if i == 0 else ""
        print(f"    {i+1}. {r.bat:20s} score {r.score:+6.1f}  (phase {r.phase_fit:+.0f}, matchup {r.matchup_sr:+.0f}, safety {r.safety:+.2f}){tag}")
    return df

CANDS = ["Suryakumar Yadav", "Tilak Varma", "Tim David", "Hardik Pandya",
         "Nicholas Pooran", "Marcus Stoinis", "Heinrich Klaasen", "Ravindra Jadeja"]

# Scenario A: death slog, high required rate, spinners done, pace to come
recommend(CANDS, over_now=15, pace_share=0.85, need=0.90,
          label="A. Chase, 15 overs gone, ~13 an over needed, only pace left")
# Scenario B: early collapse, must rebuild, spin operating, low aggression
recommend(CANDS, over_now=9, pace_share=0.30, need=0.30,
          label="B. Chase, 3 down cheap in over 9, spinners on, rebuild needed")
print("\n(Advisory. Ranks candidates by validated phase + matchup fit for the passage remaining.)")
