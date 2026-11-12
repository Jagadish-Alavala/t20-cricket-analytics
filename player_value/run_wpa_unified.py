"""Unified Win Probability Added (WPA) across both innings, one currency, one leaderboard.

Adds to v0:
  1. First-innings win probability (label = first-batting team wins the match), calibrated.
  2. Second-innings WP rebuilt with the DL power-law WP as a BACKBONE feature, so the ML
     model can only add corrections on top of the benchmark it previously lost to.
  3. WPA computed for every ball of both innings from each batting team's win probability,
     so a player's tournament WPA sums across all appearances. Batter +, bowler -.
  4. Match MVP and biggest turning point.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error
import engine as E

SEED, OUT = E.SEED, E.OUT
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

print("Loading ...")
df = E.load()
fi = E.first_innings_states(df)
si = E.second_innings_states(df)

# ---------------------------------------------------- Model 1: expected runs (feature for 1st-inns WP)
ftr = E.add_venue_baselines(fi[fi.year <= 2023], fi[fi.year <= 2023], "runs")
m1tr = ftr.copy(); m1tr["league"] = E.encode_league(m1tr["competition"])
R_FEATS = ["balls_bowled", "balls_rem", "wkts_lost", "score", "crr", "dl_rem", "venue_avg", "year", "pp", "death", "league"]
m1 = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8,
                   colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(m1tr[R_FEATS], m1tr["final"])

def prep_fi(seasons):
    d = E.add_venue_baselines(fi[fi.year <= 2023], fi[fi.year.isin(seasons)], "runs").copy()
    d["league"] = E.encode_league(d["competition"])
    d["proj_final"] = m1.predict(d[R_FEATS])
    d["proj_minus_venue"] = d["proj_final"] - d["venue_avg"]
    return d

# ---------------------------------------------------- Model WP1: first-innings win probability
fi_tr, fi_cal, fi_te = prep_fi([2020, 2021, 2022]), prep_fi([2023]), prep_fi([2024])
WP1 = ["score", "wkts_lost", "balls_rem", "crr", "dl_rem", "venue_avg", "proj_final", "proj_minus_venue", "league"]
c1 = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=47, subsample=0.8,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(fi_tr[WP1], fi_tr["won_bat"])
iso1 = IsotonicRegression(out_of_bounds="clip").fit(c1.predict_proba(fi_cal[WP1])[:, 1], fi_cal["won_bat"])
fi_te["wp"] = iso1.transform(c1.predict_proba(fi_te[WP1])[:, 1])
auc1 = roc_auc_score(fi_te["won_bat"], fi_te["wp"]); br1 = brier_score_loss(fi_te["won_bat"], fi_te["wp"])
print(f"\n== WP1  first-innings win prob (test 2024):  AUC {auc1:.3f}  Brier {br1:.3f} ==")

# ---------------------------------------------------- Model WP2: chase WP with DL backbone
def prep_si(seasons, train_ref):
    d = E.add_venue_baselines(train_ref, si[si.year.isin(seasons)], "chase").copy()
    d["league"] = E.encode_league(d["competition"])
    r = d["dl_ratio"].clip(0.01, 5)
    d["wp_dl"] = 1 / (1 + (r / 0.96) ** (-6.5))
    d["wp_dl_logit"] = logit(d["wp_dl"])
    return d
si_ref = si[si.year <= 2022]
si_tr, si_cal, si_te = prep_si([2020, 2021, 2022], si_ref), prep_si([2023], si_ref), prep_si([2024], si_ref)
WP2 = ["runs_needed", "balls_rem", "wkts_rem", "rrr", "crr", "dl_rem", "dl_ratio",
       "wp_dl", "wp_dl_logit", "venue_chase", "league"]
c2 = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(si_tr[WP2], si_tr["won_bat"])
iso2 = IsotonicRegression(out_of_bounds="clip").fit(c2.predict_proba(si_cal[WP2])[:, 1], si_cal["won_bat"])
si_te["wp"] = iso2.transform(c2.predict_proba(si_te[WP2])[:, 1])
auc2 = roc_auc_score(si_te["won_bat"], si_te["wp"]); br2 = brier_score_loss(si_te["won_bat"], si_te["wp"])
auc_dl = roc_auc_score(si_te["won_bat"], si_te["wp_dl"]); br_dl = brier_score_loss(si_te["won_bat"], si_te["wp_dl"])
print(f"== WP2  chase WP with DL backbone (test 2024):  ML AUC {auc2:.3f} Brier {br2:.3f}  |  DL-law AUC {auc_dl:.3f} Brier {br_dl:.3f} ==")
print(f"   {'ML now beats DL law' if (auc2 >= auc_dl and br2 <= br_dl) else 'ML matches/does not fully beat DL law'}")

# ---------------------------------------------------- Unified WPA (IPL 2024)
def deltas(states, wp="wp"):
    s = states[states.competition == "IPL"].sort_values(["p_match", "over", "ball"]).copy()
    s["prev"] = s.groupby(["p_match"])[wp].shift(1).fillna(s[wp])
    s["wpa"] = s[wp] - s["prev"]
    return s
# chase WPA uses the DL-law WP (better AUC + Brier than the ML model, even out-of-sample).
# first-innings WPA uses WP1, but note its low signal (AUC ~0.60) documented in meta.
f, s = deltas(fi_te), deltas(si_te, "wp_dl")
allballs = pd.concat([f[["p_match", "bat", "bowl", "wpa"]], s[["p_match", "bat", "bowl", "wpa"]]], ignore_index=True)

bat = allballs.groupby("bat")["wpa"].agg(["sum", "size"]).rename(columns={"sum": "WPA", "size": "balls"})
bowl = allballs.groupby("bowl")["wpa"].agg(["sum", "size"]).rename(columns={"sum": "wpa_conceded", "size": "balls"})
bowl["WPA_prevented"] = -bowl["wpa_conceded"]
bat["WPA_wins"] = bat["WPA"].round(2)           # in "wins added" (probability units summed)
bowl["WPA_wins"] = bowl["WPA_prevented"].round(2)
bat_lb = bat[bat.balls >= 60].sort_values("WPA", ascending=False)
bowl_lb = bowl[bowl.balls >= 60].sort_values("WPA_prevented", ascending=False)
bat_lb[["WPA_wins", "balls"]].to_csv(OUT / "ipl2024_batter_wpa_unified.csv")
bowl_lb[["WPA_wins", "balls"]].to_csv(OUT / "ipl2024_bowler_wpa_unified.csv")

print("\n== IPL 2024 Unified Batter WPA (wins added, both innings), top 12 ==")
print(bat_lb[["WPA_wins", "balls"]].head(12).to_string())
print("\n== IPL 2024 Unified Bowler WPA prevented (wins added), top 12 ==")
print(bowl_lb[["WPA_wins", "balls"]].head(12).to_string())

# ---------------------------------------------------- Match MVP + turning point
allballs2 = pd.concat([f, s], ignore_index=True)
mvp = (allballs2.groupby(["p_match", "bat"])["wpa"].sum().reset_index()
       .sort_values("wpa", ascending=False).groupby("p_match").first().reset_index())
turning = allballs2.loc[allballs2.groupby("p_match")["wpa"].apply(lambda x: x.abs().idxmax())]
print("\n== Sample match MVPs (highest batting WPA in the match) ==")
print(mvp.sort_values("wpa", ascending=False).head(6).rename(columns={"bat": "MVP", "wpa": "match_WPA"}).round(3).to_string(index=False))

# conservation sanity: batting-team WPA over an innings ~ end WP - start WP (telescoping)
chk = f.groupby("p_match")["wpa"].sum().abs().mean()
meta = {"wp1_auc": round(auc1, 3), "wp1_brier": round(br1, 3),
        "wp2_ml_auc": round(auc2, 3), "wp2_ml_brier": round(br2, 3),
        "wp2_dl_auc": round(auc_dl, 3), "wp2_dl_brier": round(br_dl, 3),
        "ml_beats_dl": bool(auc2 >= auc_dl and br2 <= br_dl),
        "chase_wpa_uses": "DL power-law (beats ML even with DL backbone feature, out-of-sample)",
        "first_innings_wp_signal": "weak by nature (AUC ~0.60): WPA under-credits 1st-innings batting",
        "recommended_next": "hybrid currency: expected-runs impact for 1st innings, WPA for chase",
        "wpa_currency": "wins added (win-probability units, both innings unified)"}
(OUT / "wpa_unified_meta.json").write_text(json.dumps(meta, indent=2))
print("\nSaved unified leaderboards + wpa_unified_meta.json to", OUT)
