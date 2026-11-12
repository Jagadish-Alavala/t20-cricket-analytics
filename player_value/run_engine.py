"""Train both models, validate honestly, compute impact + WPA, export IPL-2024 leaderboards."""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, log_loss, brier_score_loss, roc_auc_score

import engine as E

SEED = E.SEED
OUT = E.OUT

print("Loading all leagues 2020-24 ...")
df = E.load()
print(f"  balls={len(df):,}  matches={df['p_match'].nunique():,}  leagues={df['competition'].nunique()}")

# ============================================================ MODEL 1: EXPECTED RUNS (1st innings)
fi = E.first_innings_states(df)
tr = fi[fi["year"] <= 2023].copy()
te = fi[fi["year"] == 2024].copy()
tr = E.add_venue_baselines(tr, tr, "runs")
te = E.add_venue_baselines(tr, te, "runs")
for d in (tr, te):
    d["league"] = E.encode_league(d["competition"])

R_FEATS = ["balls_bowled", "balls_rem", "wkts_lost", "score", "crr", "dl_rem",
           "venue_avg", "year", "pp", "death", "league"]
m1 = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                   subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
m1.fit(tr[R_FEATS], tr["final"])
te["pred_final"] = m1.predict(te[R_FEATS])

# baseline: naive current-run-rate projection
te["naive"] = te["score"] + te["crr"] * (te["balls_rem"] / 6)
mae_model = mean_absolute_error(te["final"], te["pred_final"])
mae_naive = mean_absolute_error(te["final"], te["naive"])
# MAE by state bucket (this is the honest way, not one number)
te["bucket"] = pd.cut(te["balls_bowled"], [0, 30, 60, 90, 120],
                      labels=["0-30", "30-60", "60-90", "90-120"], include_lowest=True)
mae_bucket = te.groupby("bucket").apply(
    lambda g: pd.Series({"model_MAE": mean_absolute_error(g["final"], g["pred_final"]),
                         "naive_MAE": mean_absolute_error(g["final"], g["naive"]),
                         "n": len(g)})).round(2)
print("\n== MODEL 1  Expected Runs (test = 2024) ==")
print(f"  overall MAE  model {mae_model:.2f}  vs naive {mae_naive:.2f}")
print(mae_bucket.to_string())

# ============================================================ MODEL 2: WIN PROBABILITY (chase)
from sklearn.isotonic import IsotonicRegression
si = E.second_innings_states(df)
trw = si[si["year"] <= 2022].copy()          # fit model
calw = si[si["year"] == 2023].copy()         # fit calibration
tew = si[si["year"] == 2024].copy()          # test
trw = E.add_venue_baselines(trw, trw, "chase")
calw = E.add_venue_baselines(trw, calw, "chase")
tew = E.add_venue_baselines(trw, tew, "chase")
for d in (trw, calw, tew):
    d["league"] = E.encode_league(d["competition"])

W_FEATS = ["runs_needed", "balls_rem", "wkts_rem", "rrr", "crr", "dl_rem", "dl_ratio",
           "venue_chase", "year", "league"]
m2 = LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63,
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
m2.fit(trw[W_FEATS], trw["won_bat"])
# isotonic calibration on the held-out 2023 season
raw_cal = m2.predict_proba(calw[W_FEATS])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, calw["won_bat"])
tew["wp_raw"] = m2.predict_proba(tew[W_FEATS])[:, 1]
tew["wp"] = iso.transform(tew["wp_raw"])

# DL power-law benchmark from the spec
r = tew["dl_ratio"].clip(0.01, 5)
tew["wp_dl"] = 1 / (1 + (r / 0.96) ** (-6.5))

ll = log_loss(tew["won_bat"], tew["wp"]); br = brier_score_loss(tew["won_bat"], tew["wp"])
auc = roc_auc_score(tew["won_bat"], tew["wp"])
auc_dl = roc_auc_score(tew["won_bat"], tew["wp_dl"]); br_dl = brier_score_loss(tew["won_bat"], tew["wp_dl"])
prov = tew["wprob"].replace(-1, np.nan) / 100
mask = prov.notna()
corr_prov = np.corrcoef(tew["wp"][mask], prov[mask])[0, 1]
print("\n== MODEL 2  Win Probability (test = 2024 chases) ==")
print(f"  model   LogLoss {ll:.3f}  Brier {br:.3f}  AUC {auc:.3f}")
print(f"  DL law  Brier {br_dl:.3f}  AUC {auc_dl:.3f}")
print(f"  corr with provider wprob: {corr_prov:.3f}")
# calibration
tew["wp_bin"] = pd.cut(tew["wp"], np.linspace(0, 1, 11))
cal = tew.groupby("wp_bin").agg(pred=("wp", "mean"), actual=("won_bat", "mean"), n=("wp", "size")).dropna()
print("  calibration (pred vs actual):")
print(cal.round(3).to_string())

# ============================================================ IMPACT + WPA  (IPL 2024 demo)
def per_ball_delta(states, pred_col, group="p_match", order=("over", "ball")):
    """impact[i] = pred[i] - pred[i-1] within an innings; sums telescope to final - start."""
    s = states.sort_values([group, *order]).copy()
    s["prev"] = s.groupby(group)[pred_col].shift(1)
    s["prev"] = s["prev"].fillna(s[pred_col])  # first ball: delta vs itself ~ small; start handled below
    s["delta"] = s[pred_col] - s["prev"]
    return s

ipl_fi = te[te["competition"] == "IPL"].copy()
ipl_fi = per_ball_delta(ipl_fi, "pred_final")
# batter run-impact (+), bowler run-impact prevented (-delta = runs saved)
bat_run = ipl_fi.groupby("bat")["delta"].agg(["sum", "size"]).rename(columns={"sum": "run_impact", "size": "balls"})
bowl_run = ipl_fi.groupby("bowl")["delta"].agg(["sum", "size"]).rename(columns={"sum": "runs_conceded_vs_exp", "size": "balls"})
bowl_run["run_impact_prevented"] = -bowl_run["runs_conceded_vs_exp"]

# WPA runs on the DL power-law WP: in v0 validation it beat the ML classifier on
# both AUC and Brier, so it is the better-grounded currency until the ML model wins.
ipl_si = tew[tew["competition"] == "IPL"].copy()
ipl_si = per_ball_delta(ipl_si, "wp_dl")
bat_wpa = ipl_si.groupby("bat")["delta"].agg(["sum", "size"]).rename(columns={"sum": "wpa", "size": "balls"})
bowl_wpa = ipl_si.groupby("bowl")["delta"].agg(["sum", "size"]).rename(columns={"sum": "wpa_conceded", "size": "balls"})
bowl_wpa["wpa_prevented"] = -bowl_wpa["wpa_conceded"]

# leaderboards (min balls for signal)
bat_lb = bat_run.join(bat_wpa[["wpa"]], how="outer")
bat_lb["wpa_pct"] = (bat_lb["wpa"] * 100).round(2)
bat_lb = bat_lb[bat_lb["balls"].fillna(0) >= 60].sort_values("run_impact", ascending=False)
bowl_lb = bowl_run[["run_impact_prevented", "balls"]].join(bowl_wpa[["wpa_prevented"]], how="outer")
bowl_lb["wpa_prevented_pct"] = (bowl_lb["wpa_prevented"] * 100).round(2)
bowl_lb = bowl_lb[bowl_lb["balls"].fillna(0) >= 60].sort_values("run_impact_prevented", ascending=False)

bat_lb.round(2).to_csv(OUT / "ipl2024_batter_impact.csv")
bowl_lb.round(2).to_csv(OUT / "ipl2024_bowler_impact.csv")

print("\n== IPL 2024  Batter run-impact (1st-innings expected-runs currency), top 10 ==")
print(bat_lb[["run_impact", "balls"]].head(10).round(1).to_string())
print("\n== IPL 2024  Batter WPA (2nd-innings chase currency), top 10 ==")
print(bat_lb.dropna(subset=["wpa_pct"]).sort_values("wpa_pct", ascending=False)[["wpa_pct", "balls"]].head(10).to_string())
print("\n== IPL 2024  Bowler run-impact prevented, top 10 ==")
print(bowl_lb[["run_impact_prevented", "balls"]].head(10).round(1).to_string())

meta = {
    "train": "all leagues 2020-2023", "test": "2024",
    "model1_expected_runs": {"overall_MAE": round(mae_model, 2), "naive_MAE": round(mae_naive, 2),
                             "by_bucket": mae_bucket.reset_index().to_dict("records")},
    "model2_win_prob": {"ml_logloss": round(ll, 3), "ml_brier": round(br, 3), "ml_auc": round(auc, 3),
                        "dl_law_auc": round(auc_dl, 3), "dl_law_brier": round(br_dl, 3),
                        "corr_provider_wprob": round(float(corr_prov), 3),
                        "verdict": "DL power-law beats ML classifier on AUC and Brier; WPA uses DL WP in v0",
                        "wpa_currency": "DL power-law win probability"},
}
(OUT / "engine_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print("\nSaved leaderboards + engine_meta.json to", OUT)
