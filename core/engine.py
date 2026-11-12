"""
Cricket Impact Analytics Engine  (v0 kickoff)
=============================================

Two models, two currencies, deliberately NOT blended:

  Model 1  Expected Runs   -> first-innings setting.  Per-ball run impact = delta of expected final score.
  Model 2  Win Probability -> second-innings chase.   Per-ball WPA        = delta of win probability.

Design guards (carried over from the EBI review):
  - No hand-picked rating weights. Leaderboards are reported per currency, not fused.
  - Time-aware split: train seasons <= 2023, test 2024 (walk-forward, no leakage).
  - Venue / chase baselines are computed on TRAIN rows only, then mapped to test.
  - Targets reported by match-state bucket, because a single MAE number is meaningless
    (error at ball 3 is structurally huge, at ball 110 tiny).
  - DL resource (dlxr) used as a physics-informed feature, and as a benchmark for WP.

Python 3.10+, deterministic seed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
# folder holding t20_2020.csv ... t20_2024.csv (not in the repo); default: the repo's parent
DATA = Path(os.environ.get("T20_DATA", ROOT.parent))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- DL resource
def dlxr(u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Duckworth-Lewis style expected runs available with u balls left and w wickets down.
    Provided by the project spec. u = balls remaining, w = wickets lost (0-9)."""
    u = np.asarray(u, float); w = np.clip(np.asarray(w, float), 0, 9)
    beta = 5.73e-2 * w**3 + 4.21e-1 * w**2 - 22.19 * w + 137.61
    alpha = 1.087e-1 * w**3 + 0.6637 * w**2 - 44.69 * w + 286.07
    beta = np.where(beta <= 1, 1.0, beta)
    return alpha * (1 - np.exp(-u / beta))


# ----------------------------------------------------------------------------- load
def load(seasons=(2020, 2021, 2022, 2023, 2024)) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        d = pd.read_csv(DATA / f"t20_{yr}.csv", low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # standard 20-over games only, drop rain-shortened for modelling stability
    df = df[df["max_balls"] == 120].copy()
    for c in ["over", "inns_runs", "inns_wkts", "inns_balls_rem", "score", "batruns", "target"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["won_bat"] = (df["winner"] == df["team_bat"]).astype(int)
    return df


# ----------------------------------------------------------------------------- feature build
def first_innings_states(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ball of every first innings, at the AFTER-ball match state, with the final score label."""
    d = df[df["inns"] == 1].copy()
    d["balls_rem"] = d["inns_balls_rem"].clip(0, 120)
    d["balls_bowled"] = 120 - d["balls_rem"]
    d["wkts_lost"] = d["inns_wkts"].clip(0, 10)
    d["score"] = d["inns_runs"]
    d["crr"] = d["score"] / (d["balls_bowled"].replace(0, np.nan) / 6)
    d["crr"] = d["crr"].fillna(0)
    d["dl_rem"] = dlxr(d["balls_rem"].values, d["wkts_lost"].values)
    d["pp"] = (d["over"] <= 6).astype(int)
    d["death"] = (d["over"] >= 16).astype(int)
    final = d.groupby(["p_match"])["score"].max().rename("final")
    d = d.merge(final, on="p_match", how="left")
    return d


def second_innings_states(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ball of every chase, at the AFTER-ball state, with the win label."""
    d = df[df["inns"] == 2].copy()
    d["balls_rem"] = d["inns_balls_rem"].clip(0, 120)
    d["balls_bowled"] = 120 - d["balls_rem"]
    d["wkts_rem"] = (10 - d["inns_wkts"]).clip(0, 10)
    d["score"] = d["inns_runs"]
    d["runs_needed"] = (d["target"] - d["score"]).clip(lower=0)
    d["rrr"] = d["runs_needed"] / (d["balls_rem"].replace(0, np.nan) / 6)
    d["rrr"] = d["rrr"].replace([np.inf, -np.inf], np.nan).fillna(36)
    d["crr"] = d["score"] / (d["balls_bowled"].replace(0, np.nan) / 6)
    d["crr"] = d["crr"].fillna(0)
    d["dl_rem"] = dlxr(d["balls_rem"].values, d["inns_wkts"].clip(0, 9).values)
    d["dl_ratio"] = d["dl_rem"] / d["runs_needed"].replace(0, np.nan)
    d["dl_ratio"] = d["dl_ratio"].replace([np.inf, -np.inf], np.nan).fillna(5.0).clip(0, 5)
    return d


def add_venue_baselines(train: pd.DataFrame, apply_to: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Leakage-safe: baselines learned on train only, mapped onto apply_to."""
    if kind == "runs":
        vb = train.groupby("ground")["final"].mean()
        gmean = train["final"].mean()
        apply_to = apply_to.copy()
        apply_to["venue_avg"] = apply_to["ground"].map(vb).fillna(gmean)
    else:  # chase success by venue
        wins = train.groupby("ground")["won_bat"].mean()
        gmean = train["won_bat"].mean()
        apply_to = apply_to.copy()
        apply_to["venue_chase"] = apply_to["ground"].map(wins).fillna(gmean)
    return apply_to


def encode_league(s: pd.Series) -> pd.Series:
    cats = ["IPL", "T20I", "T20 Blast", "PSL", "SA20", "Super Smash", "BBL", "CPL", "The Hundred"]
    m = {c: i for i, c in enumerate(cats)}
    return s.map(m).fillna(-1).astype(int)
