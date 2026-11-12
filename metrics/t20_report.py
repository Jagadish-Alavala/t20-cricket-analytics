"""Batting and bowling report over any date range, from t20_ball_metrics (run t20_metrics.py first).

  python t20_report.py --since 2025-01-01                  # all men's T20 since 2025, top batters by runs
  python t20_report.py --since 2026-01-01 --league IPL --bowling
  python t20_report.py --match 2024-06-29                  # Impact scorecard (primer §5.8, Table 6); a date lists that day's matches
  python t20_report.py --seasons --league IPL --min-balls 100   # player-seasons ranked by WPA (add --bowling for bowlers)
Writes the full table to exports/t20_<batting|bowling>_<since>[_<league>].csv.

Batting: balls faced exclude wides; runs are off the bat. SR10 = strike rate over the first 10 balls of each innings.
B/DIS, B/4, B/6, B/BDY = balls per dismissal / four / six / boundary. BDY% = boundaries per ball, BRNS% = share of
runs from boundaries, NB-SR = strike rate off non-boundary balls, DOT% = balls with no run at all, RUN% = balls
with 1-3 off the bat. PP / MID / DTH = overs 1-6 / 7-15 / 16-20. 30+% = innings of 30 or more.
Impact, RAA, WAA, WPA: see t20_metrics.py; /100 is per 100 balls, /INNS per innings. ADJ IMP/100: Impact adjusted for
the quality of the opposing bowlers (batting) or batters (bowling), primer §5.3.
"""
import argparse, os, sqlite3
import numpy as np, pandas as pd


def batting(d):
    b = d[~d.wide.astype(bool)].copy()
    b["bdy"] = b.bat_runs.isin([4, 6])  # ponytail: an all-run 4 counts as a boundary; the feeds don't say
    b["ball_no"] = b.groupby(["batter_key", "match_id", "innings"]).cumcount()
    phase = pd.cut(b.balls_before, [-1, 35, 89, 999], labels=["pp", "mid", "dth"])
    # dismissals by who was out, so a non-striker run out counts against the non-striker
    out = d[d.is_wicket.astype(bool)]
    outs = out.out_key.fillna(out.batter_key).value_counts()
    inns = b.groupby(["batter_key", "match_id", "innings"]).bat_runs.sum().groupby("batter_key")
    g = b.groupby("batter_key")
    sr = lambda r, n: 100 * r / n.replace(0, np.nan)
    t = pd.DataFrame({
        "player": g.batter_name.first(), "balls": g.size(), "runs": g.bat_runs.sum(), "hs": inns.max(),
        "fours": g.bat_runs.agg(lambda s: (s == 4).sum()), "sixes": g.bat_runs.agg(lambda s: (s == 6).sum()),
        "bdy_runs": b[b.bdy].groupby("batter_key").bat_runs.sum(), "dots": b[b.runs == 0].groupby("batter_key").size(),
        "ones_to_threes": b[b.bat_runs.between(1, 3)].groupby("batter_key").size(),
        "first10_runs": b[b.ball_no < 10].groupby("batter_key").bat_runs.sum(), "first10_balls": b[b.ball_no < 10].groupby("batter_key").size(),
        "inns": inns.size(), "thirties": inns.agg(lambda s: (s >= 30).sum()),
        "impact": g.impact.sum(), "raa": g.raa.sum(), "exp_outs": g.exp_wkts.sum(), "adj": g.adj_bat.sum(), "wpa": g.wpa.sum(),
    }).fillna(0)
    for ph in ["pp", "mid", "dth"]:
        t[f"{ph}_sr"] = sr(b[phase == ph].groupby("batter_key").bat_runs.sum(), b[phase == ph].groupby("batter_key").size())
    t["outs"] = outs.reindex(t.index).fillna(0)
    t["waa"] = t.exp_outs - t.outs  # own dismissals: a non-striker run out counts against the batter who was out
    bdys = t.fours + t.sixes
    return pd.DataFrame({
        "PLAYER": t.player, "BALLS": t.balls, "RUNS": t.runs, "HS": t.hs, "SR": sr(t.runs, t.balls),
        "SR10": sr(t.first10_runs, t.first10_balls), "AVG": t.runs / t.outs.replace(0, np.nan),
        "B/DIS": t.balls / t.outs.replace(0, np.nan), "B/4": t.balls / t.fours.replace(0, np.nan),
        "B/6": t.balls / t.sixes.replace(0, np.nan), "B/BDY": t.balls / bdys.replace(0, np.nan),
        "BDY%": 100 * bdys / t.balls, "BRNS%": 100 * t.bdy_runs / t.runs.replace(0, np.nan),
        "NB-SR": sr(t.runs - t.bdy_runs, t.balls - bdys), "DOT%": 100 * t.dots / t.balls, "RUN%": 100 * t.ones_to_threes / t.balls,
        "PP SR": t.pp_sr, "MID SR": t.mid_sr, "DTH SR": t.dth_sr, "30+%": 100 * t.thirties / t.inns,
        "IMPACT": t.impact, "IMP/100": 100 * t.impact / t.balls, "RAA": t.raa, "RAA/100": 100 * t.raa / t.balls,
        "WAA": t.waa, "WAA/100": 100 * t.waa / t.balls, "INNS": t.inns, "IMP/INNS": t.impact / t.inns,
        "RAA/INNS": t.raa / t.inns, "WAA/INNS": t.waa / t.inns, "ADJ IMP/100": 100 * t.adj / t.balls,
        "WPA": t.wpa, "WPA/100": 100 * t.wpa / t.balls,
    }).sort_values("RUNS", ascending=False)


def bowling(d):
    b = d[~d.wide.astype(bool)]  # wides left out, as in the primer
    g = b.groupby("bowler_key")
    # balls include no-balls (as the primer counts them); signs flipped so positive is good for the bowler
    t = pd.DataFrame({"player": g.bowler.first(), "balls": g.size(), "runs": g.runs.sum(), "inns": g.match_id.nunique(),
                      "wkts": (b.is_wicket.astype(bool) & ~b.run_out.astype(bool)).groupby(b.bowler_key).sum(),
                      "impact": -g.bowl_impact.sum(), "raa": -g.raa_runs.sum(), "waa": -g.waa.sum(),
                      "adj": g.adj_bowl.sum(), "wpa": -g.wpa.sum()})
    t = t[~t.index.str.startswith("?:")]  # bowler not identified
    return pd.DataFrame({
        "PLAYER": t.player, "BALLS": t.balls, "RUNS": t.runs, "WKTS": t.wkts, "ECON": 6 * t.runs / t.balls,
        "IMPACT": t.impact, "IMP/100": 100 * t.impact / t.balls, "RAA": t.raa, "RAA/100": 100 * t.raa / t.balls,
        "WAA": t.waa, "WAA/100": 100 * t.waa / t.balls, "INNS": t.inns, "IMP/INNS": t.impact / t.inns,
        "RAA/INNS": t.raa / t.inns, "WAA/INNS": t.waa / t.inns, "ADJ IMP/100": 100 * t.adj / t.balls,
        "WPA": t.wpa, "WPA/100": 100 * t.wpa / t.balls,
    }).sort_values("IMPACT", ascending=False)


def scorecard(d):
    """Primer §5.8 / Table 6: each side's total Impact, Impact by over, and players by WPA and Impact."""
    for inn, x in d.groupby("innings"):
        print(f"\n{inn}{'st' if inn == 1 else 'nd'} innings: {x.batting_team.iat[0]} {x.runs.sum()}-{x.is_wicket.sum()}, "
              f"Impact {x.impact.sum():+.1f}, par {x.par.iat[0]:.0f}")
        by_over = x.groupby(x.balls_before // 6 + 1).impact.sum()
        print("  Impact by over: " + " ".join(f"{o}:{v:+.0f}" for o, v in by_over.items()))
        for who, key, imp, sign in (("batting", "batter_name", "impact", 1), ("bowling", "bowler", "bowl_impact", -1)):
            g = x.groupby(key).agg(balls=("runs", "size"), wpa=("wpa", "sum"), impact=(imp, "sum"))
            g[["wpa", "impact"]] *= sign
            g["wpa/100b"] = 100 * g.wpa / g.balls
            print(f"  {who}:\n" + g.sort_values("wpa", ascending=False)[["wpa", "wpa/100b", "impact"]].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--match", help="a match id, or a date (YYYY-MM-DD) to list that day's matches")
    ap.add_argument("--until", default="9999-12-31")
    ap.add_argument("--league", help="IPL, BBL, T20 Blast, CPL, SMAT, T20I, ...")
    ap.add_argument("--bowling", action="store_true")
    ap.add_argument("--min-balls", type=int, default=100)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--seasons", action="store_true", help="one row per player-season, ranked by WPA")
    a = ap.parse_args()
    con = sqlite3.connect(os.environ.get("CRICKET_DB", "cricket.db"), timeout=120)
    os.makedirs("exports", exist_ok=True)
    if a.match:
        d = pd.read_sql("SELECT * FROM t20_ball_metrics WHERE match_id = ? OR substr(start_date, 1, 10) = ?", con, params=[a.match] * 2)
        ids = d.match_id.unique()
        if len(ids) == 1:
            return scorecard(d)
        for mid, x in d.groupby("match_id"):
            print(mid, " v ".join(x.batting_team.unique()), x.venue.iat[0])
        return
    if a.seasons:
        who, view = ("bowler", "t20_bowling") if a.bowling else ("batter", "t20_batting")
        t = pd.read_sql(f"SELECT {who} AS player, year, league, ROUND(wpa, 2) AS wpa, balls, "
                        f"{'runs_conceded' if a.bowling else 'runs'} AS runs, ROUND(impact, 2) AS imp, raa, waa FROM {view}"
                        f" WHERE balls >= ? AND year BETWEEN ? AND ?{' AND league = ?' if a.league else ''} ORDER BY wpa DESC",
                        con, params=[a.min_balls, (a.since or "0")[:4], a.until[:4]] + ([a.league] if a.league else []))
        path = f"exports/t20_{who}_seasons{'_' + a.league if a.league else ''}.csv"
        t.to_csv(path, index=False)
        print(t.head(a.top).to_string())
        return print(f"\n{len(t)} {who}-seasons, full table in {path}")
    if not a.since:
        ap.error("--since, --match or --seasons is needed")
    sql, args = "SELECT * FROM t20_ball_metrics WHERE start_date >= ? AND start_date <= ?", [a.since, a.until]
    if a.league:
        sql, args = sql + " AND league = ?", args + [a.league]
    d = pd.read_sql(sql, con, params=args)
    t = (bowling if a.bowling else batting)(d)
    t = t[t.BALLS >= a.min_balls].round(2)
    path = f"exports/t20_{'bowling' if a.bowling else 'batting'}_{a.since}{'_' + a.league if a.league else ''}.csv"
    t.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t.head(a.top).to_string(index=False))
    print(f"\n{len(t)} players, full table in {path}")


if __name__ == "__main__":
    main()
