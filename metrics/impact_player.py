"""Did the Impact Player rule change T20 batting? IPL (rule from 2023) and SMAT (rule in the 2022-23 and 2023-24
seasons only, dropped from 2024-25) against the T20 Blast and CPL, which never had it. BBL is left out: its X-Factor
substitute (2020-21 to 2022-23) is a similar rule.

  python impact_player.py      # prints every table and writes exports/impact_player_*.csv

Only matches where both innings ran the full 20 overs (no rain cut), so totals compare like with like, and only
2019 on: the Blast, CPL and SMAT data start there (IPL goes back to 2008).
"""
import os, sqlite3
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

LEAGUES = ["IPL", "SMAT", "T20 Blast", "CPL"]


def rule_on(league, year):
    return (league == "IPL") & (year >= 2023) | (league == "SMAT") & year.isin([2022, 2023])


def load(con):
    d = pd.read_sql(f"""SELECT b.match_id, b.innings, b.seq, b.league, b.year, b.batting_team, b.runs, b.bat_runs,
                               b.is_wicket, b.wide, b.balls_before, b.wkts_before, b.score_before, b.batter_key, b.batter_name, b.bowler_key,
                               b.out_key, b.impact, b.raa, b.exp_wkts, b.wpa,
                               p.ground
                        FROM t20_ball_metrics b JOIN t20_par p USING (match_id)
                        WHERE p.clean = 1 AND p.balls1 = 120 AND p.balls2 = 120
                          AND b.league IN ({','.join('?' * len(LEAGUES))})
                          AND b.year >= 2019""", con, params=LEAGUES)  # the controls start in 2019
    d["wide"] = d.wide.astype(bool)
    d["rule"] = rule_on(d.league, d.year)
    return d


def totals(d):
    """First-innings totals per league-season, with run rate and wickets by phase."""
    f = d[d.innings == 1]
    f = f.assign(phase=pd.cut(f.balls_before, [-1, 35, 89, 999], labels=["pp", "mid", "death"]))
    inn = f.groupby(["league", "year", "match_id"]).agg(total=("runs", "sum"), wkts=("is_wicket", "sum"))
    t = inn.groupby(["league", "year"]).agg(matches=("total", "size"), total=("total", "mean"), wkts=("wkts", "mean"))
    t["rule"] = rule_on(t.index.get_level_values(0), pd.Series(t.index.get_level_values(1), index=t.index))
    legal = f[~f.wide]
    t["boundary%"] = 100 * legal.bat_runs.isin([4, 6]).groupby([legal.league, legal.year]).mean()
    for ph, x in f.groupby("phase", observed=True):
        t[f"{ph} rpo"] = 6 * x.groupby(["league", "year"]).runs.sum() / x[~x.wide].groupby(["league", "year"]).size()
    return t.round(2)


def did(m, y, controls):
    """Difference in differences: season fixed effects plus ground (when given) or league fixed effects, HC1 standard errors.
    The IPL and SMAT rule terms are separate, because the rule is not the same treatment in a franchise league with
    overseas stars as in a domestic one. smat_after (SMAT once the rule was dropped) keeps a later jump in Indian
    domestic scoring from being credited to the rule."""
    m = m[m.league.isin(["IPL", "SMAT"] + controls)].copy()
    m["rule_ipl"] = ((m.league == "IPL") & (m.year >= 2023)).astype(int)
    m["rule_smat"] = ((m.league == "SMAT") & m.year.isin([2022, 2023])).astype(int)
    m["smat_after"] = ((m.league == "SMAT") & (m.year >= 2024)).astype(int)
    terms = [t for t in ["rule_ipl", "rule_smat", "smat_after"] if m[t].any()]
    fe = "C(ground)" if "ground" in m else "C(league)"  # a ground belongs to one league, so it replaces the league term
    fit = smf.ols(f"{y} ~ {' + '.join(terms)} + {fe} + C(year)", m).fit(cov_type="HC1")
    return fit.params[terms].to_frame("effect").join(
        fit.conf_int().loc[terms].set_axis(["lo95", "hi95"], axis=1)).round(1).assign(n=len(m))


def first_totals(d):
    """One row per match: first-innings total. Ground absorbs the IPL's 2020-21 seasons in the UAE and 2022 in
    Mumbai and Pune."""
    return d[d.innings == 1].groupby("match_id").agg(total=("runs", "sum"), league=("league", "first"),
                                                    year=("year", "first"), ground=("ground", "first"))


def by_season(m):
    """IPL first-innings total minus the Blast and CPL in the same season, relative to 2019 (league and season fixed
    effects). A rule that works at once shows as a step in 2023; a gap that keeps growing points to something else."""
    m = m[m.league.isin(["IPL", "T20 Blast", "CPL"])].copy()
    years = sorted(m.year.unique())[1:]
    for y in years:
        m[f"ipl{y}"] = ((m.league == "IPL") & (m.year == y)).astype(int)
    fit = smf.ols(f"total ~ {' + '.join(f'ipl{y}' for y in years)} + C(league) + C(year)", m).fit(cov_type="HC1")
    ci = fit.conf_int()
    return pd.DataFrame({"gap": fit.params[[f"ipl{y}" for y in years]], "lo95": ci[0], "hi95": ci[1]}).dropna().set_axis(years).round(1)


def wicket_cost(d):
    """Runs a first innings adds after 10 overs, against wickets already down at that point (holding the score
    at 10 overs fixed). If a 12th player makes batting deeper, each early wicket should cost fewer runs."""
    f = d[d.innings == 1]
    total = f.groupby("match_id").runs.sum()
    at10 = f[f.balls_before == 60].groupby("match_id").first()  # first delivery of the 11th over
    at10 = at10.assign(rest=total - at10.score_before)
    rows = []
    for (lg, rule), x in at10.groupby(["league", "rule"]):
        fit = smf.ols("rest ~ wkts_before + score_before", x).fit(cov_type="HC1")
        lo, hi = fit.conf_int().loc["wkts_before"]
        rows.append({"league": lg, "rule": rule, "innings": len(x), "runs per wicket": fit.params.wkts_before,
                     "lo95": lo, "hi95": hi, "years": f"{x.year.min()}-{x.year.max()}"})
    return pd.DataFrame(rows).round(1)


def batting_order(d):
    """Each batter's position in each innings, taken as the order of the first ball they faced."""
    first = d.sort_values("seq").groupby(["match_id", "batting_team", "batter_key"]).seq.first().reset_index()
    first["pos"] = first.groupby(["match_id", "batting_team"]).seq.rank()
    return first


def allrounder_overs(d):
    """Share of each side's overs bowled by its own top-7 batters (batting order = order of first ball faced).
    Only sides that used at least 7 batters, so positions 1-7 are all known. The complaint about the rule is that
    teams stop asking batting all-rounders to bowl, because a specialist bowler can be brought in."""
    first = batting_order(d)
    deep = first.groupby(["match_id", "batting_team"]).pos.max() >= 7
    top7 = first[first.pos <= 7].set_index(["match_id", "batting_team", "batter_key"]).index
    teams = d.groupby("match_id").batting_team.unique()
    b = d[~d.wide].copy()
    b["bowling_team"] = [next(t for t in teams[m] if t != bt) if len(teams[m]) == 2 else None
                         for m, bt in zip(b.match_id, b.batting_team)]
    b = b.dropna(subset=["bowling_team"])
    b = b[deep.reindex(pd.MultiIndex.from_arrays([b.match_id, b.bowling_team])).fillna(False).values]
    b["by_top7"] = pd.MultiIndex.from_arrays([b.match_id, b.bowling_team, b.bowler_key]).isin(top7)
    side = b.groupby(["league", "year", "match_id", "bowling_team"]).by_top7.mean()
    t = side.groupby(["league", "year"]).agg(["size", "mean"]).rename(columns={"size": "sides", "mean": "top7 share"})
    t["top7 share"] = (100 * t["top7 share"]).round(1)
    return t, (100 * side).rename("share").reset_index()


def allrounder_players(d, min_matches=15):
    """IPL players who batted in their side's top 7 in at least min_matches games both before (2019-22) and after
    (2023-26) the rule, and bowled at least an over a game before it: overs per game and share of games bowled in."""
    i = d[d.league == "IPL"]
    first = batting_order(i)
    games = first[first.pos <= 7][["match_id", "batter_key"]].rename(columns={"batter_key": "player"})
    balls = i[~i.wide].groupby(["match_id", "bowler_key"]).size().rename("balls")
    games = games.join(balls, on=["match_id", "player"]).fillna({"balls": 0})
    games["after"] = games.match_id.map(i.groupby("match_id").rule.first())
    p = games.groupby(["player", "after"]).agg(games=("balls", "size"), overs=("balls", lambda b: b.mean() / 6),
                                              bowled=("balls", lambda b: 100 * (b > 0).mean())).unstack("after")
    p.columns = [f"{c} {'after' if a else 'before'}" for c, a in p.columns]
    p = p[(p["games before"] >= min_matches) & (p["games after"] >= min_matches) & (p["overs before"] >= 1)]
    p.insert(0, "name", i.groupby("batter_key").batter_name.first().reindex(p.index))
    p["change"] = p["overs after"] - p["overs before"]
    return p.sort_values("change").round(2)


def top7_split(d):
    """IPL, before and after the rule: how many of a side's top 7 bowled, and how many overs each of them bowled.
    Separates teams picking fewer bowling options in the top 7 from each all-rounder bowling less."""
    i = d[d.league == "IPL"]
    first = batting_order(i)
    deep = first.groupby(["match_id", "batting_team"]).pos.transform("max") >= 7
    t7 = first[deep & (first.pos <= 7)].join(i[~i.wide].groupby(["match_id", "bowler_key"]).size().rename("balls"),
                                             on=["match_id", "batter_key"]).fillna({"balls": 0})
    t7["after"] = t7.match_id.map(i.groupby("match_id").rule.first())
    side = t7.groupby(["after", "match_id", "batting_team"]).balls.agg(bowlers=lambda b: (b > 0).sum(), overs=lambda b: b.sum() / 6)
    t = side.groupby("after").mean()
    t["overs per bowler"] = t7[t7.balls > 0].groupby("after").balls.mean() / 6
    return t.rename(index={False: "2019-22", True: "2023-26"}).round(2)


def roles(d, n_boot=1000, seed=0):
    """Batting by order (1-3, 4-7, 8-11), IPL against the Blast and CPL, 2019-22 vs 2023-26: balls per innings,
    strike rate, and per 100 balls RAA, WAA, Impact and WPA (primer metrics from t20_metrics, not centred). The
    metrics share one pooled baseline, so a league-wide jump shows in every row: compare a group's change with the
    controls' (diff) or with the IPL's 4-7 (vs 4-7). 95% ranges from resampling matches."""
    d = d[d.league.isin(["IPL", "T20 Blast", "CPL"])].copy()
    pos = batting_order(d).set_index(["match_id", "batting_team", "batter_key"]).pos
    grp = lambda k: pd.cut(pos.reindex(pd.MultiIndex.from_arrays([d.match_id, d.batting_team, k])).values,
                           [0, 3, 7, 11], labels=["1-3", "4-7", "8-11"])
    d["grp"], d["out_grp"] = grp(d.batter_key), grp(d.out_key.fillna(d.batter_key))  # run outs: whoever was out
    d["ipl"], d["after"] = d.league == "IPL", d.year >= 2023
    d["out"] = d.is_wicket.where(d.is_wicket == 1)
    m = d[~d.wide].groupby(["match_id", "ipl", "after", "grp"], observed=True).agg(
        balls=("raa", "size"), runs=("bat_runs", "sum"), raa=("raa", "sum"), impact=("impact", "sum"),
        wpa=("wpa", "sum"), xw=("exp_wkts", "sum")).join(
        d.dropna(subset=["out"]).groupby(["match_id", "ipl", "after", "out_grp"], observed=True).size().rename("outs")
        .rename_axis(["match_id", "ipl", "after", "grp"])).fillna({"outs": 0}).reset_index()
    inns = m.groupby(["ipl", "after"]).match_id.nunique() * 2

    def table(x):
        t = x.groupby(["ipl", "after", "grp"], observed=True)[["balls", "runs", "raa", "impact", "wpa", "xw", "outs"]].sum()
        out = pd.DataFrame({"balls/inns": t.balls / inns.reindex(t.index.droplevel(2)).values, "sr": 100 * t.runs / t.balls})
        for c in ("raa", "impact", "wpa"):
            out[f"{c}/100"] = 100 * t[c] / t.balls
        out["waa/100"] = 100 * (t.xw - t.outs) / t.balls
        return out

    def changes(x):
        t = table(x)
        ch = t.xs(True, level="after") - t.xs(False, level="after")
        return pd.concat({"ipl change": ch.xs(True), "diff": ch.xs(True) - ch.xs(False),
                          "vs 4-7": ch.xs(True) - ch.xs(True).loc["4-7"]})
    est, rng = changes(m), np.random.default_rng(seed)
    ids = {k: g.match_id.unique() for k, g in m.groupby("ipl")}
    byid = m.set_index("match_id")
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(v, len(v)) for v in ids.values()])
        boots.append(changes(byid.loc[pick].reset_index()).stack())
    b = pd.concat(boots, axis=1)
    ci = pd.DataFrame({"lo95": b.quantile(0.025, axis=1), "hi95": b.quantile(0.975, axis=1)})
    t = table(m).rename(index={True: "IPL", False: "Blast+CPL"}, level=0).rename(index={True: "2023-26", False: "2019-22"}, level=1)
    return t.round(2), est.stack().rename("change").to_frame().join(ci).round(2)


def leaders(d, top=10):
    """IPL 2023-26 batters by total WPA, with the primer's rate stats."""
    i = d[(d.league == "IPL") & (d.year >= 2023) & ~d.wide]
    outs = d[(d.league == "IPL") & (d.year >= 2023) & (d.is_wicket == 1)].assign(k=lambda x: x.out_key.fillna(x.batter_key)).groupby("k").size()
    p = i.groupby("batter_key").agg(name=("batter_name", "first"), balls=("raa", "size"), runs=("bat_runs", "sum"),
                                    raa=("raa", "sum"), impact=("impact", "sum"), wpa=("wpa", "sum"), xw=("exp_wkts", "sum"))
    p["waa"] = p.xw - outs.reindex(p.index).fillna(0)
    p["sr"] = 100 * p.runs / p.balls
    for c in ("raa", "waa", "impact"):
        p[f"{c}/100"] = 100 * p[c] / p.balls
    return p.sort_values("wpa", ascending=False).head(top)[["name", "balls", "sr", "wpa", "raa/100", "waa/100", "impact/100"]].round(2)


def main():
    con = sqlite3.connect(os.environ.get("CRICKET_DB", "cricket.db"), timeout=120)
    d = load(con)
    os.makedirs("exports", exist_ok=True)
    pd.set_option("display.width", 200)
    t = totals(d)
    print("First-innings totals and phase run rates (full 20-over matches)\n", t.to_string(), "\n")
    m = first_totals(d)
    e = pd.concat({"all controls": did(m, "total", ["T20 Blast", "CPL"]),
                   "IPL vs Blast and CPL only": did(m[m.league != "SMAT"], "total", ["T20 Blast", "CPL"])})
    print("Rule effect on first-innings total, runs (season and ground fixed effects)\n", e.to_string(), "\n")
    s = by_season(m)
    print("IPL first-innings total vs Blast and CPL, change from 2019, runs\n", s.to_string(), "\n")
    w = wicket_cost(d)
    print("Runs lost per wicket down at 10 overs\n", w.to_string(index=False), "\n")
    a, side = allrounder_overs(d)
    print("Overs bowled by top-7 batters, % of the side's balls\n", a.to_string(), "\n")
    ae = did(side, "share", ["T20 Blast", "CPL"])
    print("Rule effect on top-7 batters' share of overs, percentage points (league and season fixed effects)\n", ae.to_string())
    print("\nIPL top 7: bowlers per side, their overs per side, overs per bowler\n", top7_split(d).to_string(), "\n")
    pl = allrounder_players(d)
    print("IPL all-rounders, overs per game in the top 7, before and after the rule\n", pl.to_string())
    pl.to_csv("exports/impact_player_allrounders.csv")
    rt, rc = roles(d)
    print("\nBatting by order, per 100 balls (primer metrics)\n", rt.to_string())
    print("\nChange 2019-22 to 2023-26: IPL, IPL minus Blast/CPL, IPL group minus IPL 4-7\n", rc.to_string())
    ld = leaders(d)
    print("\nIPL 2023-26 batters by WPA\n", ld.to_string())
    for name, x in {"roles": rt, "roles_change": rc, "wpa_leaders": ld, "totals": t, "effect": e, "by_season": s, "wicket_cost": w, "allrounder_overs": a, "allrounder_effect": ae}.items():
        x.to_csv(f"exports/impact_player_{name}.csv")


if __name__ == "__main__":
    main()
