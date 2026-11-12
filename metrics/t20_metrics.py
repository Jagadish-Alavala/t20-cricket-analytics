"""Contextual T20 metrics (men's senior T20) from the ball-by-ball commentary in cricket.db.

Follows Himanish Ganjoo, "T20 Metrics: A Primer" (https://hganjoo.github.io/t20basics/), the method
behind Big Hit Cricket's Impact:
  par       expected 1st-innings total per match (§2): nested shrinkage global -> league -> year -> ground
            (T20Is: global -> year -> host country)
  R(b,w,λ)  Duckworth-Lewis "Pro" run-production curve (§4), fit to our data (n0 too); λ per match from its par, λ >= 1.
            Rain-shortened matches use the balls each innings actually had, and a DLS-revised target (S1 x R2/R1 + 1)
  Impact    change in projected final total caused by a ball (§5; runs added; batter +, bowler -)
  Adjusted  Impact adjusted for the quality of the opposing batter / bowler, iterated to convergence (§5.3)
  RAA/WAA   runs / wickets above the xgboost expectation for (balls left, wickets, par/target, innings) (§3)
  WP, WPA   win probability P = r^n / (1 + r^n) from the DL score ratio r (§6), and its change per ball
  Leverage  WP if the ball were a six minus WP if it were a wicket (§7)

Writes tables t20_par, t20_ball_metrics and the views t20_batting, t20_bowling to cricket.db.
The models are fit on complete matches; every match is scored. Wides are left out of the player views. T10 is left out.

  python t20_metrics.py
  SELECT * FROM t20_batting WHERE league='IPL' AND year=2026 AND balls>=100 ORDER BY impact DESC;
"""
import os, re, sqlite3, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq, least_squares, minimize, minimize_scalar
import xgboost as xgb

DB, PYTH = os.environ.get("CRICKET_DB", "cricket.db"), 6  # the primer's best-fit Pythagorean power (n0 is fit to our data: the primer got 1.04)

# host country of each T20I city (primer §2.1 nests T20Is by country, not ground)
T20I_COUNTRY = {c: k for k, cs in {
    "India": "Ahmedabad Bengaluru Chennai Cuttack Dehradun Dharamshala Greater_Noida Guwahati Gwalior Hyderabad Indore Jaipur "
             "Kolkata Lucknow Mohali,_Chandigarh Mumbai Nagpur Naya_Raipur New_Chandigarh New_Delhi Pune Rajkot Ranchi Trivandrum Vizag",
    "Australia": "Adelaide Brisbane Cairns Canberra Carrara Darwin Geelong Hobart Melbourne Perth Sydney",
    "New Zealand": "Auckland Christchurch Dunedin Hamilton Napier Nelson Queenstown Tauranga Wellington",
    "South Africa": "Cape_Town Centurion Durban East_London Gqeberha Johannesburg Paarl",
    "England": "Birmingham Bristol Cardiff Chester-le-Street Leeds London Manchester Nottingham Southampton",
    "West Indies": "Basseterre Bridgetown Coolidge Gros_Islet Kingston Kingstown North_Sound Providence Roseau St_George's Tarouba",
    "USA": "Grand_Prairie Houston Lauderhill New_York", "UAE": "Abu_Dhabi Dubai Sharjah The_Sevens_Stadium,_Dubai",
    "Pakistan": "Karachi Lahore Rawalpindi", "Bangladesh": "Chattogram Cox's_Bazar Mirpur Sylhet",
    "Sri Lanka": "Colombo Dambulla Kandy", "Zimbabwe": "Bulawayo Harare", "Ireland": "Belfast Derry Dublin",
    "Scotland": "Edinburgh", "Netherlands": "Rotterdam Voorburg", "Oman": "Al_Amarat", "Namibia": "Windhoek",
    "Japan": "Nisshin Sano", "China": "Zhejiang_University_of_Technology_Cricket_Field,_Hangzhou",
}.items() for c in (x.replace("_", " ") for x in cs.split())}  # "_" stands for a space in a name


def league(level, comp):
    if level == "International":
        return "T20I"
    for pat, name in [("Indian Premier", "IPL"), ("Mushtaq", "SMAT"), ("Blast|England Domestic T20", "T20 Blast"),
                      ("BBL|Big Bash", "BBL"), ("Caribbean|CPL", "CPL"), ("Major League", "MLC"),
                      ("International League", "ILT20"), ("Tamil Nadu", "TNPL"), ("Global Super", "GSL")]:
        if re.search(pat, comp):
            return name
    return "Other T20"


def load(con):
    d = pd.read_sql("""
        SELECT c.match_id, c.innings, c.seq, c.batting_team, c.runs, c.is_wicket, c.scoring, c.bowler,
               c.batter, c.batter_id, c.out_player_id, c.commentary, i.level,
               json_extract(c.raw, '$.runsScored') AS ca_bat, i.competition, i.year, i.venue, i.city, i.start_date, i.result,
               json_extract(c.raw, '$.isWide') AS ca_wide, json_extract(c.raw, '$.isNoBall') AS ca_nb,
               COALESCE(bb.bowler_id, json_extract(c.raw, '$.bowlerPlayerId')) AS bowler_id,
               COALESCE(pb.known_as, pc.known_as) AS bowler_known_as, pt.known_as AS batter_known_as
        FROM commentary c JOIN match_info i USING (match_id)
        LEFT JOIN ball_bowler bb USING (match_id, innings, seq)
        LEFT JOIN players pb ON pb.source = 'B' AND pb.player_id = bb.bowler_id
        LEFT JOIN players pc ON pc.source = 'CS' AND pc.player_id = json_extract(c.raw, '$.bowlerPlayerId') AND c.match_id LIKE 'CS:%'
        LEFT JOIN players pt ON pt.source = CASE WHEN c.match_id LIKE 'CS:%' THEN 'CS' ELSE 'B' END
                            AND pt.player_id = c.batter_id AND c.match_id NOT LIKE 'CA:%'
        WHERE i.format = 'T20' AND i.gender = 'Men' AND i.age_group = 'Senior' AND c.innings IN (1, 2)
          AND c.match_id NOT IN (SELECT match_id FROM duplicates)
          AND i.competition NOT LIKE '%T10%'
        ORDER BY c.match_id, c.innings, c.seq""", con)
    ca = d.match_id.str.startswith(("CA:", "CS:"))  # Cricsheet balls are stored in the CA shape
    sc = d.scoring.fillna("")
    d["wide"] = np.where(ca, d.ca_wide == 1, sc.str.contains("ʷᵈ"))
    d["legal"] = ~d.wide & ~np.where(ca, d.ca_nb == 1, sc.str.contains("ⁿᵇ"))
    d["run_out"] = d.is_wicket.astype(bool) & d.commentary.fillna("").str.contains("run out", case=False)
    # runs off the bat. bcci.tv scoring: "4", "1ˡᵇ" leg bye, "5ⁿᵇ" no ball + 4 off the bat, "2ⁿ⁺ᵇ" no ball + byes
    n = sc.str.extract(r"^(\d+)")[0].astype(float).fillna(0)
    bcci_bat = np.where(sc.str.contains("ⁿᵇ"), n - 1, np.where(sc.str.contains("[ʷˡᵇⁿ]"), 0, n))
    d["bat_runs"] = np.where(ca, d.ca_bat.fillna(0), bcci_bat).astype(int)
    d["league"] = [league(l, c) for l, c in zip(d.level, d.competition)]
    host = d.city.where(d.city.fillna("") != "", d.venue).map(T20I_COUNTRY)
    d["ground"] = np.where(d.league == "T20I", host.fillna(d.venue), d.venue)  # shrinkage's lowest level
    # Players are keyed by id; bcci.tv and CA ids are separate number spaces, so prefix them. Names: bcci.tv
    # scorecard name where known, else the player's most common name in the commentary.
    src = np.where(d.match_id.str.startswith("CS:"), "CS:", np.where(ca, "CA:", "B:"))
    d["batter_key"] = src + d.batter_id.astype("Int64").astype(str)
    d["batter_name"] = d.batter_known_as.fillna(d.groupby("batter_key").batter.transform(mode))
    d["out_key"] = np.where(d.out_player_id.notna(), src + d.out_player_id.astype("Int64").astype(str), None)  # who was out (run outs: maybe the non-striker)
    # a bcci.tv ball whose bowler couldn't be matched to the scorecard (~1%, or no scorecard yet) keeps its text name
    d["bowler_key"] = np.where(d.bowler_id.notna(), src + d.bowler_id.astype("Int64").astype(str), "?:" + d.bowler.fillna(""))
    d["bowler"] = d.bowler_known_as.fillna(d.groupby("bowler_key").bowler.transform(mode))
    # the same player in both sources (Buttler in the IPL and the T20 Blast): merge CA ids into the bcci.tv id
    # when the full name is the same and belongs to only one player on each side
    # ponytail: exact names only; "JC Buttler" vs "Jos Buttler" spellings stay split
    names = pd.concat([d[["batter_key", "batter_name"]].set_axis(["key", "name"], axis=1),
                       d[["bowler_key", "bowler"]].set_axis(["key", "name"], axis=1)]).dropna().drop_duplicates()
    names = names[~names.key.str.startswith("?:")]
    names["src"] = names.key.str[:2]
    once = names.groupby(["src", "name"]).key.transform("nunique") == 1
    side = {s: g.set_index("name").key for s, g in names[once].groupby("src")}
    if "B:" in side and "CA" in side:
        both = side["CA"].index.intersection(side["B:"].index)
        to_b = dict(zip(side["CA"][both], side["B:"][both]))
        for col in ("batter_key", "bowler_key", "out_key"):
            d[col] = d[col].replace(to_b)
        d["batter_name"] = d.groupby("batter_key").batter_name.transform("first")
        d["bowler"] = d.groupby("bowler_key").bowler.transform("first")
    # Cricsheet (IPL 2008-11): link each player to the one IPL player on bcci.tv with the same scorecard name
    # ("CH Gayle", both sources use it), else the same full name (Cricsheet's from Wikidata), else the same initial
    # and surname when one first name starts with the other ("Anirudha"/"Anirudh"; not Yashpal/Yuvraj Singh),
    # so careers run across 2012
    # ponytail: a name shared by two bcci.tv IPL players at every step stays split
    ipl_keys = set(d.loc[d.league == "IPL", "batter_key"]) | set(d.loc[d.league == "IPL", "bowler_key"])
    pl = pd.read_sql("SELECT source, player_id, known_as, name FROM players WHERE source IN ('B', 'CS')", con)
    pl["key"] = pl.source.map({"B": "B:", "CS": "CS:"}) + pl.player_id.astype(str)
    ipl_b = pl[(pl.source == "B") & pl.key.isin(ipl_keys)]
    short = lambda n: (n.split()[0][0] + " " + n.split()[-1]).lower() if isinstance(n, str) and n.split() else ""
    unique = lambda col: ipl_b.assign(c=col).drop_duplicates("c", keep=False).set_index("c").key
    by_card, by_full, by_init = unique(ipl_b.name), unique(ipl_b.known_as), unique(ipl_b.known_as.map(short))
    cs = pl[(pl.source == "CS") & pl.key.isin(ipl_keys)]
    first = lambda n: n.split()[0].rstrip(".").lower() if isinstance(n, str) and n.split() else ""
    b_name = ipl_b.set_index("key").known_as

    def init_link(f):
        k = by_init.get(short(f))
        a, b = first(f), first(b_name.get(k))
        return k if k and len(min(a, b, key=len)) >= 3 and (a.startswith(b) or b.startswith(a)) else None
    to_b = {k: by_card.get(c, by_full.get(f, init_link(f))) for k, c, f in zip(cs.key, cs.name, cs.known_as)}
    to_b = {k: v for k, v in to_b.items() if v is not None}
    for col in ("batter_key", "bowler_key", "out_key"):
        d[col] = d[col].replace(to_b)
    ours = ~d.match_id.str.startswith("CS:")  # once linked, show the bcci.tv / CA name
    for key, name in (("batter_key", "batter_name"), ("bowler_key", "bowler")):
        d[name] = d[key].map(d[ours].groupby(key)[name].first()).fillna(d[name])
    return d


def state(d):
    g = d.groupby(["match_id", "innings"])
    d["balls_before"] = g.legal.cumsum() - d.legal  # legal balls bowled before this one
    d["wkts_before"] = g.is_wicket.cumsum() - d.is_wicket
    d["score_before"] = g.runs.cumsum() - d.runs
    return d


def mode(s):
    return s.mode().iat[0] if s.notna().any() else None


def complete_matches(d):
    """Matches with a full-length 1st innings and a decided 2nd innings (no rain reduction, no tie)."""
    inn = d.groupby(["match_id", "innings"]).agg(runs=("runs", "sum"), balls=("legal", "sum"), wkts=("is_wicket", "sum")).unstack()
    s1, s2 = inn["runs"][1], inn["runs"][2]
    full1 = (inn["balls"][1] == 120) | ((inn["wkts"][1] >= 10) & (inn["balls"][1] <= 120))
    done2 = (s2 > s1) | (inn["wkts"][2] >= 10) | (inn["balls"][2] == 120)
    ok = full1 & done2 & (inn["balls"][2] <= 120) & (s1 != s2)
    return pd.DataFrame({"first_total": s1, "won_batting_first": s1 > s2})[ok.fillna(False)]


def innings_balls(d):
    """Balls each innings was scheduled for (DLS resources): 120, unless the innings shows it was cut by rain.
    The feeds don't carry revised overs, so they are read from how innings ended; an interruption during an innings
    is treated as if the reduced overs were known from its start."""
    inn = d.groupby(["match_id", "innings"]).agg(balls=("legal", "sum"), wkts=("is_wicket", "sum")).unstack()
    res = d.groupby("match_id").result.first().reindex(inn.index).fillna("")
    over = lambda b: np.round(b / 6) * 6  # nearest over: a stray uncounted no-ball shouldn't add an over
    nr = res.str.contains("no result|abandon", case=False)
    dls = res.str.contains("DLS|D/L", case=False)
    chased = res.str.contains(r"by \d+ wicket", case=False)  # the chase got there: its own length is unknown
    b1, w1, b2, w2 = inn.balls[1], inn.wkts[1], inn.balls.get(2), inn.wkts.get(2)
    # innings 1 stopped short of 20 overs with wickets in hand: its overs were cut
    l1 = pd.Series(np.where((b1 < 120) & (w1 < 10) & ~nr, over(b1), 120), index=inn.index)
    # a chase that ended short with wickets in hand without getting there ran out of overs: revised by DLS, or cut like
    # the first innings. Neither (a full first innings, no DLS) means the commentary is missing balls: 20 overs.
    short2 = (b2 < 120) & (w2 < 10) & ~nr & ~chased & (dls | (l1 < 120))
    l2 = pd.Series(np.where(short2, over(b2), l1), index=inn.index)
    return l1, l2


def nested_par(m):
    """Primer §2.1: shrink each level's mean toward its parent's shrunk mean, w = n / (n + σ²within/σ²between)."""
    levels = ["league", "year", "ground"]
    m = m.copy()
    m["par"] = m.first_total.mean()
    for i in range(len(levels)):
        keys, parent = levels[: i + 1], levels[:i]
        sub = m.groupby(keys).first_total.agg(mean="mean", n="count").reset_index()
        # σ²between: spread of the sub-bucket means inside each parent; σ²within: spread of scores inside the parent
        sub["between"] = sub.groupby(parent)["mean"].transform("var") if parent else sub["mean"].var()
        within = m.groupby(parent).first_total.transform("var") if parent else m.first_total.var()
        s = m.join(sub.set_index(keys), on=keys)
        w = (s.n / (s.n + within / s.between)).where(s.between > 0).fillna(0)  # no spread / no bucket: keep parent
        m["par"] = w * s["mean"].fillna(0) + (1 - w) * m.par
    return m


# --- Duckworth-Lewis run production: R_std(b, w) = R0 F(w) [1 - exp(-b / β(w))] ---------------------
def dl_parts(p, w):
    r0, f1, f2, f3, b0, b1, b2, b3 = p[:8]
    F = 1 + f1 * w + f2 * w**2 + f3 * w**3
    beta = b0 * (1 + b1 * w + b2 * w**2 + b3 * w**3)
    return r0, np.clip(F, 0, None), np.clip(beta, 1, None)


def R(p, b, w, lam=1.0):
    """DL Pro (primer §4.2): R0 F λ^(n_w+1) [1 - exp(-b / (β λ^n_w))], n_w = n0 F; 0 when all out or no balls.
    p = (R0, F and β coefficients, n0)."""
    w = np.asarray(w, float)
    r0, F, beta = dl_parts(p, np.minimum(w, 9))
    nw = p[8] * F
    out = r0 * F * lam ** (nw + 1) * (1 - np.exp(-np.asarray(b, float) / (beta * lam**nw)))
    return np.where((w >= 10) | (np.asarray(b) <= 0), 0.0, out)


def fit_dl(d1):
    """Fit R_std to mean runs-still-to-come by (balls left, wickets down) in full first innings."""
    t = d1[d1.legal].assign(b=lambda x: 120 - x.balls_before, rest=lambda x: x.first_total - x.score_before)
    cell = t.groupby(["b", "wkts_before"]).rest.agg(["mean", "count"]).reset_index()
    cell = cell[cell["count"] >= 20]
    res = lambda p: (R([*p, 0], cell.b, cell.wkts_before) - cell["mean"]) * np.sqrt(cell["count"])
    p = [*least_squares(res, x0=[170, -0.05, -0.005, 0, 140, -0.05, 0, 0]).x, 0.0]
    p[8] = fit_n0(t, p)
    return p


def solve_lam(p, total, n0):
    """λ with R_pro(120, 0, λ) = total for an array of totals, by bisection; 1 for an average or lower total."""
    q, lo, hi = [*p[:8], n0], np.ones(len(total)), np.full(len(total), 3.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        big = R(q, 120, 0, mid) > total
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return lo


def fit_n0(t, p):
    """§4.2 n0, the front-loading of high-scoring innings: fit on first innings that beat R_std(120, 0), each curve
    calibrated to its own total (§4.3), to the runs still to come from every ball."""
    hi = t[t.first_total > float(R(p, 120, 0))]
    totals = hi.groupby("match_id").first_total.first()

    def sse(n0):
        lam = hi.match_id.map(pd.Series(solve_lam(p, totals.values, n0), index=totals.index))
        return np.sum((R([*p[:8], n0], hi.b, hi.wkts_before, lam) - hi.rest) ** 2)
    return minimize_scalar(sse, bounds=(0, 4), method="bounded").x


def shrunk_mean(v, key):
    """Per-player mean of v, shrunk toward 0 (the average player) with the §2.1 weight n / (n + σ²within/σ²between)."""
    g = v.groupby(key).agg(["sum", "count", "mean"])
    big = g[g["count"] >= 100]["mean"]
    k = v.var() / max(big.var() - v.var() / g["count"][big.index].mean(), 1e-9)  # spread of true means, noise removed
    return g["sum"] / (g["count"] + k)


def adjusted_impact(d, tol=1e-5):
    """§5.3: I_bat(t+1) = Impact + Ī_bowl(t), I_bowl(t+1) = -Impact + Ī_bat(t), player means shrunk; iterate to convergence.
    Batting against a bowler who usually concedes little earns credit, and vice versa."""
    bat = pd.Series(0.0, index=d.batter_key.unique())
    bowl = pd.Series(0.0, index=d.bowler_key.unique())
    for _ in range(100):
        adj_bat = d.impact + d.bowler_key.map(bowl).values
        adj_bowl = -d.bowl_impact + d.batter_key.map(bat).values
        # quality = shrunk mean relative to the average player, so the adjustment only moves players against each other
        nbat, nbowl = shrunk_mean(adj_bat - adj_bat.mean(), d.batter_key), shrunk_mean(adj_bowl - adj_bowl.mean(), d.bowler_key)
        done = max((nbat - bat).abs().max(), (nbowl - bowl).abs().max()) < tol
        bat, bowl = nbat, nbowl
        if done:
            break
    return adj_bat, adj_bowl


def main():
    con = sqlite3.connect(DB, timeout=120)
    d = load(con)
    # Models (par, DL curve, baselines, win probability) are fit on complete matches only; every match is scored.
    # A match left out of the fits (rain, tie, no result, a commentary innings off by a ball) takes the par of its
    # league/year/ground bucket: nested_par skips its missing first_total.
    clean = complete_matches(d)  # winner and completeness from the real totals, wides included
    # rain-shortened matches are scored with the overs they actually had: published by the source (innings_limits)
    # where we have it, else read from how the innings ended
    lim = pd.read_sql("SELECT * FROM innings_limits", con).set_index(["match_id", "innings"])
    l1, l2 = innings_balls(d)
    for n, l in ((1, l1), (2, l2)):
        pub = lim.xs(n, level="innings").ball_limit if len(lim) else pd.Series(dtype=float)
        l.update(pub[pub.index.isin(l.index)])
    # Wides (primer: "removed from the dataset"): their runs stay in the score, par and DL curve, since they are part
    # of the total, but wide deliveries are not scored for anyone and are left out of the baselines
    first = d[d.innings == 1].groupby("match_id").runs.sum()
    d = state(d)
    d = d[~d.wide].copy()
    meta = d.drop_duplicates("match_id").set_index("match_id")[["league", "year", "venue", "ground", "start_date"]]
    m = nested_par(meta.join(clean))
    m["clean"] = m.index.isin(clean.index)
    m["first_total"] = first
    d = d.join(m[["first_total", "won_batting_first", "par", "clean"]], on="match_id")
    print(f"{len(m)} men's T20 matches ({m.clean.sum()} complete, used for the fits), {len(d)} balls")

    p = fit_dl(d[(d.innings == 1) & d.clean])
    print(f"DL fit: R(120,0) = {float(R(p, 120, 0)):.1f} runs, n0 = {p[8]:.2f} (primer 1.04)")
    short = (l1 < 120) | (l2 < 120)
    print(f"{short.sum()} rain-shortened matches scored with their reduced overs "
          f"({short[short].index.isin(lim.index.get_level_values(0)).sum()} with overs published by the source)")
    m["balls1"], m["balls2"] = l1, l2
    # §4.3: λ solves R_pro(120, 0, λ) = par; an average or low par keeps the standard curve (λ = 1)
    r_std = float(R(p, 120, 0))
    lam = {mid: brentq(lambda l: float(R(p, 120, 0, l)) - par, 1.0, 2.0) if par > r_std else 1.0 for mid, par in m.par.items()}
    m["lam"] = pd.Series(lam)
    d["lam"] = d.match_id.map(m.lam)

    d["sched"] = np.where(d.innings == 1, d.match_id.map(m.balls1), d.match_id.map(m.balls2))
    # overs cut during an innings (bcci.tv stoppage records): the innings' starting limit applies until the cut
    cut = d.join(lim[["start_limit", "cut_at"]], on=["match_id", "innings"])
    d["sched"] = np.where(cut.cut_at.fillna(0) > d.balls_before, cut.start_limit, d.sched)
    b0, w0 = np.maximum(d.sched - d.balls_before, 0), d.wkts_before
    b1, w1 = b0 - d.legal, w0 + d.is_wicket
    r_before, r_after = R(p, b0, w0, d.lam), R(p, b1, w1, d.lam)
    d["impact"] = d.runs + r_after - r_before  # batting side's projected final total moves by this much
    # a run out isn't the bowler's wicket: they are charged for the ball as if nobody was out (matches the primer's table)
    d["bowl_impact"] = np.where(d.run_out, d.runs + R(p, b1, w0, d.lam) - r_before, d.impact)
    # not centred, as in the primer (§5.4: an innings' Impact sums to its runs minus R(120, 0, λ)), so a season's
    # total isn't zero: wides' runs belong to no batter, and a high-scoring season sits above its par

    # RAA/WAA baselines (§3.1): xgboost as a smooth averaging table, deliberately fit to all the data, on the primer's
    # four features. Conditions (league, year, ground) enter through the par score.
    # DLS target when the chase had fewer resources than the first innings: S1 x R(chase) / R(first innings) + 1
    r1, r2 = (R(p, d.match_id.map(m[c]), 0, d.lam) for c in ("balls1", "balls2"))
    d["target2"] = np.where(r2 < r1, np.floor(d.first_total * r2 / r1) + 1, d.first_total + 1)
    official = d.join(lim.target.rename("official"), on=["match_id", "innings"]).official  # published (DLS) target
    d["target2"] = np.where(official.fillna(0) > 0, official, d.target2)
    d["target"] = np.where(d.innings == 1, d.par, d.target2)
    X = np.c_[b0, w0, d.target, d.innings]
    # batter RAA is on runs off the bat; bowler RAA (raa_runs) on all runs conceded
    for col, y in (("exp_bat", d.bat_runs), ("exp_runs", d.runs), ("exp_wkts", d.is_wicket)):
        dm = xgb.DMatrix(X[d.clean], label=y[d.clean])
        d[col] = xgb.train({"max_depth": 6, "eta": 0.1}, dm, num_boost_round=300).predict(xgb.DMatrix(X))
        # not centred, as in the primer: one pooled baseline, so a league-season can sit above or below zero
        # (IPL 2026: about +7 RAA per 100 balls)
    d["raa"], d["raa_runs"], d["waa"] = d.bat_runs - d.exp_bat, d.runs - d.exp_runs, d.exp_wkts - d.is_wicket

    # Win probability for the batting side from the DL score ratio r (primer §6.1)
    full = r1  # what the first innings' resources were worth

    def log_r(score, b, w):  # 1st innings: projected total / R(120,0,λ); 2nd: runs still to come / runs required
        if_first = (score + R(p, b, w, d.lam)) / full
        need = d.target2 - score
        r = np.where(d.innings == 1, if_first, np.divide(R(p, b, w, d.lam), need, out=np.full(len(d), np.inf), where=need > 0))
        return np.log(np.clip(np.nan_to_num(r, posinf=1e6), 1e-6, 1e6))

    # §6.1 Pythagorean P = r^n / (1 + r^n), i.e. logistic in log r with no offset; one n fit on both innings
    wbf = d.won_batting_first.fillna(False).astype(bool)
    won = np.where(d.innings == 1, wbf, ~wbf).astype(float)
    lr0 = log_r(d.score_before, b0, w0)
    x, y = lr0[d.clean], won[d.clean]
    n = minimize(lambda c: np.sum(np.logaddexp(0, -(2 * y - 1) * c[0] * x)), [PYTH]).x[0]
    print(f"win probability: P = r^n / (1 + r^n), n = {n:.2f}  (primer: 6)")
    wp = lambda score, b, w: 1 / (1 + np.exp(-n * log_r(score, b, w)))

    d["wp_before"] = wp(d.score_before, b0, w0)
    d["wp_after"] = wp(d.score_before + d.runs, b1, w1)
    d["wpa"] = d.wp_after - d.wp_before
    d["leverage"] = wp(d.score_before + 6, b0 - 1, w0) - wp(d.score_before, b0 - 1, w0 + 1)
    d["adj_bat"], d["adj_bowl"] = adjusted_impact(d)

    cols = ["match_id", "innings", "seq", "league", "year", "start_date", "venue", "batting_team", "batter_key", "batter_name",
            "bowler_key", "bowler", "runs", "bat_runs", "is_wicket", "out_key", "run_out", "wide", "legal", "balls_before", "wkts_before", "score_before", "par",
            "lam", "impact", "bowl_impact", "adj_bat", "adj_bowl", "exp_bat", "exp_runs", "exp_wkts", "raa", "raa_runs", "waa", "wp_before", "wp_after", "wpa", "leverage"]
    d[cols].to_sql("t20_ball_metrics", con, if_exists="replace", index=False)
    m.reset_index().to_sql("t20_par", con, if_exists="replace", index=False)
    con.executescript("""
    CREATE INDEX IF NOT EXISTS t20_bm_batter ON t20_ball_metrics(batter_key);
    DROP VIEW IF EXISTS t20_batting; DROP VIEW IF EXISTS t20_bowling;
    -- wides are left out of both, as in the primer (a wide is not faced); bowler signs are flipped
    -- batting WAA = dismissals the average batter would suffer on the balls faced minus the batter's own dismissals,
    -- so a non-striker run out counts against the batter who was out, not the striker
    CREATE VIEW t20_batting AS
      WITH outs AS (SELECT league, year, COALESCE(out_key, batter_key) AS batter_key, COUNT(*) AS outs
                    FROM t20_ball_metrics WHERE is_wicket GROUP BY 1, 2, 3),
      b AS (SELECT league, year, batter_key, batter_name AS batter, COUNT(DISTINCT match_id) AS inns, COUNT(*) AS balls,
                   SUM(bat_runs) AS runs, SUM(impact) AS impact, SUM(raa) AS raa, SUM(exp_wkts) AS exp_outs,
                   SUM(adj_bat) AS adj, SUM(wpa) AS wpa
            FROM t20_ball_metrics WHERE NOT wide GROUP BY league, year, batter_key)
      SELECT b.league, b.year, b.batter_key, batter, inns, balls, runs, ROUND(100.0 * runs / balls, 2) AS sr,
             COALESCE(outs, 0) AS outs, ROUND(impact, 2) AS impact, ROUND(100 * impact / balls, 2) AS impact_rate,
             ROUND(impact / inns, 2) AS impact_inns, ROUND(raa, 2) AS raa, ROUND(exp_outs - COALESCE(outs, 0), 2) AS waa,
             ROUND(100 * raa / balls, 2) AS raa_100, ROUND(100 * (exp_outs - COALESCE(outs, 0)) / balls, 2) AS waa_100,
             ROUND(raa / inns, 2) AS raa_inns, ROUND((exp_outs - COALESCE(outs, 0)) / inns, 2) AS waa_inns,
             ROUND(100 * adj / balls, 2) AS adj_impact_rate, ROUND(wpa, 3) AS wpa, ROUND(100 * wpa / balls, 2) AS wpa_100
      FROM b LEFT JOIN outs USING (league, year, batter_key);
    CREATE VIEW t20_bowling AS
      SELECT league, year, bowler_key, MAX(bowler) AS bowler, COUNT(DISTINCT match_id) AS inns, COUNT(*) AS balls, SUM(runs) AS runs_conceded,
             ROUND(6.0 * SUM(runs) / COUNT(*), 2) AS econ, SUM(is_wicket AND NOT run_out) AS wkts,
             ROUND(-SUM(bowl_impact), 2) AS impact, ROUND(-100 * SUM(bowl_impact) / COUNT(*), 2) AS impact_rate,
             ROUND(-SUM(bowl_impact) / COUNT(DISTINCT match_id), 2) AS impact_inns,
             ROUND(-SUM(raa_runs), 2) AS raa, ROUND(-SUM(waa), 2) AS waa,
             ROUND(-100 * SUM(raa_runs) / COUNT(*), 2) AS raa_100, ROUND(-100 * SUM(waa) / COUNT(*), 2) AS waa_100,
             ROUND(-SUM(raa_runs) / COUNT(DISTINCT match_id), 2) AS raa_inns, ROUND(-SUM(waa) / COUNT(DISTINCT match_id), 2) AS waa_inns,
             ROUND(100 * SUM(adj_bowl) / COUNT(*), 2) AS adj_impact_rate,
             ROUND(-SUM(wpa), 3) AS wpa, ROUND(-100 * SUM(wpa) / COUNT(*), 2) AS wpa_100
      FROM t20_ball_metrics WHERE NOT wide GROUP BY league, year, bowler_key;
    """)
    print("wrote t20_par, t20_ball_metrics, views t20_batting / t20_bowling")


if __name__ == "__main__":
    main()
