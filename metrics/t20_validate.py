"""Check t20_metrics.py against "T20 Metrics: A Primer" (Ganjoo): the worked examples, the published IPL tables,
and the model checks of §5.4-5.6, §6.2 and §7. Run after t20_metrics.py.

  python t20_validate.py
"""
import sqlite3
import numpy as np, pandas as pd
from scipy.optimize import brentq, minimize
import t20_metrics as tm

con = sqlite3.connect(tm.DB, timeout=120)
d = pd.read_sql("SELECT * FROM t20_ball_metrics", con)
d["legal"] = d.legal.astype(bool)
par = pd.read_sql("SELECT * FROM t20_par", con).set_index("match_id")
ALIAS = {"Mitchell Marsh": "Mitch Marsh", "Rasikh Salam": "Rasikh Dar", "Varun Chakravarthy": "Varun Chakaravarthy"}


def auc(y, s):
    r = pd.Series(s).rank().values
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic(x, y):
    """y ~ sigmoid(a + b x); returns (a, b)."""
    return minimize(lambda c: np.mean(np.logaddexp(0, -(2 * y - 1) * (c[0] + c[1] * x))), [0.0, 0.0]).x


def show(title, ours, primer, cols):
    print(f"\n== {title}")
    ours = ours.rename(index={v: k for k, v in ALIAS.items()})
    t = primer.join(ours[cols].add_suffix(" ours"))
    print(t[[c for pair in zip(cols, [c + " ours" for c in cols]) for c in pair]].round(2).to_string())


def table(src):
    rows = [r.split("|") for r in src.strip().splitlines()]
    return pd.DataFrame([[float(x) for x in r[1:]] for r in rows], index=[r[0] for r in rows])


# --- §4/§5.2 worked example: 60-2 after 6 overs, par 180 -------------------------------------------------
p = tm.fit_dl(d.assign(first_total=d.match_id.map(par.first_total))[(d.innings == 1) & d.match_id.map(par.clean).astype(bool)])
lam = max(1.0, brentq(lambda l: float(tm.R(p, 120, 0, l)) - 180, 0.5, 2))
before = 60 + float(tm.R(p, 84, 2, lam))
print(f"== §5.2 example (60-2 after 6 overs, par 180, λ={lam:.3f})")
print(f"projected before {before:.2f} (primer 183.57); four {4 + float(tm.R(p, 83, 2, lam)) + 60 - before:+.2f} (primer +3.08); "
      f"wicket {60 + float(tm.R(p, 83, 3, lam)) - before:+.2f} (primer -11.8)")

# --- Published tables (IPL) -----------------------------------------------------------------------------------
ipl = d[d.league == "IPL"]
bat = lambda x: x.groupby("batter_name").agg(Inns=("match_id", "nunique"), Balls=("runs", "size"), Runs=("bat_runs", "sum"),
                                             Impact=("impact", "sum"), RAA=("raa", "sum"), WAA=("waa", "sum"), WPA=("wpa", "sum"))
b = bat(ipl[ipl.year.between(2023, 2026)])
b["SR"] = 100 * b.Runs / b.Balls
show("Table 1: IPL 2023-26 batters by RAA", b, table("""
Abhishek Sharma|51|1624|191.74|309.41|-8.66
Vaibhav Sooryavanshi|18|752|221.83|233.02|-2.62
Suryakumar Yadav|53|1861|169.03|232.76|10.31
Yashasvi Jaiswal|54|1932|159.80|209.45|7.11
Phil Salt|40|1258|174.48|209.43|-3.97
Heinrich Klaasen|50|1908|169.00|204.60|19.50
Nicholas Pooran|53|1564|173.97|185.81|10.16
Travis Head|38|1302|177.63|156.46|-5.26
Shreyas Iyer|40|1288|163.66|110.90|9.38
Sanju Samson|48|1580|154.15|90.32|12.23""").set_axis(["Inns", "Runs", "SR", "RAA", "WAA"], axis=1), ["Runs", "SR", "RAA", "WAA"])

b = bat(ipl[ipl.year == 2026])
b["SR"], b["Imp/Inns"] = 100 * b.Runs / b.Balls, b.Impact / b.Inns
b["RAA/100"], b["WAA/100"], b["RAA/inns"], b["WAA/inns"] = 100 * b.RAA / b.Balls, 100 * b.WAA / b.Balls, b.RAA / b.Inns, b.WAA / b.Inns
show("Table 4: IPL 2026 batters by Impact", b, table("""
Vaibhav Sooryavanshi|776|237.31|232.31|14.52|78.93|-0.61
Rajat Patidar|501|192.69|111.67|7.98|43.08|0.66
Virat Kohli|675|165.85|71.15|4.45|16.91|1.75
Ishan Kishan|602|182.42|68.79|4.59|24.86|0.29
KL Rahul|593|174.41|65.52|4.68|22.06|1.48
Mitchell Marsh|563|163.19|65.15|5.01|13.36|1.25
Abhishek Sharma|563|204.73|59.27|3.95|37.85|-0.68
Shubman Gill|732|163.03|49.64|3.10|8.14|1.85
Heinrich Klaasen|624|160.00|29.54|1.97|2.71|1.59
Sai Sudharsan|722|157.99|22.71|1.34|4.75|1.28""").set_axis(["Runs", "SR", "Impact", "Imp/Inns", "RAA/100", "WAA/100"], axis=1),
     ["Runs", "Impact", "RAA/100", "WAA/100"])
show("Table 7: IPL 2026 batters by WPA", b, table("""
Vaibhav Sooryavanshi|16|327|16.13|-0.12|17.05|1.71
Shreyas Iyer|13|295|2.81|0.55|5.38|1.49
Virat Kohli|16|407|4.30|0.45|7.52|1.00
Ishan Kishan|15|330|5.47|0.06|6.52|0.75
Heinrich Klaasen|15|390|0.70|0.41|5.29|0.73
Shubman Gill|16|449|2.29|0.52|7.36|0.70""").set_axis(["Inns", "Balls", "RAA/inns", "WAA/inns", "Imp/Inns", "WPA"], axis=1),
     ["Balls", "RAA/inns", "Imp/Inns", "WPA"])

w = ipl[ipl.year == 2026]
o = w.groupby("bowler").agg(Inns=("match_id", "nunique"), Balls=("runs", "size"), R=("runs", "sum"), Impact=("bowl_impact", "sum"),
                            RAA=("raa_runs", "sum"), WAA=("waa", "sum"), WPA=("wpa", "sum"))
o[["Impact", "RAA", "WAA", "WPA"]] *= -1
o["Econ"], o["Imp/Inns"], o["RAA/100"], o["WAA/100"] = 6 * o.R / o.Balls, o.Impact / o.Inns, 100 * o.RAA / o.Balls, 100 * o.WAA / o.Balls
show("Table 5: IPL 2026 bowlers by Impact", o, table("""
Bhuvneshwar Kumar|379|7.71|243.05|15.19|14.68|1.76
Jofra Archer|361|8.78|139.86|8.74|-4.14|1.31
Jason Holder|231|7.38|110.56|10.05|8.11|3.15
Kagiso Rabada|388|9.48|106.95|6.29|-18.04|3.00
Sunil Narine|306|6.65|105.43|8.11|35.92|-0.00
Prince Yadav|312|8.46|94.57|6.76|8.94|-0.29
Eshan Malinga|326|8.87|92.72|6.18|12.09|0.37
Lungi Ngidi|248|8.06|85.86|7.81|25.09|-0.45
Rashid Khan|341|8.82|59.96|3.53|-9.84|1.37
Jasprit Bumrah|301|7.67|36.71|2.82|31.13|-3.71""").set_axis(["Balls", "Econ", "Impact", "Imp/Inns", "RAA/100", "WAA/100"], axis=1),
     ["Balls", "Econ", "Impact", "RAA/100", "WAA/100"])
show("Table 8: IPL 2026 bowlers by WPA", o, table("""
Bhuvneshwar Kumar|16|379|1.70
Jofra Archer|16|361|1.22
Sunil Narine|13|306|0.93
Eshan Malinga|15|326|0.86
Jasprit Bumrah|13|301|0.58""").set_axis(["Inns", "Balls", "WPA"], axis=1), ["Balls", "WPA"])

# --- Table 6: T20 World Cup final 2024 (India v South Africa, Barbados, 29 June 2024) -------------------------------
f = d[(d.start_date.str[:10] == "2024-06-29") & (d.league == "T20I") & d.venue.str.contains("Kensington|Bridgetown", na=False)]
if len(f):
    print("\n== Table 6: 2024 T20 World Cup final, batting (primer: Axar +0.174 / +19.0, Kohli +0.077 / +14.7, Klaasen +0.481 / +22.1)")
    print(f.groupby(["innings", "batter_name"]).agg(WPA=("wpa", "sum"), Impact=("impact", "sum")).round(3).to_string())
    print("bowling (primer: Arshdeep +0.637 / +21.3, Bumrah +0.419 / +18.5, Maharaj +0.104 / +12.2)")
    g = f.groupby(["innings", "bowler"]).agg(WPA=("wpa", "sum"), Impact=("bowl_impact", "sum"))
    print((-g).round(3).to_string())
else:
    print("\n== Table 6: 2024 T20 World Cup final not in the database")

# --- §5.4 Impact and match results -------------------------------------------------------------------------------
clean = par[par.clean.astype(bool)]
inn = d[d.match_id.isin(clean.index)].groupby(["match_id", "innings"]).impact.sum().unstack().dropna()
delta, y = (inn[1] - inn[2]).values, clean.won_batting_first.reindex(inn.index).astype(float).values
k = minimize(lambda c: np.mean(np.logaddexp(0, -(2 * y - 1) * c[0] * delta)), [0.1]).x[0]
# the primer fits the curve to the win rate in bins of Δ: same fit on 5-run bins
bm = pd.DataFrame({"x": delta, "y": y}).groupby((delta // 5) * 5 + 2.5).agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))
bm = bm[bm.n >= 20]
kb = minimize(lambda c: np.sum(bm.n * (bm.y - 1 / (1 + np.exp(-c[0] * bm.x))) ** 2), [0.1]).x[0]
print(f"\n== §5.4 win rate vs Δ (team batting first Impact minus second): p = 1/(1+e^(-kΔ)), k = {k:.3f} (primer 0.25), "
      f"odds per extra run x{np.exp(k):.2f} (primer x1.28), {len(delta)} matches (primer 2754); fit to binned win rates k = {kb:.3f}")

# team strength = dataset-wide adjusted Impact per ball of each player, summed over the balls they faced / bowled
bat_q, bowl_q = d.groupby("batter_key").adj_bat.mean(), d.groupby("bowler_key").adj_bowl.mean()
c = d[d.match_id.isin(clean.index)].assign(bq=lambda x: x.batter_key.map(bat_q), oq=lambda x: x.bowler_key.map(bowl_q))
s = c.groupby(["match_id", "innings"]).agg(bat=("bq", "sum"), bowl=("oq", "sum")).unstack().dropna()
team1 = s["bat"][1] + s["bowl"][2]  # team batting first: its batting in innings 1, its bowling in innings 2
team2 = s["bat"][2] + s["bowl"][1]
x, y = (team1 - team2).values, clean.won_batting_first.reindex(s.index).astype(float).values
rng = np.random.default_rng(0)
tr = rng.random(len(x)) < 0.65
a, b_ = logistic(x[tr], y[tr])
pt = 1 / (1 + np.exp(-(a + b_ * x[~tr])))
print(f"   match winner from team strength difference: coef {b_:.3f} (primer 0.057), test AUC {auc(y[~tr], pt):.3f} (primer 0.725), "
      f"log-loss {logloss(y[~tr], pt):.3f} vs null {logloss(y[~tr], np.full((~tr).sum(), y[tr].mean())):.3f} (primer 0.61 vs 0.693)")
print("   (strengths come from the same matches, so this AUC is optimistic)")

# --- §5.5 boundary on a ball vs batter-bowler skill gap ---------------------------------------------------------------
ok = d.batter_key.map(bat_q).notna() & d.bowler_key.map(bowl_q).notna()
d = d[ok]
gap = (d.batter_key.map(bat_q) - d.bowler_key.map(bowl_q)).values
yb = d.bat_runs.isin([4, 6]).astype(float).values
bk = d.batter_key.unique()
test_bat = set(rng.choice(bk, int(len(bk) * 0.35), replace=False))
te = d.batter_key.isin(test_bat).values
a, b_ = logistic(gap[~te], yb[~te])
pb = 1 / (1 + np.exp(-(a + b_ * gap[te])))
print(f"\n== §5.5 boundary from batter-bowler adjusted Impact gap: AUC {auc(yb[te], pb):.3f} (primer 0.572), "
      f"log-loss {logloss(yb[te], pb):.3f} vs null {logloss(yb[te], np.full(te.sum(), yb[~te].mean())):.3f} (primer 0.492 vs 0.497)")

# --- §5.6 Impact rate vs average and strike rate -------------------------------------------------------------------
out = d[d.is_wicket.astype(bool)]
outs = out.groupby(["league", "year", out.out_key.fillna(out.batter_key)]).size()
bs = d.groupby(["league", "year", "batter_key"]).agg(balls=("runs", "size"), runs=("bat_runs", "sum"), imp=("impact", "sum"))
bs["outs"] = outs.reindex(bs.index).fillna(0)
bs = bs[(bs.balls >= 200) & (bs.outs > 0)]
X = np.c_[np.ones(len(bs)), 100 * bs.runs / bs.balls, bs.runs / bs.outs]
ir = 100 * bs.imp / bs.balls
coef, *_ = np.linalg.lstsq(X, ir, rcond=None)
r2 = 1 - ((ir - X @ coef) ** 2).sum() / ((ir - ir.mean()) ** 2).sum()
print(f"\n== §5.6 Impact rate = {coef[0]:.1f} + {coef[1]:.3f} SR + {coef[2]:.3f} Avg, R² {r2:.2f}, {len(bs)} batter seasons "
      f"(primer: -106.7 + 0.713 SR + 0.293 Avg, ~0.75, 741)")

# --- §6.1-6.2 win probability ------------------------------------------------------------------------------------
wc = d[d.match_id.isin(clean.index)].copy()
wbf = wc.match_id.map(clean.won_batting_first).astype(bool)
wc["won"] = np.where(wc.innings == 1, wbf, ~wbf).astype(float)
bins = np.clip((wc.wp_before * 10).astype(int), 0, 9)
ece = (wc.groupby(bins).apply(lambda g: abs(g.won.mean() - g.wp_before.mean()) * len(g)).sum()) / len(wc)
print(f"\n== §6.2 win probability: Brier {np.mean((wc.wp_before - wc.won) ** 2):.3f} (primer 0.182), ECE {ece:.4f} (primer 0.0177), "
      + ", ".join(f"innings {i} AUC {auc(g.won.values, g.wp_before.values):.3f}" for i, g in wc.groupby("innings")) + " (primer 0.703, 0.809)")

# --- §7 leverage by phase -------------------------------------------------------------------------------------------
ph = pd.cut(d.balls_before, [-1, 35, 89, 999], labels=["Powerplay", "Middle", "Death"])
print("\n== §7 leverage by innings and phase (primer: highest in the 1st-innings Powerplay, falling through the innings;"
      " 2nd innings more spread out, with high-leverage death overs)")
print(d.groupby(["innings", ph], observed=True).leverage.describe()[["mean", "25%", "50%", "75%"]].round(3).to_string())
