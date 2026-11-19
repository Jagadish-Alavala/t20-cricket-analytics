import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from lib import (balls, footer, tiles, rank_tiles, heatmap, wheel, LENGTHS, LINES, VARIATIONS,
                 EXPLORER_PHASES as PHASE_LBL)

DEF = {
    "RAA": "Runs saved against an average bowler in the same game states (over, wickets, ground par). Positive is good.",
    "WAA": "Wickets taken above what an average bowler would expect in the same states.",
    "Impact": "Runs prevented given the game state, counting the value of the wickets he took.",
    "WPA": "Win probability added for his side: how much his bowling moved the batting team's chance of winning down.",
}
FS_HELP = ("Share of balls where the batter played a false shot: edged, beaten, mistimed or hit on the pad. Recorded "
           "ball by ball only in Cricket Australia's data (T20 Blast, some T20Is); shows – elsewhere.")
MIN_FIELD = 60

st.title("🎳 Bowler explorer")
d = balls()
if d is None:
    st.info("This page needs the ball-by-ball table. Build it with `python metrics/ball_tags.py`, then reload.")
    st.stop()

# ---------------------------------------------------------------- filters
vol = d.groupby("bowler", observed=True).size()
names = sorted(vol[vol >= 300].index)
with st.sidebar.form("bowl_filters"):
    bow = st.selectbox("Bowler", names, index=names.index("Jasprit Bumrah") if "Jasprit Bumrah" in names else 0)
    lg = st.multiselect("League", sorted(d.league.cat.categories), placeholder="All leagues")
    y0, y1 = int(d.year.min()), int(d.year.max())
    yrs = st.slider("Years", y0, y1, (y0, y1))
    hand = st.pills("Batter", ["Right", "Left"], selection_mode="multi", default=["Right", "Left"],
                    format_func=lambda h: f"{h}-hander")
    var = st.pills("Delivery", VARIATIONS, selection_mode="multi", help="Variation named in the commentary")
    length = st.pills("Length", LENGTHS, selection_mode="multi")
    line = st.pills("Line", LINES, selection_mode="multi")
    side = st.pills("Over / round the wicket", ["Over", "Round"], selection_mode="multi",
                    help="Recorded by Cricket Australia, not in BCCI commentary")
    inns = st.pills("Innings", [1, 2], selection_mode="multi", default=[1, 2], format_func=lambda i: ("1st", "2nd")[i - 1])
    phase = st.pills("Phase", list(PHASE_LBL), selection_mode="multi", default=list(PHASE_LBL))
    venue = st.multiselect("Venue", sorted(d.venue.dropna().cat.categories), placeholder="Any venue")
    st.form_submit_button("Apply filters", type="primary", width="stretch")

base = d[d.year.between(*yrs) & (d.innings <= 2) & (d.league.isin(lg) if lg else True)]  # no super overs
f = base[base.innings.isin(inns or [1, 2]) & base.phase.isin(phase or list(PHASE_LBL))]
if hand and len(hand) < 2:
    f = f[f.bat_hand.isin(hand)]
for col, sel in (("variation", var), ("length", length), ("line", line), ("side", side), ("venue", venue)):
    if sel:
        f = f[f[col].isin(sel)]
me = f[f.bowler == bow]
if me.empty:
    st.warning(f"No balls for {bow} with these filters.")
    st.stop()
nw = me[me.wide == 0]  # metric rows: the season tables count balls without wides


def summary(x):
    legal, runs, w = int(x.legal.sum()), int(x.conceded.sum()), int(x.bowler_wkt.sum())
    lb = x[x.legal == 1]
    c = x.ctrl.dropna()
    c = c if len(c) >= 0.5 * len(x) else c.iloc[:0]
    return {"Overs": f"{legal // 6}.{legal % 6}", "Runs": runs, "Wickets": w, "Economy": 6 * runs / legal if legal else np.nan,
            "Average": runs / w if w else np.nan, "Strike rate": legal / w if w else np.nan,
            "Dot %": 100 * (lb.conceded == 0).mean() if len(lb) else np.nan,
            "Boundary %": 100 * lb.bat_runs.isin([4, 6]).mean() if len(lb) else np.nan,
            "Wides": int(x.wide.sum()), "False shot %": 100 * (1 - c.mean()) if len(c) else np.nan}


fmt = lambda v, p=1: v if isinstance(v, str) else "–" if pd.isna(v) else \
    f"{v:,.{p}f}" if isinstance(v, (float, np.floating)) else f"{v:,}"
pos = lambda v: "#4ade80" if v > 0 else "#f87171" if v < 0 else None
kind = me.kind.dropna().mode()
style = me["style"].dropna().mode()
chosen = [", ".join(lg) or "All leagues", f"{yrs[0]}–{yrs[1]}"] + [", ".join(map(str, s)) for s in
                                                                    (var, length, line, side) if s]
if hand and len(hand) < 2:
    chosen.append(f"vs {hand[0].lower()}-handers")
st.markdown(f"**{bow}** · {style.iloc[0] if len(style) else ''} · {' · '.join(chosen)} · **{len(me):,} balls in view**")

nd = {"Runs": 0, "Wickets": 0, "Wides": 0, "Economy": 2, "Average": 1, "Strike rate": 1}
tiles([(k, fmt(v, nd.get(k, 1)), "", "#a78bfa" if k in ("Economy", "Wickets") else None,
        FS_HELP if k == "False shot %" else None) for k, v in summary(me).items()], min_w=105)
tiles([(f"{k} per ball", f"{nw[c].sum() / max(len(nw), 1):+.3f}", f"{nw[c].sum():+.1f} total · {len(nw):,} balls",
        pos(nw[c].sum()), DEF[k]) for k, c in [("RAA", "b_raa"), ("WAA", "b_waa"), ("Impact", "b_impact"), ("WPA", "b_wpa")]],
      min_w=170)

fw = f[f.wide == 0]
agg = fw.groupby("bowler", observed=True).agg(n=("b_raa", "size"), raa=("b_raa", "mean"), waa=("b_waa", "mean"),
                                              impact=("b_impact", "mean"), wpa=("b_wpa", "mean"))
field = agg[agg.n >= MIN_FIELD]
if bow in field.index:
    with st.container(border=True):
        rank_tiles(field, bow, [("RAA", "raa"), ("WAA", "waa"), ("Impact", "impact"), ("WPA", "wpa")],
                   f"bowlers with {MIN_FIELD}+ balls")
else:
    st.caption(f"Rankings need {MIN_FIELD}+ balls under these filters.")


def bowl_table(x, col, order=None):
    x = x.assign(dot=((x.legal == 1) & (x.conceded == 0)).astype(int), raa_nw=x.b_raa.where(x.wide == 0))
    g = x.groupby(col, observed=True)
    t = pd.DataFrame({"Balls": g.size(), "Runs": g.conceded.sum(), "Wkts": g.bowler_wkt.sum(), "legal": g.legal.sum(),
                      "dots": g.dot.sum(), "RAA saved/ball": g.raa_nw.mean()})
    t = (t.reindex(order) if order else t.sort_values("Balls", ascending=False)).dropna(subset=["Balls"])
    t = t[t.Balls > 0]
    t["% of balls"] = 100 * t.Balls / t.Balls.sum()
    t["Economy"] = 6 * t.Runs / t.legal.replace(0, np.nan)
    t["Dot %"] = 100 * t.dots / t.legal.replace(0, np.nan)
    t["Balls/wkt"] = t.legal / t.Wkts.replace(0, np.nan)
    return t[["% of balls", "Economy", "Dot %", "Wkts", "Balls/wkt", "RAA saved/ball", "Balls"]]


NUM = {c: st.column_config.NumberColumn(format="%.1f") for c in ["% of balls", "Dot %", "Balls/wkt"]}
NUM["Economy"] = st.column_config.NumberColumn(format="%.2f")
NUM["RAA saved/ball"] = st.column_config.NumberColumn(format="%+.3f")

tab_p, tab_ll, tab_v, tab_w = st.tabs(["Profile", "Lines and lengths", "Variations", "Wickets and runs"])

# ---------------------------------------------------------------- profile (league and year filters only)
with tab_p:
    p = base[base.bowler == bow]
    pn = p[p.wide == 0]
    st.caption("Uses only the league and year filters, so it describes the bowler as a whole.")
    matches = p.groupby(["match_id", "innings"], observed=True).ngroups
    over = p.balls_before // 6 + 1
    tiles([("RAA saved per ball", f"{pn.b_raa.mean():+.3f}", f"{pn.b_raa.sum():+.1f} total · {len(pn):,} balls",
            pos(pn.b_raa.mean()), DEF["RAA"]),
           ("Economy", f"{6 * p.conceded.sum() / max(p.legal.sum(), 1):.2f}", f"{int(p.bowler_wkt.sum())} wickets", None, None),
           ("Overs per match", f"{p.legal.sum() / 6 / max(matches, 1):.1f}", f"{matches} innings", None, None),
           ("Usual first over", f"{p.groupby(['match_id', 'innings'], observed=True).balls_before.min().median() // 6 + 1:.0f}",
            "median", None, None)], min_w=150)

    a, b = st.columns(2)
    with a:
        st.markdown("**When he bowls**: share of his balls by over")
        u = (100 * over.value_counts(normalize=True)).sort_index().rename_axis("over").reset_index(name="pct")
        st.altair_chart(alt.Chart(u).mark_bar(color="#22c55e").encode(
            x=alt.X("over:O", title="Over"), y=alt.Y("pct:Q", title="% of balls"),
            tooltip=["over", alt.Tooltip("pct:Q", format=".1f")]), width="stretch")
    with b:
        st.markdown("**Economy by over** (overs with 30+ balls)")
        e = p.assign(over=over).groupby("over").agg(runs=("conceded", "sum"), legal=("legal", "sum"))
        e = e[e.legal >= 30].assign(econ=lambda x: 6 * x.runs / x.legal).reset_index()
        st.altair_chart(alt.Chart(e).mark_line(point=True, color="#f59e0b").encode(
            x=alt.X("over:O", title="Over"), y=alt.Y("econ:Q", title="Economy", scale=alt.Scale(zero=False)),
            tooltip=["over", alt.Tooltip("econ:Q", format=".2f"), "legal"]), width="stretch")

    st.markdown("**By phase**")
    ph = bowl_table(p, "phase", list(PHASE_LBL))
    ph.index = ph.index.map(PHASE_LBL)
    st.dataframe(ph, width="stretch", column_config=NUM)
    st.markdown("**Against right- and left-handers**")
    st.dataframe(bowl_table(p.dropna(subset=["bat_hand"]), "bat_hand"), width="stretch", column_config=NUM)

# ---------------------------------------------------------------- lines and lengths
with tab_ll:
    tagged = me.dropna(subset=["length", "line"])
    st.caption(f"{len(tagged):,} of {len(me):,} balls in view have both line and length recorded in the commentary.")
    if len(tagged):
        metric = st.radio("Colour by", ["% of his balls", "Economy", "Wicket %", "RAA saved per ball"], horizontal=True)
        g = tagged.groupby(["length", "line"], observed=True).agg(balls=("conceded", "size"), runs=("conceded", "sum"),
                                                                 legal=("legal", "sum"), wkts=("bowler_wkt", "sum"),
                                                                 raa=("b_raa", "mean")).reset_index()
        g["% of his balls"] = 100 * g.balls / g.balls.sum()
        g["Economy"] = 6 * g.runs / g.legal.replace(0, np.nan)
        g["Wicket %"] = 100 * g.wkts / g.legal.replace(0, np.nan)
        g["RAA saved per ball"] = g.raa
        if metric != "% of his balls":
            g = g[g.balls >= 5]
        cells = g.drop(columns=["runs", "legal", "raa"])
        if metric == "% of his balls":
            heatmap(cells, metric, ".1f", None, scheme="greens")
            st.caption("Where he lands it: darker means more of his deliveries.")
        else:
            mid = {"Economy": float(6 * tagged.conceded.sum() / max(tagged.legal.sum(), 1)), "Wicket %": float(
                100 * tagged.bowler_wkt.sum() / max(tagged.legal.sum(), 1)), "RAA saved per ball": 0.0}[metric]
            heatmap(cells, metric, ".2f" if metric != "Wicket %" else ".1f", mid, good_high=(metric == "Economy"))
            st.caption("Blue = good for the bowler, red = good for the batter. Cells with fewer than 5 balls are hidden.")
    st.markdown("**By length**")
    st.dataframe(bowl_table(me, "length", LENGTHS), width="stretch", column_config=NUM)
    st.markdown("**By line**")
    st.dataframe(bowl_table(me, "line", LINES), width="stretch", column_config=NUM)

# ---------------------------------------------------------------- variations and the shots they draw
with tab_v:
    v = me.dropna(subset=["variation"])
    st.caption(f"{len(v):,} of {len(me):,} balls in view name the delivery type.")
    if len(v):
        t = bowl_table(v, "variation")
        st.markdown("**His deliveries**")
        st.dataframe(t, width="stretch", column_config=NUM)
        pts = t.reset_index()
        st.altair_chart(alt.Chart(pts[pts.Balls >= 12]).mark_circle(opacity=.85, color="#38bdf8").encode(
            x=alt.X("% of balls:Q", title="% of his balls"), y=alt.Y("Economy:Q", scale=alt.Scale(zero=False)),
            size=alt.Size("Wkts:Q", legend=None),
            tooltip=["variation", alt.Tooltip("% of balls:Q", format=".1f"), alt.Tooltip("Economy:Q", format=".2f"),
                     "Wkts", alt.Tooltip("RAA saved/ball:Q", format="+.3f")])
                        + alt.Chart(pts[pts.Balls >= 12]).mark_text(dy=-12, color="#e5e7eb", fontSize=11).encode(
                            x="% of balls:Q", y="Economy:Q", text="variation:N"), width="stretch")
        st.caption("How often he uses each delivery against what it costs. Bigger circle = more wickets.")
    sh = me[(me.wide == 0)].dropna(subset=["shot"])
    if len(sh):
        st.markdown("**Shots batters play against him**")
        g = sh.groupby("shot", observed=True).agg(Balls=("bat_runs", "size"), Runs=("bat_runs", "sum"), Outs=("out", "sum"))
        g = g[g.Balls > 0].sort_values("Balls", ascending=False)
        g["% of balls"], g["SR"] = 100 * g.Balls / g.Balls.sum(), 100 * g.Runs / g.Balls
        st.dataframe(g[["% of balls", "SR", "Outs", "Balls"]], width="stretch",
                     column_config={"% of balls": st.column_config.NumberColumn(format="%.1f"),
                                    "SR": st.column_config.NumberColumn(format="%.1f")})

# ---------------------------------------------------------------- wickets and where the runs go
with tab_w:
    a, b = st.columns([1, 1])
    with a:
        st.markdown("**How he takes wickets**")
        dis = me[me.bowler_wkt == 1].dismissal.value_counts().rename_axis("How").reset_index(name="Wickets")
        dis = dis[dis.Wickets > 0]
        if len(dis):
            st.altair_chart(alt.Chart(dis).mark_bar(color="#38bdf8").encode(
                y=alt.Y("How:N", sort="-x", title=None), x="Wickets:Q", tooltip=["How", "Wickets"]), width="stretch")
        st.caption(f"{int(me.bowler_wkt.sum() - dis.Wickets.sum())} wickets without a recorded type.")
    with b:
        st.markdown("**Wickets by length**")
        wl = me[me.bowler_wkt == 1].length.value_counts().reindex(LENGTHS).fillna(0).rename_axis("Length").reset_index(name="Wickets")
        st.altair_chart(alt.Chart(wl).mark_bar(color="#a78bfa").encode(
            y=alt.Y("Length:N", sort=LENGTHS, title=None), x="Wickets:Q", tooltip=["Length", "Wickets"]), width="stretch")
    st.markdown("**Where batters score off him**")
    z = me[(me.wide == 0)].dropna(subset=["zone"])
    h = st.radio("Against", ["Right", "Left"], horizontal=True, format_func=lambda x: f"{x}-handers")
    zz = z[z.bat_hand == h]
    if len(zz) >= 20:
        wheel(zz, h == "Left", title=f"vs {h.lower()}-handers · {int(zz.bat_runs.sum())} runs off the bat")
    else:
        st.caption(f"Not enough balls against {h.lower()}-handers with a recorded shot area.")
    st.caption("Seen from behind the batter, bowler's end at the top; the off side is on the right for a right-hander, "
               "on the left for a left-hander.")

st.caption("Economy, average and strike rate count legal balls; runs conceded include wides and no-balls but not byes or leg byes. RAA, WAA, "
           "Impact and WPA follow the season tables (wides left out). Line, length, delivery type, shot and shot area "
           "are read from the commentary; false shots, over/round the wicket and 'in the air' come only from Cricket "
           "Australia's records. Speed, swing angle and exact pitch position are not in public commentary.")
footer()
