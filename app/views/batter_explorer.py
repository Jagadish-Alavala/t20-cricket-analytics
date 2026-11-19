import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from lib import (balls, footer, tiles, rank_tiles, heatmap, wheel, LENGTHS, LINES, VARIATIONS,
                 EXPLORER_PHASES as PHASE_LBL)

DEF = {
    "RAA": "Runs scored above average, given the game state (over, wickets lost, ground par).",
    "WAA": "Same as RAA, but for wickets: the value of not getting out, given the state.",
    "Impact": "Runs added given the game state (balls left, wickets lost, ground par), including the cost of losing your wicket.",
    "WPA": "Win probability added: how much this batting changed the team's chance of winning.",
}
STYLES = ["Pace", "OB", "LB", "SLA", "LWS"]
CTRL_HELP = ("Share of balls the batter was in control of: not edged, beaten, mistimed or hit on the pad. Leaves count "
             "as control. Recorded ball by ball only in Cricket Australia's data for the T20 Blast and some T20Is; "
             "BCCI commentary doesn't record it, so it shows – for IPL and SMAT.")
MIN_FIELD = 40

st.title("🔎 Batter explorer")
d = balls()
if d is None:
    st.info("This page needs the ball-by-ball table. Build it with `python metrics/ball_tags.py`, then reload.")
    st.stop()
d = d[d.wide == 0]  # a wide is not a ball faced

# ---------------------------------------------------------------- filters
vol = d.groupby("batter", observed=True).size()
names = sorted(vol[vol >= 300].index)
with st.sidebar.form("bat_filters"):
    bat = st.selectbox("Batter", names, index=names.index("Virat Kohli") if "Virat Kohli" in names else 0)
    lg = st.multiselect("League", sorted(d.league.cat.categories), placeholder="All leagues")
    y0, y1 = int(d.year.min()), int(d.year.max())
    yrs = st.slider("Years", y0, y1, (y0, y1))
    kind = st.pills("Pace / spin", ["pace", "spin"], selection_mode="multi", default=["pace", "spin"])
    style = st.pills("Bowler type", STYLES, selection_mode="multi",
                     help="Spin types come from the commentary. The pace bowler's arm isn't reliably recorded.")
    var = st.pills("Delivery", VARIATIONS, selection_mode="multi", help="Variation named in the commentary")
    length = st.pills("Length", LENGTHS, selection_mode="multi")
    line = st.pills("Line", LINES, selection_mode="multi")
    side = st.pills("Over / round the wicket", ["Over", "Round"], selection_mode="multi",
                    help="Recorded by Cricket Australia, not in BCCI commentary")
    inns = st.pills("Innings", [1, 2], selection_mode="multi", default=[1, 2], format_func=lambda i: ("1st", "2nd")[i - 1])
    phase = st.pills("Phase", list(PHASE_LBL), selection_mode="multi", default=list(PHASE_LBL))
    entry = st.slider("Came in during overs", 1, 20, (1, 20), help="Innings where he arrived in this over range")
    venue = st.multiselect("Venue", sorted(d.venue.dropna().cat.categories), placeholder="Any venue")
    st.form_submit_button("Apply filters", type="primary", width="stretch")

base = d[d.year.between(*yrs) & (d.innings <= 2) & (d.league.isin(lg) if lg else True)]  # no super overs
f = base[base.innings.isin(inns or [1, 2]) & base.phase.isin(phase or list(PHASE_LBL))
         & ((base.entry_ball - 1) // 6 + 1).between(*entry)]
for col, sel in (("kind", kind), ("style", style), ("variation", var), ("length", length), ("line", line),
                 ("side", side), ("venue", venue)):
    if sel:
        f = f[f[col].isin(sel)]
me = f[f.batter == bat]
if me.empty:
    st.warning(f"No balls for {bat} with these filters.")
    st.stop()


def summary(x):
    b, r, o = len(x), int(x.bat_runs.sum()), int(x.out.sum())
    fours, sixes = int((x.bat_runs == 4).sum()), int((x.bat_runs == 6).sum())
    c = x.ctrl.dropna()
    c = c if len(c) >= 0.5 * len(x) else c.iloc[:0]  # only when recorded for most balls in view
    return {"Runs": r, "Balls": b, "Strike rate": 100 * r / b, "Average": r / o if o else np.nan,
            "4 %": 100 * fours / b, "6 %": 100 * sixes / b, "4:6 ratio": fours / sixes if sixes else np.nan,
            "Dot %": 100 * (x.bat_runs == 0).mean(), "Control %": 100 * c.mean() if len(c) else np.nan,
            "Dismissals": o}


fmt = lambda v, p=1: "–" if pd.isna(v) else f"{v:,.{p}f}" if isinstance(v, (float, np.floating)) else f"{v:,}"
pos = lambda v: "#4ade80" if v > 0 else "#f87171" if v < 0 else None
hand = me.bat_hand.dropna().mode()
lhb = bool(len(hand)) and hand.iloc[0] == "Left"
chosen = [", ".join(lg) or "All leagues", f"{yrs[0]}–{yrs[1]}"] + [", ".join(map(str, s)) for s in
                                                                    (style, var, length, line, side) if s]
st.markdown(f"**{bat}** · {(hand.iloc[0] + '-hand bat') if len(hand) else ''} · {' · '.join(chosen)} · "
            f"**{len(me):,} balls in view**")

nd = {"Runs": 0, "Balls": 0, "Dismissals": 0, "4:6 ratio": 2}
tiles([(k, fmt(v, nd.get(k, 1)), "", "#a78bfa" if k in ("Strike rate", "Control %") else None,
        CTRL_HELP if k == "Control %" else None) for k, v in summary(me).items()], min_w=105)
tiles([(f"{k} per ball", f"{me[c].sum() / len(me):+.3f}", f"{me[c].sum():+.1f} total · {len(me):,} balls",
        pos(me[c].sum()), DEF[k]) for k, c in [("RAA", "raa"), ("WAA", "waa"), ("Impact", "impact"), ("WPA", "wpa")]],
      min_w=170)

agg = f.groupby("batter", observed=True).agg(n=("raa", "size"), raa=("raa", "mean"), waa=("waa", "mean"),
                                             impact=("impact", "mean"), wpa=("wpa", "mean"))
field = agg[agg.n >= MIN_FIELD]
if bat in field.index:
    with st.container(border=True):
        rank_tiles(field, bat, [("RAA", "raa"), ("WAA", "waa"), ("Impact", "impact"), ("WPA", "wpa")],
                   f"batters with {MIN_FIELD}+ balls")
else:
    st.caption(f"Rankings need {MIN_FIELD}+ balls under these filters.")


def zone_table(x, col, order=None):
    g = x.groupby(col, observed=True)
    t = pd.DataFrame({"Balls": g.size(), "Runs": g.bat_runs.sum(), "Outs": g.out.sum(), "Control %": 100 * g.ctrl.mean(),
                      "RAA/ball": g.raa.mean(), "4s": g.bat_runs.apply(lambda r: (r == 4).sum()),
                      "6s": g.bat_runs.apply(lambda r: (r == 6).sum())})
    t = (t.reindex(order) if order else t.sort_values("Balls", ascending=False)).dropna(subset=["Balls"])
    t = t[t.Balls > 0]
    t["% of balls"] = 100 * t.Balls / t.Balls.sum()
    t["SR"] = 100 * t.Runs / t.Balls
    t["Average"] = t.Runs / t.Outs.replace(0, np.nan)
    t["Balls/dis"] = t.Balls / t.Outs.replace(0, np.nan)
    cols = ["% of balls", "SR", "Average", "Balls/dis", "RAA/ball", "4s", "6s", "Outs", "Balls"]
    return t[(["Control %"] if x.ctrl.notna().mean() >= 0.5 else []) + cols]


NUM = {c: st.column_config.NumberColumn(format="%.1f") for c in ["Control %", "SR", "Average", "Balls/dis", "% of balls"]}
NUM["RAA/ball"] = st.column_config.NumberColumn(format="%+.3f")
NUM["In the air %"] = st.column_config.NumberColumn(format="%.0f")

tab_p, tab_ll, tab_sh, tab_s = st.tabs(["Profile", "Lines and lengths", "Shots and dismissals", "Shot areas"])

# ---------------------------------------------------------------- profile (filters except league and years)
with tab_p:
    p = base[base.batter == bat]
    st.caption("Uses only the league and year filters, so it describes the batter as a whole.")
    first5 = p[p.ball_no <= 5].groupby("kind", observed=True).raa.mean()
    entry_over = (p.groupby(["match_id", "innings"], observed=True).entry_ball.first() - 1) // 6 + 1
    tiles([("RAA per ball", f"{p.raa.mean():+.3f}", f"{p.raa.sum():+.1f} total · {len(p):,} balls", pos(p.raa.mean()), DEF["RAA"]),
           ("First 5 balls · RAA vs pace", f"{first5.get('pace', np.nan):+.3f}", "mean per ball", "#38bdf8", None),
           ("First 5 balls · RAA vs spin", f"{first5.get('spin', np.nan):+.3f}", "mean per ball", "#f59e0b", None),
           ("Usual entry", f"Over {entry_over.median():.0f}", "median", None, None),
           ("Innings", f"{len(entry_over):,}", "", None, None)], min_w=150)

    a, b, e = st.columns(3)
    with a:
        st.markdown("**Getting started**: mean RAA by ball faced")
        g = p[p.ball_no <= 15].groupby(["ball_no", "kind"], observed=True).raa.agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 10]
        st.altair_chart(alt.Chart(g).mark_line(point=True).encode(
            x=alt.X("ball_no:Q", title="Ball of his innings"), y=alt.Y("mean:Q", title="Mean RAA"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["pace", "spin"], range=["#38bdf8", "#f59e0b"]), title=None),
            tooltip=["ball_no", "kind", alt.Tooltip("mean:Q", format="+.3f"), alt.Tooltip("size:Q", title="balls")]),
            width="stretch")
    with b:
        st.markdown("**Entry points**: over he came in")
        eo = entry_over.value_counts().sort_index().rename_axis("over").reset_index(name="innings")
        st.altair_chart(alt.Chart(eo).mark_bar(color="#22c55e").encode(
            x=alt.X("over:O", title="Over"), y=alt.Y("innings:Q"), tooltip=["over", "innings"]), width="stretch")
    with e:
        st.markdown("**Impact trajectory**: expected impact after n balls")
        t = p[p.ball_no <= 60].groupby("ball_no").impact.agg(["mean", "size"])
        t = t[t["size"] >= 10].assign(cum=lambda x: x["mean"].cumsum()).reset_index()
        st.altair_chart(alt.Chart(t).mark_area(line={"color": "#ef4444"}, color="#ef444433").encode(
            x=alt.X("ball_no:Q", title="Balls faced"), y=alt.Y("cum:Q", title="Cumulative impact"),
            tooltip=["ball_no", alt.Tooltip("cum:Q", format="+.2f"), alt.Tooltip("size:Q", title="innings")]),
            width="stretch")

    ph = p.groupby("phase", observed=True).agg(Balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                                raa=("raa", "mean"), impact=("impact", "mean"))
    ph = ph.reindex(list(PHASE_LBL)).dropna(how="all")
    ph.index = ph.index.map(PHASE_LBL)
    ph.insert(1, "SR", 100 * ph.runs / ph.Balls)
    st.markdown("**By phase**")
    st.dataframe(ph.drop(columns="runs").rename(columns={"raa": "RAA/ball", "impact": "Impact/ball"}), width="stretch",
                 column_config={"SR": st.column_config.NumberColumn(format="%.1f"),
                                "RAA/ball": st.column_config.NumberColumn(format="%+.3f"),
                                "Impact/ball": st.column_config.NumberColumn(format="%+.3f"),
                                "Balls": st.column_config.NumberColumn(format="%d")})

# ---------------------------------------------------------------- lines and lengths
with tab_ll:
    tagged = me.dropna(subset=["length", "line"])
    st.caption(f"{len(tagged):,} of {len(me):,} balls in view have both line and length recorded in the commentary.")
    if len(tagged):
        has_ctrl = tagged.ctrl.notna().mean() >= 0.5
        metric = st.radio("Colour by", ["Strike rate", "RAA per ball"] + (["Control %"] if has_ctrl else []), horizontal=True)
        g = tagged.groupby(["length", "line"], observed=True).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                                                 ctrl=("ctrl", "mean"), raa=("raa", "mean")).reset_index()
        g = g[g.balls >= 5]
        g["Strike rate"], g["Control %"], g["RAA per ball"] = 100 * g.runs / g.balls, 100 * g.ctrl, g.raa
        mid = {"Strike rate": float(100 * tagged.bat_runs.mean()), "RAA per ball": 0.0,
               "Control %": float(100 * tagged.ctrl.mean()) if has_ctrl else 0.0}[metric]
        heatmap(g.drop(columns=["ctrl", "raa"]), metric, ".2f" if metric == "RAA per ball" else ".0f", mid)
        st.caption("Red = good for the batter, blue = good for the bowler. Cells with fewer than 5 balls are hidden.")
    st.markdown("**By length**")
    st.dataframe(zone_table(me, "length", LENGTHS), width="stretch", column_config=NUM)
    st.markdown("**By line**")
    st.dataframe(zone_table(me, "line", LINES), width="stretch", column_config=NUM)

# ---------------------------------------------------------------- shots, deliveries and dismissals
with tab_sh:
    sh = me.dropna(subset=["shot"])
    st.caption(f"{len(sh):,} of {len(me):,} balls in view name the shot.")
    if len(sh):
        t = zone_table(sh, "shot")
        air = sh.groupby("shot", observed=True).aerial.mean()
        if air.notna().any():
            t.insert(0, "In the air %", 100 * air)
        st.markdown("**By shot**")
        st.dataframe(t, width="stretch", column_config=NUM)
        pts = t.reset_index().dropna(subset=["Balls/dis"])
        pts = pts[pts.Balls >= 10]
        if len(pts):
            st.altair_chart(alt.Chart(pts).mark_circle(opacity=.8, color="#a78bfa").encode(
                x=alt.X("SR:Q", title="Strike rate"), y=alt.Y("Balls/dis:Q", title="Balls per dismissal (higher = safer)"),
                size=alt.Size("Balls:Q", legend=None),
                tooltip=["shot", "Balls", alt.Tooltip("SR:Q", format=".0f"), alt.Tooltip("Balls/dis:Q", format=".0f"),
                         alt.Tooltip("RAA/ball:Q", format="+.3f")])
                            + alt.Chart(pts).mark_text(dy=-12, color="#e5e7eb", fontSize=11).encode(
                                x="SR:Q", y="Balls/dis:Q", text="shot:N"), width="stretch")
            st.caption("Risk against reward: right is faster scoring, up is safer. Shots with 10+ balls and at least one dismissal.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Against each delivery type**")
        v = me.dropna(subset=["variation"])
        if len(v):
            st.dataframe(zone_table(v, "variation")[["SR", "Balls/dis", "RAA/ball", "Balls"]], width="stretch", column_config=NUM)
    with b:
        st.markdown("**How he gets out**")
        dis = me[me.out == 1].dismissal.value_counts().rename_axis("How").reset_index(name="Dismissals")
        dis = dis[dis.Dismissals > 0]
        if len(dis):
            st.altair_chart(alt.Chart(dis).mark_bar(color="#f87171").encode(
                y=alt.Y("How:N", sort="-x", title=None), x="Dismissals:Q", tooltip=["How", "Dismissals"]), width="stretch")
        st.caption(f"{int(me.out.sum() - dis.Dismissals.sum())} dismissals without a recorded type.")

# ---------------------------------------------------------------- shot areas
with tab_s:
    z = me.dropna(subset=["zone"])
    st.caption(f"{len(z):,} of {len(me):,} balls in view name where the ball went.")
    if len(z):
        g = wheel(z, lhb)
        st.caption("Seen from behind the batter, bowler's end at the top, "
                   f"{'off side on the left (left-hander)' if lhb else 'off side on the right'}. "
                   "Outs = dismissals caught or fielded in that sector.")
        st.dataframe(g.set_index("zone")[["runs", "share", "balls", "outs"]].rename(columns={"share": "% of runs"}),
                     width="stretch", column_config={"% of runs": st.column_config.NumberColumn(format="%.1f")})
        ft = me.dropna(subset=["foot"]).groupby("foot", observed=True).agg(
            balls=("bat_runs", "size"), runs=("bat_runs", "sum"), ctrl=("ctrl", "mean"))
        if len(ft):
            ft["SR"], ft["Control %"] = 100 * ft.runs / ft.balls, 100 * ft.ctrl
            st.markdown("**By footwork**")
            st.dataframe(ft[["SR"] + (["Control %"] if ft.ctrl.notna().any() else []) + ["balls"]],
                         width="stretch", column_config=NUM)

st.caption("Line, length, footwork, shot, delivery type and shot area are read from the ball-by-ball commentary "
           "(checked against Cricket Australia's own fields: 99% agreement), so coverage varies: IPL 2012 onward and "
           "the CA leagues are near complete, SMAT has almost none, IPL 2008–11 (Cricsheet) has none. Control %, "
           "over/round the wicket and 'in the air' come only from Cricket Australia's records. Speed, swing angle, "
           "bowling arm and exact pitch position are not in public commentary.")
footer()
