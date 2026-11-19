import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from lib import balls, footer, tiles

DEF = {
    "RAA": "Runs scored above average, given the game state (over, wickets lost, ground par).",
    "WAA": "Same as RAA, but for wickets: the value of not getting out, given the state.",
    "Impact": "Runs added given the game state (balls left, wickets lost, ground par), including the cost of losing your wicket.",
    "WPA": "Win probability added: how much this batting changed the team's chance of winning.",
}
LENGTHS = ["Full toss", "Yorker", "Full", "Good", "Back of length", "Short"]
LINES = ["Wide off", "Outside off", "Stumps", "Leg", "Wide leg"]
STYLES = ["Pace", "OB", "LB", "SLA", "LWS"]
PHASE_LBL = {"PP": "Powerplay (1-6)", "Mid1": "Middle 1 (7-11)", "Mid2": "Middle 2 (12-16)", "Slog": "Slog (17-20)"}
SECTORS = ["Third man", "Point", "Cover", "Long off", "Long on", "Mid wicket", "Square leg", "Fine leg"]
MIN_FIELD = 40
CTRL_HELP = ("Share of balls the batter was in control of: not edged, beaten, mistimed or hit on the pad. Leaves count "
             "as control. Recorded ball by ball only in Cricket Australia's data for the T20 Blast and some T20Is; "
             "BCCI commentary doesn't record it, so it shows – for IPL and SMAT.")

st.title("🔎 Batter explorer")
d = balls()
if d is None:
    st.info("This page needs the ball-by-ball table. Build it with `python metrics/ball_tags.py` "
            "(reads your local `cricket.db`), then reload.")
    st.stop()

# ---------------------------------------------------------------- filters
vol = d.groupby("batter", observed=True).size()
names = sorted(vol[vol >= 300].index)
with st.sidebar.form("filters"):
    bat = st.selectbox("Batter", names, index=names.index("Virat Kohli") if "Virat Kohli" in names else 0)
    leagues = sorted(d.league.cat.categories)
    lg = st.multiselect("League", leagues, placeholder="All leagues")
    y0, y1 = int(d.year.min()), int(d.year.max())
    yrs = st.slider("Years", y0, y1, (y0, y1))
    kind = st.pills("Pace / spin", ["pace", "spin"], selection_mode="multi", default=["pace", "spin"])
    style = st.pills("Bowler type", STYLES, selection_mode="multi",
                     help="Spin types come from the commentary. The pace bowler's arm isn't reliably recorded, so pace is one group.")
    length = st.pills("Length", LENGTHS, selection_mode="multi")
    line = st.pills("Line", LINES, selection_mode="multi")
    inns = st.pills("Innings", [1, 2], selection_mode="multi", default=[1, 2], format_func=lambda i: ("1st", "2nd")[i - 1])
    phase = st.pills("Phase", list(PHASE_LBL), selection_mode="multi", default=list(PHASE_LBL))
    entry = st.slider("Came in during overs", 1, 20, (1, 20), help="Innings where he arrived in this over range")
    venue = st.multiselect("Venue", sorted(d.venue.dropna().cat.categories), placeholder="Any venue")
    st.form_submit_button("Apply filters", type="primary", width="stretch")

base = d[d.year.between(*yrs) & (d.innings <= 2) & (d.league.isin(lg) if lg else True)]  # no super overs
f = base[base.innings.isin(inns or [1, 2]) & base.phase.isin(phase or list(PHASE_LBL))
         & ((base.entry_ball - 1) // 6 + 1).between(*entry)]
for col, sel in (("kind", kind), ("style", style), ("length", length), ("line", line), ("venue", venue)):
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
hand = me.bat_hand.dropna().mode()
tags = " · ".join(filter(None, [", ".join(lg) or "All leagues", f"{yrs[0]}–{yrs[1]}", "/".join(kind) if kind and len(kind) < 2 else "",
                                ", ".join(style or []), ", ".join(length or []), ", ".join(line or [])]))
st.markdown(f"**{bat}** · {(hand.iloc[0] + '-hand bat') if len(hand) else ''} · {tags} · **{len(me):,} balls in view**")

s = summary(me)
nd = {"Runs": 0, "Balls": 0, "Dismissals": 0, "4:6 ratio": 2}
tiles([(k, fmt(v, nd.get(k, 1)), "", "#a78bfa" if k in ("Strike rate", "Control %") else None,
        CTRL_HELP if k == "Control %" else None) for k, v in s.items()], min_w=105)
pos = lambda v: "#4ade80" if v > 0 else "#f87171" if v < 0 else None
tiles([(f"{k} per ball", f"{me[col].sum() / len(me):+.3f}", f"{me[col].sum():+.1f} total · {len(me):,} balls",
        pos(me[col].sum()), DEF[k]) for k, col in [("RAA", "raa"), ("WAA", "waa"), ("Impact", "impact"), ("WPA", "wpa")]],
      min_w=170)

agg = f.groupby("batter", observed=True).agg(n=("raa", "size"), raa=("raa", "mean"), waa=("waa", "mean"),
                                             impact=("impact", "mean"), wpa=("wpa", "mean"))
field = agg[agg.n >= MIN_FIELD]
if bat in field.index:
    with st.container(border=True):
        st.markdown(f"**Rankings** among the {len(field):,} batters with {MIN_FIELD}+ balls under the same filters")
        items = []
        for k, col in [("RAA", "raa"), ("WAA", "waa"), ("Impact", "impact"), ("WPA", "wpa")]:
            rk = int(field[col].rank(ascending=False, method="min")[bat])
            pct = round(100 * (1 - (rk - 1) / len(field)))
            suf = "th" if 10 <= pct % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
            items.append((f"{k} per ball", f"#{rk}", f"of {len(field):,} · {pct}{suf} percentile",
                          "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 40 else "#f87171", None))
        tiles(items, min_w=170)
else:
    st.caption(f"Rankings need {MIN_FIELD}+ balls under these filters.")

tab_p, tab_ll, tab_s = st.tabs(["Profile", "Lines and lengths", "Shot areas"])

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

    ph = p.groupby("phase", observed=True).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                                raa=("raa", "mean"), impact=("impact", "mean"))
    ph = ph.reindex(list(PHASE_LBL)).dropna(how="all")
    ph.index = ph.index.map(PHASE_LBL)
    ph.insert(2, "SR", 100 * ph.runs / ph.balls)
    st.markdown("**By phase**")
    st.dataframe(ph.drop(columns="runs").rename(columns={"raa": "RAA/ball", "impact": "Impact/ball", "balls": "Balls"}),
                 width="stretch", column_config={
                     "SR": st.column_config.NumberColumn(format="%.1f"),
                     "RAA/ball": st.column_config.NumberColumn(format="%+.3f"),
                     "Impact/ball": st.column_config.NumberColumn(format="%+.3f"),
                     "Balls": st.column_config.NumberColumn(format="%d")})


def zone_table(x, col, order):
    g = x.groupby(col, observed=True)
    t = pd.DataFrame({"Balls": g.size(), "Runs": g.bat_runs.sum(), "Outs": g.out.sum(), "Control %": 100 * g.ctrl.mean(),
                      "RAA/ball": g.raa.mean(), "4s": g.bat_runs.apply(lambda r: (r == 4).sum()),
                      "6s": g.bat_runs.apply(lambda r: (r == 6).sum())}).reindex(order).dropna(subset=["Balls"])
    t["SR"] = 100 * t.Runs / t.Balls
    t["Average"] = t.Runs / t.Outs.replace(0, np.nan)
    t["Balls/dis"] = t.Balls / t.Outs.replace(0, np.nan)
    cols = ["SR", "Average", "Balls/dis", "RAA/ball", "4s", "6s", "Balls"]
    return t[(["Control %"] if x.ctrl.notna().mean() >= 0.5 else []) + cols]


NUM = {c: st.column_config.NumberColumn(format="%.1f") for c in ["Control %", "SR", "Average", "Balls/dis"]}
NUM["RAA/ball"] = st.column_config.NumberColumn(format="%+.3f")

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
        hi_good = alt.Scale(scheme="redblue", reverse=True, domainMid={"Strike rate": float(100 * tagged.bat_runs.mean()),
                                                                       "Control %": float(100 * tagged.ctrl.mean()) if has_ctrl else 0.0,
                                                                       "RAA per ball": 0.0}[metric])
        heat = alt.Chart(g).encode(x=alt.X("line:N", sort=LINES, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=0)),
                                   y=alt.Y("length:N", sort=LENGTHS, title=None))
        fmt_ = ".2f" if metric == "RAA per ball" else ".0f"
        st.altair_chart((heat.mark_rect().encode(color=alt.Color(f"{metric}:Q", scale=hi_good), tooltip=[
            "length", "line", "balls", "runs", alt.Tooltip("Strike rate:Q", format=".0f"),
            alt.Tooltip("Control %:Q", format=".0f"), alt.Tooltip("RAA per ball:Q", format="+.3f")])
                         + heat.mark_text(fontSize=13, fontWeight="bold").encode(
                             text=alt.Text(f"{metric}:Q", format=fmt_), color=alt.value("#0b1622")))
                        .properties(height=330), width="stretch")
        st.caption("Red = good for the batter, blue = good for the bowler. Cells with fewer than 5 balls are hidden.")
    st.markdown("**By length**")
    st.dataframe(zone_table(me, "length", LENGTHS), width="stretch", column_config=NUM)
    st.markdown("**By line**")
    st.dataframe(zone_table(me, "line", LINES), width="stretch", column_config=NUM)

# ---------------------------------------------------------------- shot areas
with tab_s:
    z = me.dropna(subset=["zone"])
    st.caption(f"{len(z):,} of {len(me):,} balls in view name where the ball went. Sectors are seen from behind the batter.")
    if len(z):
        g = z.groupby("zone", observed=True).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                                 outs=("out", "sum")).reindex(SECTORS).fillna(0).reset_index()
        g["share"] = 100 * g.runs / max(g.runs.sum(), 1)
        lhb = len(hand) and hand.iloc[0] == "Left"
        # seen from behind the batter, bowler at the top: a right-hander's off side is on the right
        order = SECTORS if lhb else SECTORS[::-1]
        g["pos"] = g.zone.map({s: i for i, s in enumerate(order)})
        g["theta0"] = (g.pos * 45 - 180) * np.pi / 180  # straight hits sit either side of 0 (top)
        g["theta1"] = g.theta0 + np.pi / 4
        g["mid"] = g.theta0 + np.pi / 8
        g["label"] = g.zone.astype(str) + " " + g.share.round(0).astype(int).astype(str) + "%"
        base_ = alt.Chart(g).properties(width=360, height=360)
        wheel = base_.mark_arc(innerRadius=40, outerRadius=140, stroke="#0b1622", strokeWidth=2).encode(
            theta=alt.Theta("theta0:Q", scale=None), theta2="theta1:Q",
            color=alt.Color("share:Q", scale=alt.Scale(scheme="goldorange"), legend=None),
            tooltip=["zone", "runs", alt.Tooltip("share:Q", format=".1f", title="% of runs"), "balls", "outs"])
        text = base_.mark_text(radius=162, fontSize=11, color="#e5e7eb").encode(
            theta=alt.Theta("mid:Q", scale=None), text="label:N")
        st.altair_chart(wheel + text, width="content")
        st.caption(f"Seen from behind the batter, bowler's end at the top: {'off side on the left (left-hander)' if lhb else 'off side on the right'}. "
                   "Outs = dismissals caught or fielded in that sector.")
        t = g.set_index("zone")[["runs", "share", "balls", "outs"]].rename(columns={"share": "% of runs"})
        st.dataframe(t, width="stretch", column_config={"% of runs": st.column_config.NumberColumn(format="%.1f")})
        ft = me.dropna(subset=["foot"]).groupby("foot", observed=True).agg(
            balls=("bat_runs", "size"), runs=("bat_runs", "sum"), ctrl=("ctrl", "mean"))
        if len(ft):
            ft["SR"], ft["Control %"] = 100 * ft.runs / ft.balls, 100 * ft.ctrl
            st.markdown("**By footwork**")
            st.dataframe(ft[["SR"] + (["Control %"] if ft.ctrl.notna().any() else []) + ["balls"]],
                         width="stretch", column_config=NUM)

st.caption("Line, length, footwork and shot area are read from the ball-by-ball commentary (checked against Cricket "
           "Australia's own fields: 99% agreement), so coverage varies by competition: IPL 2012 onward and the CA "
           "leagues are near complete, SMAT has almost none, IPL 2008–11 (Cricsheet) has none. Control % needs a "
           "ball-by-ball shot-connection record, which only CA keeps (T20 Blast, some T20Is). Speed, swing, seam, "
           "bowling arm and exact pitch position are not in public commentary.")
footer()
