import altair as alt
import pandas as pd
import streamlit as st
from lib import data, story_html, footer

st.title("🔄 What the Impact Player rule changed")
st.caption("IPL against leagues without the rule (T20 Blast, CPL), with SMAT as an on-off case. 2,017 full-length "
           "matches, 2019–2026. Difference in differences with season effects, and 95% ranges from a bootstrap over matches.")

c = st.columns(4)
c[0].metric("Top-7 bowling share, IPL", "25% → 13%", "−7.8 pts rule effect", delta_color="off")
c[1].metric("First-innings runs from the rule", "+10 to 15", "of the +23 IPL rise", delta_color="off")
c[2].metric("Top-3 strike rate (vs controls)", "+17", "RAA +6.9 per 100 balls", delta_color="off")
c[3].metric("Top-3 WPA per 100 balls", "−0.01", "no extra wins", delta_color="off")
st.info("**For a franchise:** the rule made top orders faster, not more valuable. Paying for strike rate at the "
        "top buys runs, but not wins. The real change is fewer batting all-rounders bowling, so specialist "
        "bowlers' overs matter more.")

t1, t2, t3, t4 = st.tabs(["Totals", "Roles", "Leaders since 2023", "Full story"])
with t1:
    t = data("impact_player_totals.csv")
    lg = st.multiselect("Leagues", sorted(t.league.unique()), default=sorted(t.league.unique()))
    t = t[t.league.isin(lg)]
    metric = st.radio("Measure", ["total", "boundary%", "pp rpo", "mid rpo", "death rpo", "wkts"], horizontal=True)
    base = alt.Chart(t).encode(x=alt.X("year:O", title=None), y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)),
                               color=alt.Color("league:N", title=None))
    st.altair_chart(base.mark_line() + base.mark_point(filled=True).encode(
        shape=alt.Shape("rule:N", title="Rule in force"), tooltip=list(t.columns)), width="stretch")
with t2:
    rc = data("impact_player_roles_change.csv")
    rc.columns = ["comparison", "order", "measure", "change", "lo95", "hi95"]
    d = rc[rc.comparison == "diff"].copy()
    d["batting order"] = d["order"].map({"1-3": "Top 3", "4-7": "4 to 7", "8-11": "8 to 11"})
    meas = st.radio("Measure", ["sr", "raa/100", "impact/100", "wpa/100", "waa/100", "balls/inns"], horizontal=True)
    x = d[d.measure == meas]
    bars = alt.Chart(x).mark_bar(color="#38bdf8").encode(x=alt.X("batting order:N", sort=None, title=None), y=alt.Y("change:Q", title=f"IPL change minus controls ({meas})"))
    err = alt.Chart(x).mark_errorbar(color="#e5e7eb").encode(x=alt.X("batting order:N", sort=None), y="lo95:Q", y2="hi95:Q")
    st.altair_chart(bars + err, width="stretch")
    st.caption("2023–26 against 2019–22, IPL minus the Blast and CPL. Bars that cross zero are not distinguishable from no change.")
with t3:
    ld = data("impact_player_wpa_leaders.csv").drop(columns="batter_key")
    st.dataframe(ld, hide_index=True, width="stretch")
with t4:
    st.iframe(story_html(), height=1200)
footer()
