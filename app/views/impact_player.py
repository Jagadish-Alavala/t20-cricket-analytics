import altair as alt
import pandas as pd
import streamlit as st
from lib import data, story_html, footer, header, tiles, INK2

header("What the Impact Player rule changed",
       "IPL against leagues without the rule (T20 Blast, CPL), with SMAT as an on-off case: 2,017 full-length "
       "matches, 2019–2026. Difference in differences with season effects; 95% ranges from a bootstrap over matches.",
       eyebrow="Research")
tiles([("Top-7 bowling share, IPL", "25% → 13%", "rule effect −7.8 points", None, None),
       ("First-innings runs from the rule", "+10 to 15", "of the +23 IPL rise", None, None),
       ("Top-3 strike rate vs controls", "+17", "RAA +6.9 per 100 balls", None, None),
       ("Top-3 WPA per 100 balls", "−0.01", "no extra wins", None, None)], min_w=190)
with st.container(border=True):
    st.markdown("**For a franchise:** the rule made top orders faster, not more valuable. Paying for strike rate at "
                "the top buys runs, but not wins. The real change is fewer batting all-rounders bowling, so specialist "
                "bowlers' overs matter more.")

t1, t2, t3, t4 = st.tabs(["Totals", "Roles", "Leaders since 2023", "Full story"])
with t1:
    t = data("impact_player_totals.csv")
    lg = st.multiselect("Leagues", sorted(t.league.unique()), default=sorted(t.league.unique()))
    t = t[t.league.isin(lg)].assign(rule=lambda x: x.rule.astype(str).map({"True": "Yes", "False": "No"}))
    MEAS = {"total": "First-innings total", "boundary%": "Boundary %", "pp rpo": "Powerplay run rate",
            "mid rpo": "Middle-overs run rate", "death rpo": "Death-overs run rate", "wkts": "Wickets per innings"}
    metric = st.radio("Measure", list(MEAS), format_func=MEAS.get, horizontal=True)
    base = alt.Chart(t).encode(x=alt.X("year:O", title=None), y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False), title=MEAS[metric]),
                               color=alt.Color("league:N", title=None, legend=alt.Legend(orient="top")))
    st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=70).encode(
        shape=alt.Shape("rule:N", title="Rule in force", legend=alt.Legend(orient="top")), tooltip=list(t.columns)))
                    .properties(height=380), width="stretch")
    st.caption("The point shape marks seasons with the rule in force (IPL from 2023, SMAT 2022–23).")
with t2:
    rc = data("impact_player_roles_change.csv")
    rc.columns = ["comparison", "order", "measure", "change", "lo95", "hi95"]
    d = rc[rc.comparison == "diff"].copy()
    d["batting order"] = d["order"].map({"1-3": "Top 3", "4-7": "4 to 7", "8-11": "8 to 11"})
    RM = {"sr": "Strike rate", "raa/100": "RAA per 100 balls", "impact/100": "Impact per 100 balls",
          "wpa/100": "WPA per 100 balls", "waa/100": "WAA per 100 balls", "balls/inns": "Balls per innings"}
    meas = st.radio("Measure", list(RM), format_func=RM.get, horizontal=True)
    x = d[d.measure == meas]
    bars = alt.Chart(x).mark_bar(cornerRadiusEnd=4, width={"band": 0.5}).encode(
        x=alt.X("batting order:N", sort=None, title=None), y=alt.Y("change:Q", title=f"{RM[meas]}: IPL change minus controls"),
        tooltip=["batting order", alt.Tooltip("change:Q", format="+.2f"), alt.Tooltip("lo95:Q", format="+.2f"),
                 alt.Tooltip("hi95:Q", format="+.2f")])
    err = alt.Chart(x).mark_errorbar(color=INK2, thickness=2, ticks=True).encode(
        x=alt.X("batting order:N", sort=None), y="lo95:Q", y2="hi95:Q")
    st.altair_chart((bars + err).properties(height=340), width="stretch")
    st.caption("2023–26 against 2019–22, IPL minus the Blast and CPL. Bars that cross zero are not distinguishable from no change.")
with t3:
    ld = data("impact_player_wpa_leaders.csv").drop(columns="batter_key")
    N = st.column_config.NumberColumn
    st.caption("IPL batters with the most wins added since the rule came in (2023–2026).")
    st.dataframe(ld, hide_index=True, width="stretch", column_config={
        "name": "Batter", "balls": N("Balls"), "sr": N("SR", format="%.1f"), "wpa": N("WPA", format="%+.2f"),
        "raa/100": N("RAA/100", format="%+.1f"), "waa/100": N("WAA/100", format="%+.2f"),
        "impact/100": N("Impact/100", format="%+.1f")})
with t4:
    st.iframe(story_html(), height=1200)
footer()
