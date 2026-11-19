import altair as alt
import streamlit as st
from lib import data, footer, header

header("Player value", "Situation-adjusted value for every player and season, in 11 leagues. Use it for "
       "retention, the auction and squad building.", eyebrow="Squad and auction")
METRIC_NAMES = {"wpa": "WPA (wins added)", "raa": "RAA (runs above average)", "impact": "Impact",
                "wpa_100": "WPA per 100 balls", "raa_100": "RAA per 100 balls", "waa": "WAA (wickets above average)"}

with st.expander("What the metrics mean", expanded=False):
    st.markdown("""
- **WPA (win probability added)**: wins the player added. For example, +1.0 means one extra win's worth of probability over the season. This is the bottom line.
- **RAA (runs above average)**: runs added against an average player in the same match situations (balls left, wickets, par, innings). For bowlers it means runs saved.
- **WAA (wickets above average)**: for batters, wickets kept compared with average; for bowlers, wickets taken.
- **Impact**: runs scored, plus the change in projected total. It credits scoring without losing wickets.
- **/100**: per 100 balls, so players with different workloads can be compared.

Definitions follow Himanish Ganjoo's *T20 Metrics: A Primer*; the fits are my own, on this data.
""")

tab_bat, tab_bowl = st.tabs(["Batting", "Bowling"])
for tab, file, runs_label in ((tab_bat, "batting_seasons.csv", "runs"), (tab_bowl, "bowling_seasons.csv", "runs conceded")):
    with tab:
        df = data(file)
        k = file[:4]
        c = st.columns([1.2, 2, 1, 1.2, 1.5])
        leagues = sorted(df.league.unique())
        lg = c[0].selectbox("League", leagues, index=leagues.index("IPL"), key=f"lg{k}")
        d = df[df.league == lg]
        y0, y1 = int(d.year.min()), int(d.year.max())
        yr = c[1].slider("Seasons", y0, y1, (max(y0, y1 - 3), y1), key=f"yr{k}")
        mb = c[2].number_input("Min balls", 30, 600, 120 if lg == "IPL" else 60, 30, key=f"mb{k}")
        sort = c[3].selectbox("Rank by", list(METRIC_NAMES), format_func=METRIC_NAMES.get, key=f"s{k}")
        q = c[4].text_input("Search player", key=f"q{k}")
        d = d[d.year.between(*yr) & (d.balls >= mb)]
        if q:
            d = d[d.player.str.contains(q, case=False)]
        d = d.sort_values(sort, ascending=False).reset_index(drop=True)
        d.index += 1
        hi = max(float(d.wpa.abs().max()), 0.01) if len(d) else 1.0
        N = st.column_config.NumberColumn
        st.dataframe(d.drop(columns="league"), width="stretch", height=440, column_config={
            "player": st.column_config.TextColumn("Player", width="medium"), "year": N("Season", format="%d"),
            "inns": N("Inns"), "balls": N("Balls"), "runs": N(runs_label.capitalize()),
            "sr": N("SR", format="%.1f"), "econ": N("Econ", format="%.2f"), "outs": N("Outs"), "wkts": N("Wkts"),
            "impact": N("Impact", format="%+.1f"), "raa": N("RAA", format="%+.1f"), "waa": N("WAA", format="%+.2f"),
            "raa_100": N("RAA/100", format="%+.1f"),
            "wpa": st.column_config.ProgressColumn("WPA", format="%+.2f", min_value=-hi, max_value=hi),
            "wpa_100": N("WPA/100", format="%+.2f")})
        st.caption(f"{len(d)} player-seasons")
        if len(d):
            st.markdown("**Efficiency against volume**: RAA per 100 balls against total WPA. Top right is the elite "
                        "group; the eight highest WPA seasons are named.")
            pts = d.head(300).assign(label=lambda x: x.player + " " + x.year.astype(str).str[-2:])
            base = alt.Chart(pts).encode(x=alt.X("raa_100:Q", title="RAA per 100 balls"), y=alt.Y("wpa:Q", title="WPA"))
            dots = base.mark_circle(opacity=.8, stroke="#121211", strokeWidth=1).encode(
                size=alt.Size("balls:Q", legend=None, scale=alt.Scale(range=[30, 260])),
                color=alt.Color("year:O", title="Season", scale=alt.Scale(range=["#1c5cab", "#9ec5f4"])),
                tooltip=["player", "year", "balls", "runs", alt.Tooltip("raa:Q", format="+.1f"),
                         alt.Tooltip("raa_100:Q", format="+.1f"), alt.Tooltip("wpa:Q", format="+.2f")])
            names = base.transform_filter(alt.datum.wpa >= float(pts.wpa.nlargest(8).min())).mark_text(
                align="left", dx=8, fontSize=11, color="#c3c2b7").encode(text="label:N")
            st.altair_chart((dots + names).properties(height=420).interactive(), width="stretch")
footer()
