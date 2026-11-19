import altair as alt
import streamlit as st
from lib import data, footer

st.title("💰 Player value")
st.caption("Situation-adjusted value for each player and season. Use it for retention, the auction and squad building.")

with st.expander("What the metrics mean", expanded=False):
    st.markdown("""
- **WPA (win probability added)**: wins the player added. For example, +1.0 means one extra win's worth of probability over the season. This is the bottom line.
- **RAA (runs above average)**: runs added against an average player in the same match situations (balls left, wickets, par, innings). For bowlers it means runs saved.
- **WAA (wickets above average)**: for batters, wickets kept compared with average; for bowlers, wickets taken.
- **Impact**: runs scored, plus the change in projected total. It credits scoring without losing wickets.
- **/100**: per 100 balls, so players with different workloads can be compared.

Definitions follow Himanish Ganjoo's *T20 Metrics: A Primer*; the fits are my own, on this data.
""")

tab_bat, tab_bowl = st.tabs(["🏏 Batting", "🎳 Bowling"])
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
        sort = c[3].selectbox("Rank by", ["wpa", "raa", "impact", "wpa_100", "raa_100", "waa"], key=f"s{k}")
        q = c[4].text_input("Search player", key=f"q{k}")
        d = d[d.year.between(*yr) & (d.balls >= mb)]
        if q:
            d = d[d.player.str.contains(q, case=False)]
        d = d.sort_values(sort, ascending=False).reset_index(drop=True)
        d.index += 1
        hi = max(float(d.wpa.abs().max()), 0.01) if len(d) else 1.0
        st.dataframe(d.drop(columns="league"), width="stretch", height=420, column_config={
            "year": st.column_config.NumberColumn(format="%d"),
            "runs": st.column_config.NumberColumn(runs_label),
            "wpa": st.column_config.ProgressColumn("WPA", format="%.2f", min_value=-hi, max_value=hi),
            "raa_100": "RAA/100", "wpa_100": "WPA/100"})
        st.caption(f"{len(d)} player-seasons")
        if len(d):
            st.markdown("**Efficiency against volume**: RAA per 100 balls against total WPA. Top right is the elite group.")
            ch = alt.Chart(d.head(300)).mark_circle(opacity=.75).encode(
                x=alt.X("raa_100:Q", title="RAA per 100 balls"), y=alt.Y("wpa:Q", title="WPA"),
                size=alt.Size("balls:Q", legend=None), color=alt.Color("year:O", title="Season"),
                tooltip=["player", "year", "balls", "runs", "raa", "raa_100", "wpa", "impact"])
            st.altair_chart(ch.interactive(), width="stretch")
footer()
