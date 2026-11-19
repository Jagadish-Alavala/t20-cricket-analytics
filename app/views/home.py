from pathlib import Path
import streamlit as st
from lib import data, footer

st.title("🏏 T20 Cricket Analytics")
st.markdown("#### Ball-by-ball analytics for coaches, analysts and franchise decision-makers")
st.write("Pick the question you have. Every page uses tested numbers and says how far you can trust them.")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("### 🎯 Coach")
        st.write("How do we bowl to him? Where do the fielders go? Who walks in next?")
        if (Path(__file__).parents[1] / "data" / "balls.parquet").exists():
            st.page_link("views/batter_explorer.py", label="Batter explorer", icon="🔎")
            st.page_link("views/bowler_explorer.py", label="Bowler explorer", icon="🌀")
        st.page_link("views/batter_plan.py", label="Plan for a batter", icon="🎯")
        st.page_link("views/bowler_plan.py", label="Use a bowler", icon="🎳")
        st.page_link("views/batting_order.py", label="Who bats next", icon="📋")
with c2:
    with st.container(border=True):
        st.markdown("### 💰 Franchise")
        st.write("Who actually adds runs and wins once the match situation is accounted for, season by season?")
        st.page_link("views/player_value.py", label="Player value", icon="💰")
        st.page_link("views/compare.py", label="Compare players", icon="⚖️")
with c3:
    with st.container(border=True):
        st.markdown("### 🔬 Analyst")
        st.write("What did the Impact Player rule change? How were the models built, and how were they checked?")
        st.page_link("views/impact_player.py", label="Impact Player rule", icon="🔄")
        st.page_link("views/methods.py", label="Methods and checks", icon="🔬")

st.divider()
bat = data("batting_seasons.csv")
m = st.columns(4)
m[0].metric("Men's T20 matches", "5,123")
m[1].metric("Deliveries", "1.1 million")
m[2].metric("Leagues covered", bat.league.nunique())
m[3].metric("Player-seasons rated", f"{len(bat) + len(data('bowling_seasons.csv')):,}")
m = st.columns(4)
m[0].metric("Batters with matchup plans", "77+")
m[1].metric("Zone edge, year-to-year r", "0.63")
m[2].metric("WPA agreement with the primer", "r = 0.997")
m[3].metric("Win-model AUC (out of sample)", "0.857")
footer()
