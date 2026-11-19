from pathlib import Path
import streamlit as st
from lib import data, footer, header, tiles

header("T20 Cricket Analytics",
       "Ball-by-ball analytics for coaches, analysts and franchise decision-makers. Pick the question you have: "
       "every page uses tested numbers and says how far you can trust them.",
       eyebrow="Jagadish Alavala · cricket analytics")

local = (Path(__file__).parents[1] / "data" / "balls.parquet").exists()
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/sports_cricket: Coach")
        st.caption("How do we bowl to him? Where do the fielders go? Who walks in next?")
        if local:
            st.page_link("views/batter_explorer.py", label="Batter explorer", icon=":material/person_search:")
            st.page_link("views/bowler_explorer.py", label="Bowler explorer", icon=":material/radar:")
        st.page_link("views/batter_plan.py", label="Plan for a batter", icon=":material/target:")
        st.page_link("views/bowler_plan.py", label="Use a bowler", icon=":material/sports_cricket:")
        st.page_link("views/batting_order.py", label="Who bats next", icon=":material/format_list_numbered:")
with c2:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/leaderboard: Franchise")
        st.caption("Who actually adds runs and wins once the match situation is accounted for, season by season?")
        st.page_link("views/player_value.py", label="Player value", icon=":material/leaderboard:")
        st.page_link("views/compare.py", label="Compare players", icon=":material/compare_arrows:")
with c3:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/science: Analyst")
        st.caption("What did the Impact Player rule change? How were the models built, and how were they checked?")
        st.page_link("views/impact_player.py", label="Impact Player rule", icon=":material/swap_horiz:")
        st.page_link("views/methods.py", label="Methods and checks", icon=":material/science:")

st.write("")
bat = data("batting_seasons.csv")
st.markdown("<div class='app-eyebrow'>The data</div>", unsafe_allow_html=True)
tiles([("Men's T20 matches", "5,123", "2008–2026", None, None),
       ("Deliveries", "1.1 million", "ball by ball", None, None),
       ("Leagues", f"{bat.league.nunique()}", "IPL, BBL, Blast, CPL, T20Is…", None, None),
       ("Player-seasons rated", f"{len(bat) + len(data('bowling_seasons.csv')):,}", "batting and bowling", None, None)],
      min_w=180)
st.markdown("<div class='app-eyebrow'>How it was checked</div>", unsafe_allow_html=True)
tiles([("WPA vs the primer's tables", "r = 0.997", "IPL 2026, published reference", None, None),
       ("Win model AUC", "0.857", "out of sample, 2024", None, None),
       ("Zone edge, year to year", "r = 0.63", "batter matchups repeat", None, None),
       ("Interactive field plans", "77", "batters with enough zone data", None, None)], min_w=180)
footer()
