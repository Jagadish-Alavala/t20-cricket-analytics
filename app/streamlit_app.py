from pathlib import Path
import streamlit as st
from lib import style

st.set_page_config(page_title="T20 Cricket Analytics", page_icon="🏏", layout="wide")
style()

pages = {
    "": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True)],
    "Match day": [
        st.Page("views/batter_plan.py", title="Plan for a batter", icon=":material/target:"),
        st.Page("views/bowler_plan.py", title="Use a bowler", icon=":material/sports_cricket:"),
        st.Page("views/batting_order.py", title="Who bats next", icon=":material/format_list_numbered:"),
    ],
    "Squad and auction": [
        st.Page("views/player_value.py", title="Player value", icon=":material/leaderboard:"),
        st.Page("views/compare.py", title="Compare players", icon=":material/compare_arrows:"),
    ],
    "Research": [
        st.Page("views/impact_player.py", title="Impact Player rule", icon=":material/swap_horiz:"),
        st.Page("views/methods.py", title="Methods and checks", icon=":material/science:"),
    ],
}
if (Path(__file__).parent / "data" / "balls.parquet").exists():  # per-ball data stays local, not in the repo
    pages["Explorers"] = [st.Page("views/batter_explorer.py", title="Batter explorer", icon=":material/person_search:"),
                          st.Page("views/bowler_explorer.py", title="Bowler explorer", icon=":material/radar:")]
    pages = {k: pages[k] for k in ["", "Explorers", "Match day", "Squad and auction", "Research"]}
st.navigation(pages).run()
