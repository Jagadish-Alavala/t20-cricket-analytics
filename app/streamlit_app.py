from pathlib import Path
import streamlit as st

st.set_page_config(page_title="T20 Cricket Analytics", page_icon="🏏", layout="wide")

pages = {
    "": [st.Page("views/home.py", title="Home", icon="🏠", default=True)],
    "Match day": [
        st.Page("views/batter_plan.py", title="Plan for a batter", icon="🎯"),
        st.Page("views/bowler_plan.py", title="Use a bowler", icon="🎳"),
        st.Page("views/batting_order.py", title="Who bats next", icon="📋"),
    ],
    "Squad and auction": [
        st.Page("views/player_value.py", title="Player value", icon="💰"),
        st.Page("views/compare.py", title="Compare players", icon="⚖️"),
    ],
    "Research": [
        st.Page("views/impact_player.py", title="Impact Player rule", icon="🔄"),
        st.Page("views/methods.py", title="Methods and checks", icon="🔬"),
    ],
}
if (Path(__file__).parent / "data" / "balls.parquet").exists():  # per-ball data stays local, not in the repo
    pages["Match day"][:0] = [st.Page("views/batter_explorer.py", title="Batter explorer", icon="🔎"),
                              st.Page("views/bowler_explorer.py", title="Bowler explorer", icon="🌀")]
st.navigation(pages).run()
