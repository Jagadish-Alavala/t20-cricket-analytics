import altair as alt
import pandas as pd
import streamlit as st
from lib import batters, bowler_kind, recommend, footer, header, tiles, BLUE

header("Who bats next", "Set the match state and the bowlers who still have overs left. Every batter still to come "
       "is ranked on phase fit, how he matches up against those exact bowlers, and safety. Treat it as advice, "
       "not an optimum.", eyebrow="Match day")

names, kinds = batters(), bowler_kind()
default_c = [n for n in ["Suryakumar Yadav", "Tilak Varma", "Tim David", "Hardik Pandya", "Nicholas Pooran",
                         "Heinrich Klaasen", "Ravindra Jadeja"] if n in names]
l, r = st.columns([1, 1])
with l:
    over = st.slider("Overs completed", 0, 19, 15)
    need = st.slider("Situation", 0.0, 1.0, 0.9, 0.05,
                     help="0 = protect the wicket (rebuild), 1 = must attack (death chase)")
    st.caption("0 = protect the wicket · 1 = must attack")
    cands = st.multiselect("Batters still to come", names, default=default_c)
with r:
    st.markdown("**Bowlers with overs left**")
    start = pd.DataFrame({"Bowler": [b for b in ["Jasprit Bumrah", "Matheesha Pathirana", "Yuzvendra Chahal"] if b in kinds],
                          "Overs left": [2, 2, 1][:3]})
    ed = st.data_editor(start, num_rows="dynamic", width="stretch", hide_index=True, column_config={
        "Bowler": st.column_config.SelectboxColumn(options=sorted(kinds), required=True),
        "Overs left": st.column_config.NumberColumn(min_value=0, max_value=4, step=1, required=True)})

bowlers = [(b, int(o)) for b, o in ed.dropna().itertuples(index=False) if b in kinds]
if not cands or not bowlers:
    st.caption("Add at least one batter and one bowler.")
else:
    res = recommend(cands, over, bowlers, need)
    top = res.iloc[0]
    tiles([("Send in", top.Batter, f"score {top.Score:+.0f}", None, None),
           ("Target", top.Target, "his best matchup", None, None),
           ("Careful against", top["Careful against"], "his weakest matchup", None, None),
           ("Why", f"{top['Phase fit']:+.0f} · {top.Matchup:+.0f} · {top.Safety:+.1f}", "phase fit · matchup · safety", None, None)],
          min_w=170)
    st.altair_chart(alt.Chart(res).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Batter:N", sort="-x", title=None), x=alt.X("Score:Q"),
        color=alt.condition(alt.datum.Batter == top.Batter, alt.value(BLUE), alt.value("#52514e")),
        tooltip=list(res.columns)), width="stretch")
    st.dataframe(res, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%+.1f") for c in ["Score", "Phase fit", "Matchup", "Safety"]})
    with st.expander("How the score works"):
        st.markdown("""
- **Phase fit**: the batter's runs against par in each phase, weighted by how many of the remaining overs fall in that phase.
- **Matchup**: his strike-rate edge against each remaining bowler's type, plus 2.5 × his control edge on that bowler's stock lengths. Each bowler counts in proportion to his overs left.
- **Safety**: his dismissal edge against those bowler types (higher means he gets out less often than par).
- **Score** = situation × (phase fit + matchup) + (1 − situation) × 12 × safety.
""")
footer()
