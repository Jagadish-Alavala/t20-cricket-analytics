import altair as alt
import pandas as pd
import streamlit as st
from lib import out, batters, bowler_kind, matchup_vs_bowler, NICE, ZONE_LENGTHS, footer, header, tiles

header("Use a bowler", "When he usually bowls, what he bowls, and which batters he should be bowling at. "
       "Line-and-length data, 2020–2024.", eyebrow="Match day")

mix = out("zone_bowler_length_mix.csv")
ent = out("entry_bowlers.csv")
kinds = bowler_kind()
names = sorted(kinds)
bo = st.selectbox("Bowler", names, index=names.index("Jasprit Bumrah") if "Jasprit Bumrah" in names else 0)
kind = kinds[bo]

e = ent[ent.bowl == bo]
items = [("Type", kind.title(), "", None, None)]
if len(e):
    e = e.iloc[0]
    items += [("Role", e.role, "", None, None), ("Usual first over", f"{e.first_over:.0f}", "", None, None),
              ("Overs per match", f"{e.overs_per_match:.1f}", "", None, None)]
tiles(items, min_w=150)

a, b = st.columns(2)
with a:
    st.markdown("**Where his overs go** (% of balls)")
    if len(e):
        d = pd.DataFrame({"Phase": ["Powerplay", "Middle", "Death"], "share": [e.PP, e.MID, e.DEATH]})
        st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Phase:N", sort=None, title=None), y=alt.Y("share:Q", title="% of balls"), tooltip=["Phase", "share"]),
            width="stretch")
with b:
    st.markdown("**Length mix**")
    m = mix[(mix.bowl == bo) & (mix.kind == kind)].assign(Length=lambda x: x.length.map(NICE), pct=lambda x: 100 * x.share)
    st.altair_chart(alt.Chart(m).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Length:N", sort=[NICE(l) for l in ["FULL_TOSS"] + ZONE_LENGTHS], title=None),
        x=alt.X("pct:Q", title="% of balls"), tooltip=["Length", alt.Tooltip("pct:Q", format=".1f")]),
        width="stretch")

rows = []
for bat in batters():
    sr, ctrl, _ = matchup_vs_bowler(bat, bo, kinds)
    rows.append({"Batter": bat, f"SR edge vs {kind}": sr, "Control edge on his lengths": ctrl, "Matchup": sr + 2.5 * ctrl})
df = pd.DataFrame(rows).round(1)
df = df[(df[f"SR edge vs {kind}"] != 0) | (df["Control edge on his lengths"] != 0)]
CFG = {c: st.column_config.NumberColumn(format="%+.1f") for c in df.columns if c != "Batter"}
x, y = st.columns(2)
with x:
    st.markdown("#### Bowl him at")
    st.caption("The batters this bowler's type and lengths trouble most")
    st.dataframe(df.nsmallest(12, "Matchup"), hide_index=True, width="stretch", column_config=CFG)
with y:
    st.markdown("#### Keep him away from")
    st.caption("The batters who score most freely against what he bowls")
    st.dataframe(df.nlargest(12, "Matchup"), hide_index=True, width="stretch", column_config=CFG)
st.caption("Matchup = the batter's strike-rate edge against this bowler type, plus 2.5 × his control edge on this "
           "bowler's lengths, weighted by how often he bowls each one. Negative favours the bowler.")
footer()
