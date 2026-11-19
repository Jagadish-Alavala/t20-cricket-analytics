import altair as alt
import pandas as pd
import streamlit as st
from lib import out, batters, bowler_kind, matchup_vs_bowler, NICE, LENGTHS, footer

st.title("🎳 Use a bowler")
st.caption("When he usually bowls, what he bowls, and which batters he should be bowling at.")

mix = out("zone_bowler_length_mix.csv")
ent = out("entry_bowlers.csv")
kinds = bowler_kind()
names = sorted(kinds)
bo = st.selectbox("Bowler", names, index=names.index("Jasprit Bumrah") if "Jasprit Bumrah" in names else 0)
kind = kinds[bo]

e = ent[ent.bowl == bo]
c = st.columns(4)
c[0].metric("Type", kind.title())
if len(e):
    e = e.iloc[0]
    c[1].metric("Role", e.role)
    c[2].metric("Usual first over", f"{e.first_over:.0f}")
    c[3].metric("Overs per match", f"{e.overs_per_match:.1f}")

a, b = st.columns(2)
with a:
    st.markdown("**Where his overs go** (% of balls)")
    if len(e):
        d = pd.DataFrame({"Phase": ["Powerplay", "Middle", "Death"], "share": [e.PP, e.MID, e.DEATH]})
        st.altair_chart(alt.Chart(d).mark_bar(color="#22c55e").encode(
            x=alt.X("Phase:N", sort=None, title=None), y=alt.Y("share:Q", title="% of balls"), tooltip=["Phase", "share"]),
            width="stretch")
with b:
    st.markdown("**Length mix**")
    m = mix[(mix.bowl == bo) & (mix.kind == kind)].assign(Length=lambda x: x.length.map(NICE), pct=lambda x: 100 * x.share)
    st.altair_chart(alt.Chart(m).mark_bar(color="#38bdf8").encode(
        y=alt.Y("Length:N", sort=[NICE(l) for l in LENGTHS + ["FULL_TOSS"]], title=None),
        x=alt.X("pct:Q", title="% of balls"), tooltip=["Length", alt.Tooltip("pct:Q", format=".1f")]),
        width="stretch")

rows = []
for bat in batters():
    sr, ctrl, _ = matchup_vs_bowler(bat, bo, kinds)
    rows.append({"Batter": bat, f"SR edge vs {kind}": sr, "Control edge on his lengths": ctrl, "Matchup": sr + 2.5 * ctrl})
df = pd.DataFrame(rows).round(1)
df = df[(df[f"SR edge vs {kind}"] != 0) | (df["Control edge on his lengths"] != 0)]
x, y = st.columns(2)
with x:
    st.markdown("#### ✅ Bowl him at")
    st.dataframe(df.nsmallest(12, "Matchup"), hide_index=True, width="stretch")
with y:
    st.markdown("#### ⚠️ Keep him away from")
    st.dataframe(df.nlargest(12, "Matchup"), hide_index=True, width="stretch")
st.caption("Matchup = the batter's strike-rate edge against this bowler type, plus 2.5 × his control edge on this "
           "bowler's lengths, weighted by how often he bowls each one. Negative favours the bowler.")
footer()
