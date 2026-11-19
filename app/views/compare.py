import altair as alt
import streamlit as st
from lib import data, footer

st.title("⚖️ Compare players")
st.caption("Career totals and season-by-season value, side by side.")

c = st.columns([1, 1, 3])
role = c[0].radio("Role", ["Batting", "Bowling"], horizontal=True)
df = data("batting_seasons.csv" if role == "Batting" else "bowling_seasons.csv")
leagues = ["All"] + sorted(df.league.unique())
lg = c[1].selectbox("League", leagues, index=leagues.index("IPL"))
if lg != "All":
    df = df[df.league == lg]
pool = df.groupby("player").balls.sum().sort_values(ascending=False)
names = list(pool.index)
default = [n for n in (["Virat Kohli", "Shreyas Iyer", "Suryakumar Yadav"] if role == "Batting"
                       else ["Jasprit Bumrah", "Sunil Narine", "Rashid Khan"]) if n in names]
pick = c[2].multiselect("Players (up to 5)", names, default=default, max_selections=5)

if pick:
    d = df[df.player.isin(pick)]
    car = d.groupby("player").agg(seasons=("year", "nunique"), balls=("balls", "sum"), runs=("runs", "sum"),
                                  raa=("raa", "sum"), waa=("waa", "sum"), impact=("impact", "sum"), wpa=("wpa", "sum"))
    car["RAA/100"] = 100 * car.raa / car.balls
    car["WPA/100"] = 100 * car.wpa / car.balls
    st.dataframe(car.sort_values("wpa", ascending=False).round(2), width="stretch")
    metric = st.radio("Season metric", ["wpa", "raa", "raa_100", "impact"], horizontal=True)
    s = d.groupby(["player", "year"], as_index=False)[["wpa", "raa", "impact", "balls"]].sum()
    s["raa_100"] = (100 * s.raa / s.balls).round(1)
    st.altair_chart(alt.Chart(s).mark_line(point=True).encode(
        x=alt.X("year:O", title="Season"), y=alt.Y(f"{metric}:Q"), color=alt.Color("player:N", title=None),
        tooltip=["player", "year", metric, "balls"]), width="stretch")
else:
    st.info("Pick at least one player.")
footer()
