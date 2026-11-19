import altair as alt
import streamlit as st
from lib import data, footer, header

header("Compare players", "Career totals and season-by-season value, side by side, for up to five players.",
       eyebrow="Squad and auction")

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
    N = st.column_config.NumberColumn
    st.dataframe(car.sort_values("wpa", ascending=False), width="stretch", column_config={
        "seasons": N("Seasons"), "balls": N("Balls"), "runs": N("Runs" if role == "Batting" else "Runs conceded"),
        "raa": N("RAA", format="%+.1f"), "waa": N("WAA", format="%+.1f"), "impact": N("Impact", format="%+.1f"),
        "wpa": N("WPA", format="%+.2f"), "RAA/100": N(format="%+.2f"), "WPA/100": N(format="%+.3f")})
    names_ = {"wpa": "WPA", "raa": "RAA", "raa_100": "RAA per 100 balls", "impact": "Impact"}
    metric = st.radio("Season metric", list(names_), format_func=names_.get, horizontal=True)
    s = d.groupby(["player", "year"], as_index=False)[["wpa", "raa", "impact", "balls"]].sum()
    s["raa_100"] = (100 * s.raa / s.balls).round(1)
    st.altair_chart(alt.Chart(s).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
        x=alt.X("year:O", title="Season"), y=alt.Y(f"{metric}:Q", title=names_[metric]),
        color=alt.Color("player:N", title=None, sort=pick, legend=alt.Legend(orient="top")),
        tooltip=["player", "year", alt.Tooltip(f"{metric}:Q", format="+.2f"), "balls"]).properties(height=360),
        width="stretch")
else:
    st.caption("Pick at least one player.")
footer()
