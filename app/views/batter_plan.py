import altair as alt
import pandas as pd
import streamlit as st
from lib import (out, batters, field_tool_html, PHASES, PHASE_ORDER, ZONE_LENGTHS, NICE, footer, header, tiles,
                 signed, PACE_SPIN, SURFACE, DIV_MID)

header("Plan for a batter",
       "Where he is weak, in plain words, plus the field to set. Edges are against par for the same phase or zone "
       "and shrunk toward zero on small samples; positive is good for the batter. Line-and-length data, 2020–2024.",
       eyebrow="Match day")

names = batters()
bat = st.selectbox("Batter", names, index=names.index("Virat Kohli") if "Virat Kohli" in names else 0)

kind = out("matchup_batter_vs_kind.csv").query("bat == @bat")
kp = out("matchup_batter_vs_kind_phase.csv").query("bat == @bat")
zl = out("zone_batter_length.csv").query("bat == @bat")
zn = out("zone_batter_line.csv").query("bat == @bat")
ent = out("entry_batters.csv").query("bat == @bat")
prof = out("batter_phase_profile.csv").query("bat == @bat")
sty = out("matchup_batter_vs_style.csv").query("bat == @bat")

items = []
if len(ent):
    e = ent.iloc[0]
    items += [("Usual role", e.slot, "", None, None),
              ("Typical entry", f"Over {e.entry_over:.0f}", f"{e.entry_wkts:.0f} wickets down", None, None)]
if len(prof):
    items.append(("Best phase", PHASES[prof.iloc[0].best_phase].split(" (")[0], PHASES[prof.iloc[0].best_phase].split(" (")[1][:-1], None, None))
for k in ["pace", "spin"]:
    r = kind[kind.kind == k]
    if len(r):
        items.append((f"vs {k} · strike-rate edge", f"{r.iloc[0].sr_edge:+.0f}", f"safety edge {r.iloc[0].dis_edge:+.1f}",
                      signed(r.iloc[0].sr_edge), "Strike rate against par for the same phase; safety = how much less often "
                                                  "he gets out than par (positive is good for him)"))
if items:
    tiles(items, min_w=150)

# ---- the plan in words
plan = []
if len(kp):
    w = kp.sort_values("sr_edge").iloc[0]
    if w.sr_edge < 0:
        plan.append(f"**Bowl {w.kind} in the {PHASES[w.phase].split(' (')[0].lower()}**: he scores "
                    f"{-w.sr_edge:.0f} strike-rate points below par there ({w.balls:.0f} balls).")
    s = kp.sort_values("sr_edge").iloc[-1]
    if s.sr_edge > 0:
        plan.append(f"**Avoid {s.kind} in the {PHASES[s.phase].split(' (')[0].lower()}**: "
                    f"{s.sr_edge:+.0f} strike rate against par.")
for z, col in ((zl, "length"), (zn, "line")):
    if len(z):
        w = z.sort_values("ctrl_edge").iloc[0]
        if w.ctrl_edge <= -1:
            plan.append(f"**{w.kind.title()}, {NICE(w[col]).lower()}** ({col}): his control drops "
                        f"{-w.ctrl_edge:.1f} points below par (strike rate {w.sr:.0f}).")
if len(sty):
    w = sty.sort_values("sr_edge").iloc[0]
    if w.sr_edge < 0:
        plan.append(f"**Bowling style {w.bowl_style}** troubles him most: {w.sr_edge:+.0f} strike rate against par.")
with st.container(border=True):
    st.markdown("#### The plan")
    st.markdown("\n".join(f"- {p}" for p in plan) if plan else "Not enough balls to suggest a plan.")

# ---- charts
t1, t2, t3 = st.tabs(["By phase", "Line and length", "Field plan"])
with t1:
    if len(kp):
        d = kp.assign(Phase=kp.phase.map(lambda p: PHASES[p].split(" (")[0]))
        ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Phase:N", sort=[PHASES[p].split(" (")[0] for p in PHASE_ORDER], title=None),
            xOffset="kind:N",
            y=alt.Y("sr_edge:Q", title="Strike-rate edge against par"),
            color=alt.Color("kind:N", scale=alt.Scale(**PACE_SPIN), title=None),
            tooltip=["kind", "Phase", "balls", "runs", "outs", "sr", "sr_par", "sr_edge", "dis_edge"])
        st.altair_chart(ch, width="stretch")
    if len(prof):
        p = prof.iloc[0]
        st.caption("Value added by phase (runs against par per 100 balls): " + " · ".join(
            f"{PHASES[k].split(' (')[0]} {p[f'edge_{k}']:+.0f} ({p[f'balls_{k}']:.0f} balls)" for k in PHASE_ORDER if pd.notna(p[f'edge_{k}'])))
with t2:
    a, b = st.columns(2)
    for col, z, field, order in ((a, zl, "length", ["FULL_TOSS"] + ZONE_LENGTHS),
                                  (b, zn, "line", ["WIDE_OUTSIDE_OFFSTUMP", "OUTSIDE_OFFSTUMP", "ON_THE_STUMPS", "DOWN_LEG"])):
        with col:
            st.markdown(f"**Control edge by {field}**: blue means he is less in control there, so bowl it")
            if len(z):
                d = z.assign(zone=z[field].map(NICE))
                ch = alt.Chart(d).mark_rect(cornerRadius=3, stroke=SURFACE, strokeWidth=2).encode(
                    x=alt.X("kind:N", title=None), y=alt.Y("zone:N", sort=[NICE(o) for o in order], title=None),
                    color=alt.Color("ctrl_edge:Q", scale=alt.Scale(range=["#3987e5", DIV_MID, "#e66767"], domainMid=0, interpolate="lab"),
                                    title="Control edge"),
                    tooltip=["kind", "zone", "balls", "runs", "sr", "ctrl", "ctrl_par", "ctrl_edge", "outs"])
                txt = alt.Chart(d).mark_text(fontWeight="bold").encode(
                    x=alt.X("kind:N"), y=alt.Y("zone:N", sort=[NICE(o) for o in order]),
                    text=alt.Text("ctrl_edge:Q", format="+.1f"), color=alt.value("#ffffff"))
                st.altair_chart(ch + txt, width="stretch")
            else:
                st.caption("Not enough balls.")
with t3:
    html = field_tool_html()
    if f'"{bat}"' in html:
        html = html.replace('NAMES.includes("Rohit Sharma")?"Rohit Sharma"', f'NAMES.includes({bat!r})?{bat!r}'.replace("'", '"'))
    else:
        st.info(f"{bat} is not in the field-tool set (77 batters with enough zone data). Showing the default.")
    st.iframe(html, height=900)

with st.expander("Raw tables"):
    st.dataframe(kp, hide_index=True)
    st.dataframe(zl, hide_index=True)
    st.dataframe(zn, hide_index=True)
st.caption("Bowler-type level only. Single batter-against-bowler pairs are too thin to trust (the median pair has 5 balls).")
footer()
