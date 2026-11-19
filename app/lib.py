"""Shared loaders and the matchup / batting-order logic used by the app.

The app reads only the derived tables in outputs/ and app/data/, so it runs without any raw data.
The scoring in `recommend` follows tactics/order_optimizer_v2.py.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "outputs", Path(__file__).parent / "data"
PHASES = {"PP": "Powerplay (1-6)", "MID": "Middle (7-15)", "DEATH": "Death (16-20)"}
PHASE_ORDER = list(PHASES)
LENGTHS = ["YORKER", "FULL", "GOOD_LENGTH", "SHORT_OF_A_GOOD_LENGTH", "SHORT"]
NICE = lambda s: str(s).replace("_", " ").title()


@st.cache_data
def out(name):
    return pd.read_csv(OUT / name)


@st.cache_data
def data(name, **kw):
    return pd.read_csv(DATA / name, **kw)


@st.cache_data
def field_tool_html():
    return (OUT / "field_setting_tool.html").read_text()


@st.cache_data
def story_html():
    return (ROOT / "stories" / "impact_player.html").read_text()


def batters():
    names = set(out("matchup_batter_vs_kind.csv").bat) | set(out("zone_batter_length.csv").bat)
    return sorted(names)


def bowler_kind():
    m = out("zone_bowler_length_mix.csv")
    return m.groupby("bowl")["kind"].agg(lambda s: s.mode().iloc[0]).to_dict()


def _val(df, a, b):
    try:
        v = df.loc[a, b]
        return float(v) if pd.notna(v) else 0.0
    except KeyError:
        return 0.0


@st.cache_data
def _tables():
    mk = out("matchup_batter_vs_kind.csv")
    zbl = out("zone_batter_length.csv")
    return (out("batter_phase_profile.csv").set_index("bat"),
            mk.pivot_table(index="bat", columns="kind", values="sr_edge"),
            mk.pivot_table(index="bat", columns="kind", values="dis_edge"),
            {(r.bat, r.kind, r.length): r.ctrl_edge for r in zbl.itertuples()},
            out("zone_bowler_length_mix.csv"))


def matchup_vs_bowler(bat, bowler, kinds=None):
    """Batter's projected edge against one bowler: pace/spin strike-rate edge plus the batter's
    control edge on that bowler's stock lengths (weighted by how often he bowls them)."""
    prof, mk_sr, mk_dis, ctrl, mix = _tables()
    kind = (kinds or bowler_kind()).get(bowler)
    if kind is None:
        return 0.0, 0.0, None
    m = mix[(mix.bowl == bowler) & (mix.kind == kind)]
    pairs = [(r.share, ctrl[(bat, kind, r.length)]) for r in m.itertuples() if (bat, kind, r.length) in ctrl]
    den = sum(s for s, _ in pairs)
    c = sum(s * e for s, e in pairs) / den if den else 0.0
    return _val(mk_sr, bat, kind), c, kind


def phase_weights(over_done):
    overs = list(range(over_done + 1, 21)) or [20]
    w = {p: 0 for p in PHASE_ORDER}
    for o in overs:
        w["PP" if o <= 6 else "MID" if o <= 15 else "DEATH"] += 1
    t = sum(w.values())
    return {k: v / t for k, v in w.items()}


def recommend(candidates, over_done, bowlers, need):
    """bowlers: list of (name, overs_left). need: 1 = must attack, 0 = protect the wicket."""
    prof, mk_sr, mk_dis, _, _ = _tables()
    kinds, pw = bowler_kind(), phase_weights(over_done)
    bowlers = [(b, o) for b, o in bowlers if o > 0]
    tot = sum(o for _, o in bowlers) or 1
    rows = []
    for bat in candidates:
        pf = sum(pw[p] * _val(prof, bat, f"edge_{p}") for p in PHASE_ORDER)
        m_sr = m_c = safe = 0.0
        detail = []
        for bo, ov in bowlers:
            w = ov / tot
            sr, c, kind = matchup_vs_bowler(bat, bo, kinds)
            m_sr, m_c = m_sr + w * sr, m_c + w * c
            safe += w * (_val(mk_dis, bat, kind) if kind else 0)
            detail.append((bo, sr + 2.5 * c))
        upside = pf + m_sr + 2.5 * m_c
        best = max(detail, key=lambda d: d[1]) if detail else ("-", 0)
        worst = min(detail, key=lambda d: d[1]) if detail else ("-", 0)
        rows.append({"Batter": bat, "Score": need * upside + (1 - need) * safe * 12,
                     "Phase fit": pf, "Matchup": m_sr + 2.5 * m_c, "Safety": safe,
                     "Target": best[0], "Careful against": worst[0]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True).round(1)


def footer():
    st.divider()
    st.caption("Metric definitions: Himanish Ganjoo, *T20 Metrics: A Primer*. "
               "Data: BCCI, Cricket Australia, Cricsheet (ODC-BY), Wikidata (CC0). Built by Jagadish Alavala.")


@st.cache_data(show_spinner="Loading ball-by-ball data…")
def balls():
    """Per-ball batting table from metrics/ball_tags.py, or None when it hasn't been built."""
    p = DATA / "balls.parquet"
    return pd.read_parquet(p) if p.exists() else None


def tiles(items, min_w=120):
    """Responsive stat strip: items are (label, value, sub, colour, tooltip); tiles wrap instead of truncating."""
    from html import escape
    cells = "".join(
        f"<div title='{escape(tip or '')}' style='background:#13263a;border:1px solid #22384f;border-radius:10px;padding:10px 12px'>"
        f"<div style='font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;opacity:.7'>{escape(label)}"
        f"{' ⓘ' if tip else ''}</div>"
        f"<div style='font-size:1.55rem;font-weight:700;color:{colour or '#e5e7eb'};line-height:1.25'>{escape(value)}</div>"
        f"<div style='font-size:.78rem;opacity:.65'>{escape(sub or '')}</div></div>"
        for label, value, sub, colour, tip in items)
    st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax({min_w}px,1fr));gap:8px;"
                f"margin:4px 0 12px'>{cells}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- explorer helpers (batter and bowler pages)
LENGTHS = ["Full toss", "Yorker", "Full", "Good", "Back of length", "Short"]
LINES = ["Wide off", "Outside off", "Stumps", "Leg", "Wide leg"]
SECTORS = ["Third man", "Point", "Cover", "Long off", "Long on", "Mid wicket", "Square leg", "Fine leg"]
VARIATIONS = ["Stock", "Swing/seam in", "Swing/seam away", "Reverse swing", "Slower ball", "Cutter", "Quicker ball",
              "Googly", "Arm ball", "Top spinner", "Slider/flipper", "Carrom ball", "Doosra"]
EXPLORER_PHASES = {"PP": "Powerplay (1-6)", "Mid1": "Middle 1 (7-11)", "Mid2": "Middle 2 (12-16)", "Slog": "Slog (17-20)"}


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rank_tiles(field, who, cols, label):
    """Rank `who` in `field` (one row per player, higher is better) on each (name, column) pair."""
    import numpy as np  # noqa: F401
    items = []
    for k, col in cols:
        rk = int(field[col].rank(ascending=False, method="min")[who])
        pct = round(100 * (1 - (rk - 1) / len(field)))
        items.append((f"{k} per ball", f"#{rk}", f"of {len(field):,} · {ordinal(pct)} percentile",
                      "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 40 else "#f87171", None))
    st.markdown(f"**Rankings** among the {len(field):,} {label} under the same filters")
    tiles(items, min_w=170)


def heatmap(cells, metric, fmt, mid, good_high=True, height=330, scheme=None):
    """Line x length grid. cells: one row per (length, line) with the metric column; red = good for the batter."""
    import altair as alt
    scale = alt.Scale(scheme=scheme) if scheme else alt.Scale(scheme="redblue", reverse=good_high, domainMid=mid)
    base = alt.Chart(cells).encode(x=alt.X("line:N", sort=LINES, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=0, labelOverlap=False, labelFontSize=11)),
                                   y=alt.Y("length:N", sort=LENGTHS, title=None))
    tips = [c for c in cells.columns if c not in ("length", "line")]
    st.altair_chart((base.mark_rect().encode(color=alt.Color(f"{metric}:Q", scale=scale), tooltip=["length", "line"] + tips)
                     + base.mark_text(fontSize=13, fontWeight="bold").encode(text=alt.Text(f"{metric}:Q", format=fmt),
                                                                             color=alt.value("#0b1622")))
                    .properties(height=height), width="stretch")


def wheel(z, left_handed, title=None):
    """Runs by scoring sector, seen from behind the batter with the bowler at the top."""
    import altair as alt
    import numpy as np
    g = z.groupby("zone", observed=True).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                             outs=("out", "sum")).reindex(SECTORS).fillna(0).reset_index()
    g["share"] = 100 * g.runs / max(g.runs.sum(), 1)
    order = SECTORS if left_handed else SECTORS[::-1]  # a right-hander's off side is on the right
    g["theta0"] = (g.zone.map({s: i for i, s in enumerate(order)}) * 45 - 180) * np.pi / 180
    g["theta1"], g["mid"] = g.theta0 + np.pi / 4, g.theta0 + np.pi / 8
    g["label"] = g.zone.astype(str) + " " + g.share.round(0).astype(int).astype(str) + "%"
    base = alt.Chart(g).properties(width=340, height=340, title=title or "")
    arcs = base.mark_arc(innerRadius=38, outerRadius=130, stroke="#0b1622", strokeWidth=2).encode(
        theta=alt.Theta("theta0:Q", scale=None), theta2="theta1:Q",
        color=alt.Color("share:Q", scale=alt.Scale(scheme="goldorange"), legend=None),
        tooltip=["zone", "runs", alt.Tooltip("share:Q", format=".1f", title="% of runs"), "balls", "outs"])
    text = base.mark_text(radius=152, fontSize=11, color="#e5e7eb").encode(theta=alt.Theta("mid:Q", scale=None), text="label:N")
    st.altair_chart(arcs + text, width="content")
    return g
