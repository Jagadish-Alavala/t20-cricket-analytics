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
