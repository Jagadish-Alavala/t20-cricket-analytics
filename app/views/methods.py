import json
import altair as alt
import pandas as pd
import streamlit as st
from lib import OUT, out, footer, header, tiles

header("Methods and checks", "A number goes in front of a coach only after it has passed these checks.",
       eyebrow="Research")

meta = json.loads((OUT / "engine_meta.json").read_text())
m1, m2 = meta["model1_expected_runs"], meta["model2_win_prob"]
tiles([("Expected runs MAE, 2024 test", f"{m1['overall_MAE']}", f"naive baseline {m1['naive_MAE']}", None, None),
       ("Win model AUC (DL power law)", f"{m2['dl_law_auc']}", f"ML classifier {m2['ml_auc']}", None, None),
       ("Zone edge, year to year", "r = 0.63", "the most stable trait measured", None, None),
       ("WPA vs the primer's tables", "r = 0.997", "IPL 2026", None, None),
       ("Commentary tags vs CA's own fields", "99%", "length and line agreement", None, None),
       ("Explorer totals vs season tables", "100%", "balls, runs, wickets, RAA, Impact, WPA (batter WAA 98.8%)", None, None)], min_w=200, cols=3)

a, b = st.columns(2)
with a:
    st.markdown("**Expected-runs error by stage of innings** (trained 2020–23, tested on 2024)")
    d = pd.DataFrame(m1["by_bucket"]).melt(id_vars=["bucket", "n"], value_vars=["model_MAE", "naive_MAE"], var_name="model", value_name="MAE")
    d["model"] = d.model.map({"model_MAE": "Model", "naive_MAE": "Naive baseline"})
    st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("bucket:N", title="Balls bowled", sort=None, axis=alt.Axis(labelAngle=0)), xOffset="model:N", y=alt.Y("MAE:Q", title="Mean absolute error (runs)"),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top")), tooltip=["bucket", "model", "MAE", "n"]),
        width="stretch")
with b:
    st.markdown("**Season Impact and RAA against the published reference**")
    N = st.column_config.NumberColumn
    st.dataframe(out("reference_vs_ours.csv"), hide_index=True, width="stretch", column_config={
        "bat": "Batter", "year": N("Season", format="%d"), "imp_ref": N("Impact (ref)", format="%.1f"),
        "raa_ref": N("RAA (ref)", format="%.1f"), "waa_ref": N("WAA (ref)", format="%.2f"), "balls": N("Balls"),
        "runs": N("Runs"), "sr": N("SR", format="%.1f"), "imp": N("Impact (ours)", format="%.1f"),
        "raa": N("RAA (ours)", format="%.1f"), "waa": N("WAA (ours)", format="%.2f")})

st.markdown("""
### How it is built
- **Tactics data**: about 459,000 deliveries with line, length, shot and control tags, IPL plus seven leagues, 2020–2024. Matchups are measured against par for the same phase and zone, then shrunk toward zero for small samples.
- **Metrics data**: 5,123 men's T20 matches and 1.1 million deliveries. It has par scores by league, season and ground (nested shrinkage), a Duckworth-Lewis Pro run curve, xgboost baselines for runs and wickets above average, and win probability P = rⁿ/(1+rⁿ) from the DL score ratio. Definitions are from Himanish Ganjoo's *T20 Metrics: A Primer*.
- **Commentary tags** (Batter and Bowler explorers, local only): length, line, footwork, shot, delivery type, shot area and dismissal type are read from the commentary sentence. On Cricket Australia balls the text agrees with CA's own length and line fields 99% of the time. Control, over/round the wicket and "in the air" use CA's records only, and only matches where they are recorded on nearly every ball. Every explorer total reconciles with the season tables.
- **Which metric to trust**: six impact metrics were tested on whether they predict winning and whether players repeat them. Runs above average against a simple baseline won both.

### Honesty notes
- No single batter-against-bowler matchups: the median pair has only 5 balls. Everything is at bowler-type level.
- Matchup edges hold at spell and season level. They do not predict a single delivery.
- The batting-order tool and field plans are transparent heuristics, not proven optima: a captain's counterfactual can't be tested.
- The Impact Player study is observational. The SMAT on-off case is the strongest evidence; the other leagues differ in more than the rule.

### Credits
- Metric definitions: Himanish Ganjoo, [*T20 Metrics: A Primer*](https://hganjoo.github.io/t20basics/).
- Data: BCCI (bcci.tv), Cricket Australia (cricket.com.au), [Cricsheet](https://cricsheet.org) (ODC-BY), Wikidata (CC0).
- Code: [github.com/Jagadish-Alavala/t20-cricket-analytics](https://github.com/Jagadish-Alavala/t20-cricket-analytics)
""")
footer()
