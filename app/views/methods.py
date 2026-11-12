import json
import altair as alt
import pandas as pd
import streamlit as st
from lib import OUT, out, footer

st.title("🔬 Methods and checks")
st.caption("A number goes in front of a coach only after it has passed these checks.")

meta = json.loads((OUT / "engine_meta.json").read_text())
m1, m2 = meta["model1_expected_runs"], meta["model2_win_prob"]
c = st.columns(4)
c[0].metric("Expected runs MAE (2024 test)", m1["overall_MAE"], f"naive {m1['naive_MAE']}", delta_color="off")
c[1].metric("Win model AUC: DL power law", m2["dl_law_auc"], f"ML classifier {m2['ml_auc']}", delta_color="off")
c[2].metric("Zone edge, year-to-year r", "0.63")
c[3].metric("WPA vs primer tables", "r = 0.997")

a, b = st.columns(2)
with a:
    st.markdown("**Expected-runs error by stage of innings** (trained 2020–23, tested on 2024)")
    d = pd.DataFrame(m1["by_bucket"]).melt(id_vars=["bucket", "n"], value_vars=["model_MAE", "naive_MAE"], var_name="model", value_name="MAE")
    st.altair_chart(alt.Chart(d).mark_bar().encode(x=alt.X("bucket:N", title="Balls bowled", sort=None), xOffset="model:N",
                    y="MAE:Q", color=alt.Color("model:N", title=None), tooltip=["bucket", "model", "MAE", "n"]), use_container_width=True)
with b:
    st.markdown("**Season Impact and RAA against the published reference**")
    st.dataframe(out("reference_vs_ours.csv"), hide_index=True, use_container_width=True)

st.markdown("""
### How it is built
- **Tactics data**: about 459,000 deliveries with line, length, shot and control tags, IPL plus seven leagues, 2020–2024. Matchups are measured against par for the same phase and zone, then shrunk toward zero for small samples.
- **Metrics data**: 5,123 men's T20 matches and 1.1 million deliveries. It has par scores by league, season and ground (nested shrinkage), a Duckworth-Lewis Pro run curve, xgboost baselines for runs and wickets above average, and win probability P = rⁿ/(1+rⁿ) from the DL score ratio. Definitions are from Himanish Ganjoo's *T20 Metrics: A Primer*.
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
