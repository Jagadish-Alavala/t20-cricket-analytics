# T20 Cricket Analytics

Analytics for coaching staff and franchise analysts, built from ball-by-ball data. Before a game, it shows how to bowl to a batter and where to put the field. During a game, it suggests who should walk in next. For retention and auction calls, it shows which players actually add runs and wins.

Two datasets feed it:
- **Tactics data:** about 459,000 deliveries with line, length, shot and control tags, IPL plus seven leagues, 2020–2024.
- **Metrics data:** 5,123 men's T20 matches, 1.1 million deliveries, IPL 2008–2026, plus internationals, SMAT, T20 Blast, BBL, CPL and other leagues.

## Start here (no code needed)

| You are… | Open |
|---|---|
| A coach planning for a batter | [Field-setting tool](https://jagadish-alavala.github.io/t20-cricket-analytics/outputs/field_setting_tool.html): weak zones, scoring map and a bowl-here, field-there plan for 77 batters |
| A franchise analyst or GM | [What the Impact Player rule changed](https://jagadish-alavala.github.io/t20-cricket-analytics/stories/impact_player.html): IPL against control leagues, 2019–2026 |
| A scout | `scouting_app/`: Streamlit leaderboards, player inspector and a line-and-length matrix |

## Questions it answers

| Question in the room | Module | Output |
|---|---|---|
| How do we bowl to him, and where do the fielders go? | `tactics/field_plan.py`, `build_field_tool.py` | `outputs/field_setting_tool.html` |
| Which bowler type troubles him, in which phase, on which length? | `core/matchup_engine.py`, `tactics/zone_matchup.py` | `matchup_batter_vs_kind_phase.csv`, `zone_batter_length.csv`, `zone_batter_line.csv` |
| Who walks in next, and which bowler should he target? | `tactics/order_optimizer_v2.py`, `entry_points.py` | `entry_batters.csv`, `entry_bowlers.csv` |
| Where does each batter add value: powerplay, middle or death? | `tactics/phase_profile.py` | `batter_phase_profile.csv` |
| Who adds runs and wins, adjusted for match situation? | `metrics/t20_metrics.py`, `t20_report.py` | player-season Impact, RAA, WAA and WPA for every IPL season since 2008 |
| Which value metric can we trust? | `player_value/` | `outputs/Impact_Analytics_Findings.docx` |
| Did a rule change alter how teams play? | `metrics/impact_player.py` | `stories/impact_player.html` |

```
core/          shared loader, match-state features, expected runs, DL resource, phase par
tactics/       match-day decisions: matchups, field plans, entry points, batting order
scouting_app/  Streamlit scouting app and the RAAR model behind it
metrics/       contextual player value (par, Impact, RAA/WAA, WPA) and the Impact Player study
player_value/  which impact metrics hold up: validity and repeatability tests
stories/       written-up studies
outputs/       generated tables, the field tool, charts and documents
```

---

## tactics/: match-day decisions

A connected system, where each piece feeds the next.

- **`zone_matchup.py` and `core/matchup_engine.py`:** batter weaknesses by bowler type, phase, line and length. Control % is the main signal. A batter's zone control edge repeats year to year at r = 0.63, the most stable trait measured here. The engine recovers known matchups from data alone: Dhoni weak against spin, Klaasen strong against spin, Williamson slower against pace.
- **`phase_profile.py`:** where each batter adds value by phase.
- **`entry_points.py`:** the over and wickets-down at which each batter usually arrives, and each bowler's phase window and typical first over.
- **`order_optimizer_v2.py`:** given the match state and the bowlers with overs left, it ranks the batters still to come and names the bowler each should target. Treat it as advice, not an optimum: a captain's counterfactual can't be validated.
- **`field_plan.py` and `build_field_tool.py`:** turn a batter's weak zones and scoring map into a bowl-here, field-there plan. The interactive version also has an optional catch map of dismissals.

Run order: `core/matchup_engine.py`, then `phase_profile.py`, `zone_matchup.py`, `entry_points.py`, `build_field_tool.py`, `order_optimizer_v2.py`.

## scouting_app/: player scouting

- **`raar_model.py`:** builds RAAR (runs above average, risk-adjusted) on LightGBM expected-runs and wicket models with match context: phase, wickets in hand, required rate and balls faced.
- **`evaluate_raar_model.py`:** tests the model on a rolling origin (train up to year N, test on year N+1) and measures how much shrinkage moves the rankings.
- **`app.py`:** the Streamlit front end, with batting and bowling leaderboards, a player inspector (against pace and spin, by phase, by scoring area) and a line-and-length matrix (wicket %, economy, risk against reward, best deliveries).

## metrics/: contextual player value

The metric definitions come from Himanish Ganjoo, *[T20 Metrics: A Primer](https://hganjoo.github.io/t20basics/)*. The implementation and the fits are my own, on this data.
- **Par scores:** nested shrinkage by league, season and ground.
- **Run curve:** a Duckworth-Lewis Pro run-production curve.
- **Impact:** the change in projected total per ball.
- **RAA and WAA:** runs and wickets above average, against xgboost baselines.
- **Win probability:** P = rⁿ/(1+rⁿ) from the DL score ratio, then win probability added (WPA) and leverage.

The implementation agrees with the primer's published IPL 2026 tables: WPA correlation 0.997, and RAA within about one run per 100 balls.

| Script | Purpose |
|---|---|
| `t20_metrics.py` | Builds every metric into the database |
| `t20_report.py` | Batting and bowling reports by date range, league or player-season |
| `t20_validate.py` | Checks against the primer's worked examples and published tables |
| `impact_player.py` | Difference in differences: IPL against the T20 Blast and CPL, with SMAT as an on/off case, bootstrapped over matches |

**Impact Player finding:** IPL top orders got faster (top-3 RAA up about 7 per 100 balls relative to the control leagues) but not more valuable: there was no change in WPA.

## player_value/: which metrics hold up

The headline result is a negative one: simpler metrics beat sophisticated ones on the tests that matter.
- The expected-runs model comes within 3–4% MAE of a commercial vendor's model (18.8 against 18.3).
- The DL power-law win probability beat a LightGBM classifier out of sample (AUC 0.857 against 0.76).
- Six impact metrics were judged on whether they predict winning and whether players repeat them. RAA against a simple over-by-over baseline won both.

The full write-up is `outputs/Impact_Analytics_Findings.docx`.

---

## Honesty notes

- I don't build individual batter-against-bowler matchups: the median pair has only 5 balls. Everything is at bowler-type level.
- Matchup edges are reliable at spell or season level, not for predicting a single delivery.
- The order optimizer and field plans are advisory heuristics, and all their inputs are exposed.

## Data and running it

No raw data and no data-collection code are in this repository. `outputs/` holds derived, aggregated tables only.

```bash
pip install -r requirements.txt

# tactics, player_value: folder with t20_2020.csv … t20_2024.csv
export T20_DATA=/path/to/data
python tactics/zone_matchup.py

# scouting app
cd scouting_app && T20_BBB=/path/to/t20_bbb.csv python raar_model.py && streamlit run app.py

# metrics: local SQLite database of ball-by-ball commentary
export CRICKET_DB=/path/to/cricket.db
python metrics/t20_metrics.py
python metrics/t20_report.py --seasons --league IPL --min-balls 100
python metrics/impact_player.py
```

## Credits

- **Metric definitions:** Himanish Ganjoo, *[T20 Metrics: A Primer](https://hganjoo.github.io/t20basics/)*.
- **Data for the metrics:**
  - BCCI (bcci.tv);
  - Cricket Australia (cricket.com.au);
  - Cricsheet ([cricsheet.org](https://cricsheet.org), ODC-BY) for IPL 2008–2011;
  - Wikidata (CC0) for player names.
- **Author:** Jagadish Alavala.
