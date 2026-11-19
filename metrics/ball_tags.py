"""Per-ball batting table for the app's Batter explorer: primer metrics plus delivery tags.

Joins t20_ball_metrics (run t20_metrics.py first) to the commentary and tags each ball with
length, line, footwork and shot area, plus control (1 = in control, 0 = false shot: edged, beaten, mistimed,
hit on the pad). Control comes only from Cricket Australia's structured shot-connection field; BCCI text
flags a false shot on only 2-3% of balls (a real rate is nearer 25-30%), so BCCI balls get no control value.
CA matches where the field is filled on under 90% of balls are dropped too: there it is filled mostly on
notable balls, which pushes control down to ~34%. Fully recorded matches (T20 Blast, some T20Is) give ~70%. The tags come from the
commentary sentence ("In swinger length ball, off stump on the front foot driving well timed to
deep cover..."), with Cricket Australia's structured fields used where the text has none.
Two sentence styles are handled: the older 'Bowler to Batter. <delivery>, <line> ...' and the newer
'Kohli drives a length ball outside off for six'. On Cricket Australia balls the text agrees with CA's
own length and line fields 99% of the time. Bowler style (pace, OB, LB, SLA, LWS) is the bowler's
majority over all his balls; the pace bowler's arm is left out because CA's hand field is unreliable.

  CRICKET_DB=/path/to/cricket.db python metrics/ball_tags.py   # writes app/data/balls.parquet
  python metrics/ball_tags.py --check                          # parser self-check, no database
"""
import os, re, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "balls.parquet"

SQL = """
SELECT b.match_id, b.innings, b.seq, b.year, b.league, b.venue, b.batting_team, b.batter_key, b.batter_name AS batter,
       b.bowler_key, b.bowler, b.bat_runs, b.is_wicket, b.out_key, b.balls_before, b.wkts_before,
       b.raa, b.waa, b.impact, b.wpa,
       COALESCE(json_extract(c.raw, '$.commentary.message'), json_extract(c.raw, '$.comments[0].message')) AS msg,
       json_extract(c.raw, '$.lengthTypeId') AS ca_len, json_extract(c.raw, '$.lineTypeId') AS ca_line,
       json_extract(c.raw, '$.battingConnectionId') AS ca_conn, json_extract(c.raw, '$.bowlingTypeId') AS ca_type, json_extract(c.raw, '$.battingHandId') AS ca_bat_hand
FROM t20_ball_metrics b JOIN commentary c ON c.match_id = b.match_id AND c.innings = b.innings AND c.seq = b.seq
WHERE NOT b.wide
"""

# first match wins, so the specific phrase sits before the general one
LENGTH = [("Full toss", r"full[ -]?toss|beamer"), ("Yorker", r"yorker"),
          ("Back of length", r"back of (?:a )?length|short of (?:a )?(?:good )?length"),
          ("Full", r"half[ -]?volley|\bfull\b|over-?pitched"), ("Good", r"\blength\b"),
          ("Short", r"\bshort\b|half[ -]?tracker|long hop|bouncer")]
LINE = [("Wide off", r"wide outside off|wide of off"), ("Outside off", r"outside off"),
        ("Wide leg", r"wide down leg|wide down the leg"), ("Leg", r"down leg|down the leg|on the pads|into the pads"),
        ("Stumps", r"off stump|middle stump|leg stump|at the stumps|on the stumps|middle and (?:off|leg)")]
FOOT = [("Down the track", r"down the track|down the pitch|charg|skip"), ("Back foot", r"back foot|backs away|deep in (?:the )?crease"),
        ("Front foot", r"front foot")]
CA_FALSE = {"MisTimed", "Missed", "Edged", "HitPad", "MissedLegSide", "InsideEdge", "PlayAndMissLegSide", "ThickEdge",
            "OutsideEdge", "HitBody", "TopEdge", "BottomEdge", "LeadingEdge", "Gloved", "Padded", "Spliced", "BatPad",
            "HitHelmet", "PlayAndMiss", "FalseShot"}
CA_GOOD = {"Played", "HitHard", "WellTimed", "Middled", "Left", "ShouldersArms", "NoShot"}  # a leave counts as control
CA_LEN = {"FullToss": "Full toss", "Beamer": "Full toss", "Yorker": "Yorker", "HalfVolley": "Full", "LengthBall": "Good",
          "BackOfALength": "Back of length", "Short": "Short", "HalfTracker": "Short"}
CA_LINE = {"OutsideOff": "Outside off", "Wide": "Wide off", "OffStump": "Stumps", "MiddleStump": "Stumps",
           "LegStump": "Stumps", "DownLeg": "Leg", "WideDownLeg": "Wide leg"}
# fielding positions -> the 8 scoring sectors used by the field tool (clockwise from straight, off side first)
SECTOR = [("Third man", r"third man|wicket-?keeper|keeper|slips?\b|gully|fly slip"),
          ("Point", r"point"), ("Cover", r"cover|sweeper"),
          ("Long off", r"mid[ -]?off|long[ -]?off|bowler|straight"),
          ("Long on", r"mid[ -]?on|long[ -]?on"), ("Mid wicket", r"mid[ -]?wicket|cow corner"),
          ("Square leg", r"square leg|short leg|leg gully"), ("Fine leg", r"fine leg|long leg|leg slip")]
POS_WORD = r"\b(?:to|over|past|through|towards|wide of|at|in front of|behind)\s+(?:the\s+)?(?:[a-z ]+? at\s+)?"
SECTOR_RE = [(n, POS_WORD + r"(?:deep |short |wide |fine |square |backward |forward |silly |long |extra )*(?:" + p + ")")
             for n, p in SECTOR]
STYLE = [("LWS", r"unorthodox|chinaman|left-?arm (?:leg|wrist)"), ("SLA", r"left-?arm orthodox|orthodox|arm ball"),
         ("LB", r"leg ?spin|googly|flipper|top ?spinner|slider|wrong 'un"), ("OB", r"off ?break|off ?spin|doosra|carrom")]
PACE = r"no movement|swing|seam|cutter|slower|quicker|\bfast\b|medium"
# fielding positions that contain length words ("short third man", "long leg"): blanked before the length search
POS_PHRASE = r"\b(?:deep|short|long|fine|square|backward|forward|silly|extra|wide)(?:\s+(?:backward|forward|square|fine|extra))*" \
             r"\s+(?:third man|fine leg|square leg|leg|mid[ -]?(?:wicket|on|off)|cover|point|on|off|slip)\b"
NOISE = r"run[ -]?out (?:attempt|chance).*|winviz.*|\d+ runs in the over.*"
PREFIX = r"^(?:[A-Za-z]+!\s*|Wide\.\s*|No ball\.\s*)*(?:[A-Z][\w'. -]{1,40}? to [A-Z][\w'. -]{1,40}?\.\s+)?" \
         r"(?:(?:Over|Round|Around) the wicket\.\s*)?"


def first(pats, text):
    for name, p in pats:
        if re.search(p, text):
            return name
    return None


def body(msg, names=()):
    """Delivery description, lower case: drops 'FOUR! Bowler to Batter.' prefixes and the players' own names
    (so a batter called Short doesn't read as a short ball)."""
    m = re.sub(PREFIX, "", msg if isinstance(msg, str) else "")
    for n in names:
        if isinstance(n, str) and n:
            m = re.sub(r"\b" + re.escape(n) + r"\b", " ", m, flags=re.I)
            m = re.sub(r"(?<=\S )" + re.escape(n.split()[-1]) + r"\b", " ", m)  # capitalised surname mid-sentence
    return re.sub(NOISE, "", m.lower()).strip()


def tag(msg, names=()):
    """(length, line, foot, sector, style_hint) from one commentary sentence; None where absent."""
    b = body(msg, names)
    if not b or re.fullmatch(r"(?:no run|\d+ runs?|four|six|wide|\d+ wides?|\d+ (?:leg )?byes?)[.!]?", b):
        return (None,) * 5
    nopos = re.sub(POS_PHRASE, " ", b)
    hint = first(STYLE, nopos) or ("pace" if re.search(PACE, nopos) else None)
    return first(LENGTH, nopos), first(LINE, nopos), first(FOOT, b), first(SECTOR_RE, b), hint


def check():
    T = lambda m, n=(): tag(m, n)
    assert T("FOUR! Trent Boult to Mitch Marsh. In swinger length ball, off stump on the front foot driving well timed for 4 runs") \
        == ("Good", "Stumps", "Front foot", None, "pace")
    assert T("Rahul Chahar to Shikhar Dhawan. Leg spinner length ball, down leg on the front foot reverse sweep to deep point "
             "for 2 runs, fielded by X") == ("Good", "Leg", "Front foot", "Point", "LB")
    t = T("Shoriful Islam to Fakhar Zaman. No movement back of a length, outside off on the front foot pulled to deep mid wicket for 1 run")
    assert (t[0], t[1], t[3]) == ("Back of length", "Outside off", "Mid wicket"), t
    t = T("Ben Dwarshuis to Matthew Renshaw. Short, outside off stump on the back foot pulling, top edge in the air uncontrolled past third man for 4 runs.")
    assert t[:4] == ("Short", "Outside off", "Back foot", "Third man"), t
    t = T("Kuldeep Yadav to Mahipal Lomror. Left-arm unorthodox stock length ball, middle stump on the front foot slogged to deep backward square leg for 1 run")
    assert (t[1], t[3], t[4]) == ("Stumps", "Square leg", "LWS"), t
    assert T("Chama Milind to Harpreet Singh Bhatia. 1 run") == (None,) * 5
    # newer sentence style, names inline
    assert T("Kohli drives a length ball outside off for six.  Timing, intent and execution.", ("Virat Kohli", "Mohit Sharma"))[:2] \
        == ("Good", "Outside off")
    t = T("Length ball on off stump. Kohli goes onto the back foot and pushes back to the bowler.  No run.", ("Virat Kohli", "David Miller"))
    assert t[:4] == ("Good", "Stumps", "Back foot", "Long off"), t
    t = T("Kohli pushes a length ball outside off to extra cover. Run out attempt by Miller but missed the stumps.", ("Virat Kohli", "David Miller"))
    assert (t[0], t[1], t[3]) == ("Good", "Outside off", "Cover"), t
    # a batter called Short, and a fielding position that says 'short'
    t = T("Short slogs a full ball to short third man.", ("Matt Short", "Adam Zampa"))
    assert (t[0], t[3]) == ("Full", "Third man"), t
    print("ball_tags check ok")


def build(db):
    con = sqlite3.connect(db)
    parts = []
    for d in pd.read_sql(SQL, con, chunksize=200_000):
        t = pd.DataFrame([tag(m, (a, b)) for m, a, b in zip(d.msg, d.batter, d.bowler)], columns=["length", "line", "foot", "zone", "hint"], index=d.index)
        t["length"] = t.length.fillna(d.ca_len.map(CA_LEN))
        t["line"] = t.line.fillna(d.ca_line.map(CA_LINE))
        t["ctrl"] = np.where(d.ca_conn.isin(CA_FALSE), 0.0, np.where(d.ca_conn.isin(CA_GOOD), 1.0, np.nan))
        parts.append(pd.concat([d.drop(columns=["msg", "ca_len", "ca_line", "ca_conn"]), t], axis=1))
        print(f"  {sum(len(p) for p in parts):,} balls", flush=True)
    d = pd.concat(parts, ignore_index=True)

    # bowler style: majority per bowler name over all his balls (names bridge BCCI and CA ids)
    ca_style = d.ca_type.map({"OffSpin": "OB", "LegSpin": "LB", "Orthodox": "SLA", "Unorthodox": "LWS",
                              "FastSeam": "pace", "Medium": "pace"})
    style = ca_style.fillna(d.hint).groupby(d.bowler).agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
    pb = pd.read_sql("SELECT 'B:' || player_id AS k, bowling_type, batting_hand FROM players WHERE source='B'", con)
    s = d.bowler.map(style)
    spin = s.isin(["OB", "LB", "SLA", "LWS"])
    d["kind"] = np.where(spin, "spin", np.where(s.eq("pace"), "pace", d.bowler_key.map(pb.set_index("k").bowling_type)))
    d["style"] = np.where(spin, s, np.where(d.kind == "pace", "Pace", None))
    bh = d.ca_bat_hand.where(d.ca_bat_hand.isin(["Left", "Right"])).groupby(d.batter_key).agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else None)
    d["bat_hand"] = d.batter_key.map(pb.set_index("k").batting_hand.str.title().replace("", np.nan)).fillna(d.batter_key.map(bh))

    cov = d.ctrl.notna().groupby(d.match_id).transform("mean")
    d["ctrl"] = d.ctrl.where(cov >= 0.9)

    # innings context: nth ball faced, team ball at entry, phase
    d = d.sort_values(["match_id", "innings", "seq"])
    g = d.groupby(["match_id", "innings", "batter_key"], sort=False, dropna=False, observed=True)
    d["ball_no"] = g.cumcount() + 1
    d["entry_ball"] = g.balls_before.transform("min") + 1
    over = d.balls_before // 6 + 1
    d["phase"] = pd.cut(over, [0, 6, 11, 16, 99], labels=["PP", "Mid1", "Mid2", "Slog"]).astype(str)
    d["out"] = (d.is_wicket.astype(bool) & (d.out_key.isna() | (d.out_key == d.batter_key))).astype("int8")
    # ponytail: a non-striker run out lands on a ball he did not face, so it is dropped here (tiny)

    keep = ["match_id", "innings", "year", "league", "venue", "batting_team", "batter_key", "batter", "bat_hand",
            "bowler", "kind", "style", "bat_runs", "out", "balls_before", "ball_no", "entry_ball", "phase",
            "length", "line", "foot", "zone", "ctrl", "raa", "waa", "impact", "wpa"]
    d = d[keep].reset_index(drop=True)
    for c in ["match_id", "league", "venue", "batting_team", "batter_key", "batter", "bat_hand", "bowler", "kind", "style",
              "phase", "length", "line", "foot", "zone"]:
        d[c] = d[c].astype("category")
    for c in ["raa", "waa", "impact", "wpa", "ctrl"]:
        d[c] = d[c].astype("float32")
    d = d.astype({"innings": "int8", "year": "int16", "bat_runs": "int8", "balls_before": "int16", "ball_no": "int16",
                  "entry_ball": "int16"})
    d.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(d):,} balls, {OUT.stat().st_size / 1e6:.0f} MB)")
    print("tag coverage by league:")
    print(d.groupby("league", observed=True)[["length", "line", "zone", "ctrl", "style"]]
          .agg(lambda s: round(100 * s.notna().mean())).to_string())


if __name__ == "__main__":
    check()
    if "--check" not in sys.argv:
        build(os.environ.get("CRICKET_DB", "cricket.db"))
