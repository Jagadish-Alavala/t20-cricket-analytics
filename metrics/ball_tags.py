"""Per-ball table for the app's Batter and Bowler explorers: primer metrics plus delivery tags.

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

Also tagged: shot type, delivery variation (in-swinger, slower ball, googly...), dismissal type, over or
round the wicket (CA field or 'Over the wicket.' in the text) and whether the ball went in the air (CA only).
Wides are kept (wide = 1) so bowling economy is right; batting stats skip them.

  CRICKET_DB=/path/to/cricket.db python metrics/ball_tags.py   # writes app/data/balls.parquet
  python metrics/ball_tags.py --check                          # parser self-check, no database
"""
import os, re, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "balls.parquet"

SQL = """
SELECT b.match_id, b.innings, b.seq, b.year, b.league, b.venue, b.batting_team, b.batter_key, b.batter_name AS batter,
       b.bowler_key, b.bowler, b.runs, b.bat_runs, b.wide, b.legal, b.is_wicket, b.run_out, b.out_key, b.balls_before,
       b.wkts_before, b.raa, b.waa, b.exp_wkts, b.impact, b.wpa, b.bowl_impact, b.raa_runs,
       COALESCE(json_extract(c.raw, '$.commentary.message'), json_extract(c.raw, '$.comments[0].message')) AS msg,
       json_extract(c.raw, '$.lengthTypeId') AS ca_len, json_extract(c.raw, '$.lineTypeId') AS ca_line,
       json_extract(c.raw, '$.battingConnectionId') AS ca_conn, json_extract(c.raw, '$.bowlingTypeId') AS ca_type, json_extract(c.raw, '$.battingHandId') AS ca_bat_hand,
       json_extract(c.raw, '$.battingShotTypeId') AS ca_shot, json_extract(c.raw, '$.bowlingDetailId') AS ca_detail,
       json_extract(c.raw, '$.bowlingFromId') AS ca_from, json_extract(c.raw, '$.dismissalTypeId') AS ca_dis,
       json_extract(c.raw, '$.isAirControlled') AS ca_air, json_extract(c.raw, '$.isAirNotControlled') AS ca_airn
FROM t20_ball_metrics b JOIN commentary c ON c.match_id = b.match_id AND c.innings = b.innings AND c.seq = b.seq
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
SHOT = [("Reverse sweep", r"reverse[ -]?swe(?:ep|pt)|switch[ -]?hit|reverse scoop"), ("Slog sweep", r"slog[ -]?swe(?:ep|pt)"),
        ("Sweep", r"swe(?:ep|pt)(?!er)"), ("Scoop/ramp", r"scoop|ramp|paddle"), ("Upper cut", r"upper[ -]?cut"),
        ("Late cut", r"late[ -]?cut"), ("Cut", r"\bcut(?!ter)"), ("Hook", r"\bhook"), ("Pull", r"\bpull"),
        ("Leave", r"\bleaves?\b|left alone|shoulder(?:s|ed)? arms|no shot|lets? it go"),
        ("Slog", r"\bslog|heave|swat|swipe|hoick"), ("Drive", r"\bdriv|\bdrove|\bloft"),
        ("Flick/work", r"flick|\bwork|\bclip|\bwhip|\btuck|\bnudg"), ("Glance", r"glanc|deflect|tickle"),
        ("Steer/dab", r"steer|\bdab|\bguid"), ("Defend/push", r"defen|\bblock(?!hole)|\bpush|\bpunch|\bprod|fend")]
VARIATION = [("Googly", r"googly|wrong[ -]?'?un"), ("Doosra", r"doosra"), ("Carrom ball", r"carrom"),
             ("Arm ball", r"arm ball"), ("Top spinner", r"top[ -]?spinner"), ("Slider/flipper", r"slider|flipper"),
             ("Slower ball", r"slower|knuckle"), ("Cutter", r"cutter"), ("Reverse swing", r"reverse swing"),
             ("Swing/seam in", r"in[ -]?swing|seam in|nip(?:s|ped)? back|jag(?:s|ged)? back"),
             ("Swing/seam away", r"away[ -]?swing|out[ -]?swing|seam away|shaped? away"), ("Quicker ball", r"quicker"),
             ("Stock", r"no movement|\bstock|off break|leg spinner|leg[ -]?break|orthodox")]
CA_SHOT = {"Drive": "Drive", "ForwardDefensive": "Defend/push", "BackwardDefensive": "Defend/push", "Pushed": "Defend/push",
           "Fended": "Defend/push", "Worked": "Flick/work", "Flick": "Flick/work", "Pull": "Pull", "Hook": "Hook",
           "Cut": "Cut", "LateCut": "Late cut", "UpperCut": "Upper cut", "Leave": "Leave", "ShouldersArms": "Leave",
           "NoShot": "Leave", "Slog": "Slog", "Glance": "Glance", "Steer": "Steer/dab", "Sweep": "Sweep",
           "SlogSweep": "Slog sweep", "ReverseSweep": "Reverse sweep", "SwitchHit": "Reverse sweep",
           "Scoop": "Scoop/ramp", "ReverseScoop": "Scoop/ramp"}
CA_VAR = {"NoMovement": "Stock", "StockBall": "Stock", "OffBreak": "Stock", "LegSpinner": "Stock", "LeftArmLegspin": "Stock",
          "InSwinger": "Swing/seam in", "SeamIn": "Swing/seam in", "AwaySwinger": "Swing/seam away",
          "SeamAway": "Swing/seam away", "ReverseSwingIn": "Reverse swing", "ReverseSwingOut": "Reverse swing",
          "OffCutter": "Cutter", "LegCutter": "Cutter", "SlowerBall": "Slower ball", "QuickerBall": "Quicker ball",
          "Googly": "Googly", "ArmBall": "Arm ball", "CarromBall": "Carrom ball", "TopSpinner": "Top spinner",
          "Slider": "Slider/flipper", "Flipper": "Slider/flipper", "Doosra": "Doosra"}
DISMISSAL = [("Stumped", r"stumped"), ("Run out", r"run out"), ("LBW", r"\blbw\b|leg before"), ("Hit wicket", r"hit wicket"),
             ("Bowled", r"\bbowled\b"), ("Caught", r"\bcaught\b|\bc(?:aught)? and b")]
CA_DIS = {"Caught": "Caught", "Bowled": "Bowled", "LBW": "LBW", "Stumped": "Stumped", "RunOut": "Run out",
          "HitWicket": "Hit wicket", "CaughtAndBowled": "Caught"}
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
    """(length, line, foot, sector, style_hint, shot, variation) from one commentary sentence; None where absent."""
    b = body(msg, names)
    if not b or re.fullmatch(r"(?:no run|\d+ runs?|four|six|wide|\d+ wides?|\d+ (?:leg )?byes?)[.!]?", b):
        return (None,) * 7
    nopos = re.sub(POS_PHRASE, " ", b)
    hint = first(STYLE, nopos) or ("pace" if re.search(PACE, nopos) else None)
    return (first(LENGTH, nopos), first(LINE, nopos), first(FOOT, b), first(SECTOR_RE, b), hint,
            first(SHOT, nopos), first(VARIATION, nopos))


def check():
    T = lambda m, n=(): tag(m, n)
    assert T("FOUR! Trent Boult to Mitch Marsh. In swinger length ball, off stump on the front foot driving well timed for 4 runs") \
        == ("Good", "Stumps", "Front foot", None, "pace", "Drive", "Swing/seam in")
    assert T("Rahul Chahar to Shikhar Dhawan. Leg spinner length ball, down leg on the front foot reverse sweep to deep point "
             "for 2 runs, fielded by X") == ("Good", "Leg", "Front foot", "Point", "LB", "Reverse sweep", "Stock")
    t = T("Shoriful Islam to Fakhar Zaman. No movement back of a length, outside off on the front foot pulled to deep mid wicket for 1 run")
    assert (t[0], t[1], t[3]) == ("Back of length", "Outside off", "Mid wicket"), t
    t = T("Ben Dwarshuis to Matthew Renshaw. Short, outside off stump on the back foot pulling, top edge in the air uncontrolled past third man for 4 runs.")
    assert t[:4] == ("Short", "Outside off", "Back foot", "Third man"), t
    t = T("Kuldeep Yadav to Mahipal Lomror. Left-arm unorthodox stock length ball, middle stump on the front foot slogged to deep backward square leg for 1 run")
    assert (t[1], t[3], t[4]) == ("Stumps", "Square leg", "LWS"), t
    assert T("Chama Milind to Harpreet Singh Bhatia. 1 run") == (None,) * 7
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
    t = T("Sam Curran to Andre Russell. Off cutter full, outside off on the front foot slogged to deep mid wicket for 1 run")
    assert (t[0], t[5], t[6]) == ("Full", "Slog", "Cutter"), t
    t = T("Harshit Rana to Phil Salt. Slower ball yorker, middle stump blocked for no run", ("Phil Salt", "Harshit Rana"))
    assert (t[0], t[5], t[6]) == ("Yorker", "Defend/push", "Slower ball"), t
    t = T("Ish Sodhi to Jos Buttler. Googly length ball, outside off swept to sweeper cover for 1 run")
    assert (t[3], t[5], t[6]) == ("Cover", "Sweep", "Googly"), t
    assert first(DISMISSAL, "out! a to b. full, off stump, b bowled for 12.".lower()) == "Bowled"
    assert first(DISMISSAL, "OUT! x. B Caught for 4.".lower()) == "Caught"
    print("ball_tags check ok")


def wides(con, d, seq):
    """t20_ball_metrics drops wides (no one faces them), but bowling economy needs them. Each wide takes its
    match context and bowler from the next delivery in the same innings (the previous one if it was the last)."""
    w = pd.read_sql("""
        SELECT c.match_id, c.innings, c.seq, c.runs FROM commentary c
        WHERE c.innings IN (1, 2) AND c.match_id IN (SELECT DISTINCT match_id FROM t20_ball_metrics)
          AND (json_extract(c.raw, '$.isWide') = 1 OR c.scoring LIKE '%ʷᵈ%')""", con)
    ctx = d.assign(seq=seq, match_id=d.match_id.astype(str))[
        ["match_id", "innings", "seq", "year", "league", "venue", "batting_team", "batter_key", "batter", "bat_hand",
         "bowler_key", "bowler", "kind", "style", "balls_before", "entry_ball", "phase", "side"]].sort_values("seq")
    w = w.sort_values("seq")
    m = pd.merge_asof(w, ctx, on="seq", by=["match_id", "innings"], direction="forward")
    back = pd.merge_asof(w, ctx, on="seq", by=["match_id", "innings"], direction="backward")
    m = m.fillna(back).dropna(subset=["bowler"])
    m = m.assign(bat_runs=0, wide=1, legal=0, out=0, bowler_wkt=0, ball_no=0).drop(columns="seq")
    print(f"  + {len(m):,} wides")
    return m


def build(db):
    con = sqlite3.connect(db)
    parts = []
    for d in pd.read_sql(SQL, con, chunksize=200_000):
        t = pd.DataFrame([tag(m, (a, b)) for m, a, b in zip(d.msg, d.batter, d.bowler)], columns=["length", "line", "foot", "zone", "hint", "shot", "variation"], index=d.index)
        t["length"] = t.length.fillna(d.ca_len.map(CA_LEN))
        t["line"] = t.line.fillna(d.ca_line.map(CA_LINE))
        t["shot"] = t.shot.fillna(d.ca_shot.map(CA_SHOT))
        t["variation"] = t.variation.fillna(d.ca_detail.map(CA_VAR))
        side = d.msg.str.extract(r"(?i)\b(over|round|around) the wicket", expand=False).str.lower()
        t["side"] = side.map({"over": "Over", "round": "Round", "around": "Round"}).fillna(d.ca_from.map({"Over": "Over", "Round": "Round"}))
        dis = [first(DISMISSAL, m.lower()) if w and isinstance(m, str) else None for m, w in zip(d.msg, d.is_wicket)]
        t["dismissal"] = pd.Series(dis, index=d.index).fillna(d.ca_dis.map(CA_DIS).where(d.is_wicket.astype(bool)))
        air = pd.to_numeric(d.ca_air, errors="coerce").fillna(0) + pd.to_numeric(d.ca_airn, errors="coerce").fillna(0)
        t["aerial"] = np.where(d.ca_air.notna() | d.ca_airn.notna(), (air > 0).astype(float), np.nan)
        t["ctrl"] = np.where(d.ca_conn.isin(CA_FALSE), 0.0, np.where(d.ca_conn.isin(CA_GOOD), 1.0, np.nan))
        parts.append(pd.concat([d.drop(columns=["msg", "ca_len", "ca_line", "ca_conn", "ca_shot", "ca_detail", "ca_from", "ca_dis", "ca_air", "ca_airn"]), t], axis=1))
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
    d["aerial"] = d.aerial.where(d.aerial.groupby(d.match_id).transform("mean") > 0)  # all-zero match = not recorded

    # innings context: nth ball faced, team ball at entry, phase
    d = d.sort_values(["match_id", "innings", "seq"])
    seq = d.seq.to_numpy()  # kept aside to place the wides
    nw = d[d.wide == 0]
    g = nw.groupby(["match_id", "innings", "batter_key"], sort=False, dropna=False, observed=True)
    d["ball_no"] = (g.cumcount() + 1).reindex(d.index).fillna(0)
    d["entry_ball"] = d.groupby(["match_id", "innings", "batter_key"], dropna=False, observed=True).balls_before.transform("min") + 1
    over = d.balls_before // 6 + 1
    d["phase"] = pd.cut(over, [0, 6, 11, 16, 99], labels=["PP", "Mid1", "Mid2", "Slog"]).astype(str)
    d["out"] = (d.is_wicket.astype(bool) & (d.out_key.isna() | (d.out_key == d.batter_key))).astype("int8")
    # ponytail: a non-striker run out lands on a ball he did not face, so it is dropped here (tiny)
    d["bowler_wkt"] = (d.is_wicket.astype(bool) & ~d.run_out.astype(bool)).astype("int8")
    d["b_raa"], d["b_waa"], d["b_impact"], d["b_wpa"] = -d.raa_runs, -d.waa, -d.bowl_impact, -d.wpa  # bowler's side
    # a run out at the non-striker's end belongs to the batter run out: put it on the last ball he faced
    last = d[d.wide == 0].reset_index().groupby(["match_id", "innings", "batter_key"], observed=True)["index"].max()
    ns = d[d.is_wicket.astype(bool) & d.out_key.notna() & (d.out_key != d.batter_key) & (d.wide == 0)]
    idx = last.reindex(list(zip(ns.match_id, ns.innings, ns.out_key))).dropna().astype(int)
    d.loc[idx.values, "out"] = 1
    d.loc[idx.values, "dismissal"] = "Run out"
    d["waa"] = d.exp_wkts - d.out  # batter: his own dismissals only (the view's waa, ball by ball)

    # Players are picked by name, which joins one player's BCCI and CA ids (BCCI even uses different ids for
    # IPL and internationals). Two linked ids from the same source with one name in the same league and season are two people (two Amit
    # Mishras in SMAT 2019): those ids get their team added. ponytail: same-name players who never meet in one
    # league-season stay merged; a proper people register would fix that.
    teams = d.groupby("match_id").batting_team.agg(lambda s: set(s.dropna()))
    d["bowling_team"] = [next(iter(teams[m] - {t}), None) for m, t in zip(d.match_id, d.batting_team)]
    for who, key, team in (("batter", "batter_key", "batting_team"), ("bowler", "bowler_key", "bowling_team")):
        src = d[key].astype(str).str.split(":").str[0]
        linked = d[src != "?"]  # '?:' ids are unlinked names, never a second person
        n = linked.groupby([linked[who], src[src != "?"], linked.league, linked.year], observed=True)[key].transform("nunique")
        clash_keys = set(linked.loc[n > 1, key].dropna())
        side = d[d[key].isin(clash_keys)].groupby(key, observed=True)[team].agg(
            lambda s: s.mode().iloc[0] if s.notna().any() else "")
        lab = d[key].map(side)
        d[who] = np.where(lab.notna(), d[who].astype(str) + " (" + lab.fillna("") + ")", d[who])

    keep = ["match_id", "innings", "year", "league", "venue", "batting_team", "batter_key", "batter", "bat_hand",
            "bowler_key", "bowler", "kind", "style", "runs", "bat_runs", "wide", "legal", "out", "bowler_wkt", "dismissal",
            "balls_before", "ball_no", "entry_ball", "phase", "length", "line", "foot", "zone", "shot", "variation",
            "side", "aerial", "ctrl", "raa", "waa", "impact", "wpa", "b_raa", "b_waa", "b_impact", "b_wpa"]
    d = d[keep].reset_index(drop=True)
    d = pd.concat([d, wides(con, d, seq)], ignore_index=True)
    # charged to the bowler: off the bat, wides, and the one-run no-ball penalty (byes and leg byes are not his)
    d["conceded"] = np.where(d.wide == 1, d.runs, d.bat_runs + ((d.legal == 0) & (d.wide == 0)))
    for c in ["match_id", "league", "venue", "batting_team", "batter_key", "batter", "bat_hand", "bowler_key", "bowler",
              "kind", "style", "phase", "length", "line", "foot", "zone", "shot", "variation", "side", "dismissal"]:
        d[c] = d[c].astype("category")
    for c in ["raa", "waa", "impact", "wpa", "ctrl", "aerial", "b_raa", "b_waa", "b_impact", "b_wpa"]:
        d[c] = d[c].astype("float32")
    d = d.astype({"innings": "int8", "year": "int16", "runs": "int8", "bat_runs": "int8", "conceded": "int8", "wide": "int8", "legal": "int8",
                  "balls_before": "int16", "ball_no": "int16", "entry_ball": "int16"})
    d.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(d):,} balls, {OUT.stat().st_size / 1e6:.0f} MB)")
    print("tag coverage by league:")
    print(d.groupby("league", observed=True)[["length", "line", "zone", "shot", "variation", "side", "aerial", "ctrl", "style"]]
          .agg(lambda s: round(100 * s.notna().mean())).to_string())


if __name__ == "__main__":
    check()
    if "--check" not in sys.argv:
        build(os.environ.get("CRICKET_DB", "cricket.db"))
