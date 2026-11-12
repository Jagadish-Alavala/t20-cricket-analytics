"""Field-setting engine (first cut): per-batter 'bowl here, field there' plan.

Combines two validated signals:
  - weak zones (line & length by kind, control-edge)  -> where to bowl to induce the false shot
  - scoring map (wagon zones, run share)               -> where they score, so where to post fielders
Output is a scouting card: attack the weakness, protect the strengths.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import numpy as np, pandas as pd
import engine as E, matchup_engine as M

ZONE = {1: "deep fine leg", 2: "deep square leg", 3: "deep mid-wicket", 4: "long-on",
        5: "long-off", 6: "deep cover", 7: "deep point", 8: "third man"}
LEN = {"SHORT": "short", "SHORT_OF_A_GOOD_LENGTH": "back-of-a-length", "GOOD_LENGTH": "good length",
       "FULL": "full", "YORKER": "yorker", "FULL_TOSS": "full toss"}
LINE = {"OUTSIDE_OFFSTUMP": "outside off", "ON_THE_STUMPS": "at the stumps", "DOWN_LEG": "down leg",
        "WIDE_OUTSIDE_OFFSTUMP": "wide outside off", "WIDE_DOWN_LEG": "wide down leg"}

length_prof = pd.read_csv(E.OUT / "zone_batter_length.csv")
line_prof = pd.read_csv(E.OUT / "zone_batter_line.csv")
d = M.load_balls()
d = d[(d.batruns > 0) & (d.wagonZone.between(1, 8))]
wagon = d.groupby(["bat", "wagonZone"])["batruns"].sum()

def weakest(prof, dim, bat, kind, min_balls=40):
    s = prof[(prof.bat == bat) & (prof.kind == kind) & (prof.balls >= min_balls)]
    if s.empty: return None
    r = s.sort_values("ctrl_edge").iloc[0]
    return r[dim], round(r.ctrl_edge, 1), int(r.balls)

def card(bat):
    print(f"\n{'='*66}\n  SCOUTING & FIELD PLAN:  {bat}\n{'='*66}")
    for kind in ["pace", "spin"]:
        wl = weakest(length_prof, "length", bat, kind)
        wn = weakest(line_prof, "line", bat, kind)
        if wl or wn:
            bits = []
            if wl: bits.append(f"{LEN.get(wl[0], wl[0])} (control {wl[1]:+.0f})")
            if wn: bits.append(f"{LINE.get(wn[0], wn[0])} (control {wn[1]:+.0f})")
            print(f"  vs {kind.upper():5s} -> bowl: " + ", ".join(bits))
    w = wagon.loc[bat] if bat in wagon.index.get_level_values(0) else None
    if w is not None:
        share = (100 * w / w.sum()).sort_values(ascending=False)
        strong = [f"{ZONE[int(z)]} {int(v)}%" for z, v in share.head(3).items()]
        weak = [ZONE[int(z)] for z, v in share.tail(3).items()]
        print(f"  PROTECT (top scoring): " + ", ".join(strong))
        print(f"  can leave / post catcher: " + ", ".join(weak))

for b in ["Suryakumar Yadav", "Rohit Sharma", "MS Dhoni", "Virat Kohli", "Heinrich Klaasen"]:
    card(b)
print("\n(Weak zones from control-edge, scoring zones from wagon data. Descriptive, reliable at aggregate.)")
