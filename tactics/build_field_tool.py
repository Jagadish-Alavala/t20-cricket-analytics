"""Compute per-batter wagon scoring map + weak zones, emit interactive dark split-sector
field tool. Labels sit OUTSIDE the boundary (no overlap); keeper/bowler inside the ring."""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core"))
import json, numpy as np, pandas as pd
import engine as E, matchup_engine as M

LEN = {"SHORT":"short","SHORT_OF_A_GOOD_LENGTH":"back-of-a-length","GOOD_LENGTH":"good length",
       "FULL":"full","YORKER":"yorker","FULL_TOSS":"full toss"}
LINE = {"OUTSIDE_OFFSTUMP":"outside off","ON_THE_STUMPS":"at the stumps","DOWN_LEG":"down leg",
        "WIDE_OUTSIDE_OFFSTUMP":"wide outside off","WIDE_DOWN_LEG":"wide down leg"}
lp = pd.read_csv(E.OUT/"zone_batter_length.csv"); ln = pd.read_csv(E.OUT/"zone_batter_line.csv")
d = M.load_balls()
hand = d.groupby("bat")["bat_hand"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "RHB")
sc = d[(d.batruns>0)&(d.wagonZone.between(1,8))]
wag = sc.groupby(["bat","wagonZone"])["batruns"].sum()
outs = d[(d.out==True)&(d.wagonZone.between(1,8))].groupby(["bat","wagonZone"]).size()
balls = d.groupby("bat")["ballfaced"].sum()

def weak(prof,dim,bat,kind):
    s=prof[(prof.bat==bat)&(prof.kind==kind)&(prof.balls>=40)]
    if s.empty: return None
    r=s.sort_values("ctrl_edge").iloc[0]; return {"zone":r[dim],"edge":round(float(r.ctrl_edge),1)}
def fmt(a,b):
    p=[]
    if a: p.append(f"{LEN.get(a['zone'],a['zone'])} ({a['edge']:+.0f})")
    if b: p.append(f"{LINE.get(b['zone'],b['zone'])} ({b['edge']:+.0f})")
    return ", ".join(p) if p else "no clear weak length/line"

players={}
for bat in balls[balls>=300].index:
    if bat not in wag.index.get_level_values(0): continue
    w=wag.loc[bat]; share={int(z):round(float(100*v/w.sum()),1) for z,v in w.items()}
    for z in range(1,9): share.setdefault(z,0.0)
    npc=weak(ln,"line",bat,"pace")
    ov=outs.loc[bat] if bat in outs.index.get_level_values(0) else pd.Series(dtype=int)
    players[bat]={"hand":hand.get(bat,"RHB"),"wagon":share,
                  "outs":{int(z):int(c) for z,c in ov.items()},
                  "plan_pace":fmt(weak(lp,"length",bat,"pace"),npc),
                  "plan_spin":fmt(weak(lp,"length",bat,"spin"),weak(ln,"line",bat,"spin")),
                  "weak_dir_pace":(npc or {}).get("zone","")}
data=json.dumps(players); names=json.dumps(sorted(players.keys()))
print("batters in tool:",len(players))

HTML=r"""<div id="fs" style="font-family:system-ui,Arial;max-width:560px;margin:auto;background:#0d1a2b;color:#e5e7eb;padding:14px;border-radius:12px">
<div style="display:flex;gap:10px;align-items:center;margin-bottom:6px">
<label style="font-weight:600">Batter</label>
<select id="sel" style="padding:6px 10px;font-size:15px;background:#16263a;color:#e5e7eb;border:1px solid #33475f;border-radius:6px"></select>
<span id="hand" style="color:#94a3b8;font-size:13px"></span>
<label style="margin-left:auto;font-size:13px;color:#cbd5e1;cursor:pointer"><input type="checkbox" id="showOuts"> show dismissals</label></div>
<svg id="fld" viewBox="0 0 520 560" style="width:100%"></svg>
<div style="display:flex;gap:16px;font-size:12px;margin:2px 0 4px 6px">
<span><span style="color:#ffd23f">&#9679;</span> boundary rider</span>
<span><span style="color:#4ade80">&#9679;</span> saver / deep</span>
<span><span style="color:#f87171">&#9679;</span> catcher (weak zone)</span></div>
<div id="plan" style="margin-top:2px;font-size:13.5px;line-height:1.5"></div>
<div style="margin-top:5px;color:#7c8aa0;font-size:11px">Sector shade &amp; outside % = where the batter scores. WK/B = keeper/bowler. IPL 2020-24, aggregate.</div></div>
<script>
const P=__DATA__, NAMES=__NAMES__;
const ZA={1:-157.5,2:-112.5,3:-67.5,4:-22.5,5:22.5,6:67.5,7:112.5,8:157.5};
const ZN={1:"fine leg",2:"square leg",3:"mid-wkt",4:"long-on",5:"long-off",6:"cover",7:"point",8:"third man"};
const CX=260, CY=270, R=190;
function P2(a,f){a=a*Math.PI/180;return [CX+Math.sin(a)*f*R, CY-Math.cos(a)*f*R];}
function draw(name){
 const p=P[name], mir=(p.hand==="LHB")?-1:1, w=p.wagon, mx=Math.max(...Object.values(w));
 let s=`<circle cx=${CX} cy=${CY} r=${R} fill="#0f1620" stroke="#fff" stroke-width=3/>`;
 for(let z=1;z<=8;z++){const a=ZA[z]*mir,a0=(a-22.5)*Math.PI/180,a1=(a+22.5)*Math.PI/180;
   const x0=CX+Math.sin(a0)*R,y0=CY-Math.cos(a0)*R,x1=CX+Math.sin(a1)*R,y1=CY-Math.cos(a1)*R;
   s+=`<path d="M${CX} ${CY} L${x0.toFixed(1)} ${y0.toFixed(1)} A${R} ${R} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)} Z" fill="#f2b807" opacity=${(0.05+w[z]/mx*0.5).toFixed(2)}/>`;}
 for(let k=0;k<8;k++){const t=k*45*Math.PI/180;s+=`<line x1=${CX} y1=${CY} x2=${(CX+Math.sin(t)*R).toFixed(1)} y2=${(CY-Math.cos(t)*R).toFixed(1)} stroke="#fff" stroke-width=1 opacity=.35/>`;}
 s+=`<circle cx=${CX} cy=${CY} r=${R*0.55} fill="none" stroke="#fff" stroke-width=1 stroke-dasharray="4 4" opacity=.32/>`;
 // OUTSIDE labels: zone name + run%
 for(let z=1;z<=8;z++){const[x,y]=P2(ZA[z]*mir,1.17);
   s+=`<text x=${x.toFixed(0)} y=${(y-3).toFixed(0)} font-size=11 fill="#fff" text-anchor="middle">${ZN[z]}</text>`;
   s+=`<text x=${x.toFixed(0)} y=${(y+11).toFixed(0)} font-size=11 font-weight=700 fill="#ffd23f" text-anchor="middle">${w[z]}%</text>`;}
 s+=`<rect x=${CX-10} y=${CY-36} width=20 height=72 rx=2 fill="#c9a86a" opacity=.9/>`;
 // keeper + bowler inside 30y ring
 const kp=P2(0,0.40), bw=P2(180,0.40);
 s+=`<circle cx=${CX} cy=${(CY+0.40*R).toFixed(0)} r=9 fill="#e2e8f0"/><text x=${CX} y=${(CY+0.40*R+3.5).toFixed(0)} font-size=8.5 font-weight=700 fill="#0d1a2b" text-anchor="middle">WK</text>`;
 s+=`<circle cx=${CX} cy=${(CY-0.40*R).toFixed(0)} r=9 fill="#e2e8f0"/><text x=${CX} y=${(CY-0.40*R+3.5).toFixed(0)} font-size=8.5 font-weight=700 fill="#0d1a2b" text-anchor="middle">B</text>`;
 const zs=Object.keys(w).map(Number).sort((a,b)=>w[b]-w[a]), F=[];
 zs.slice(0,3).forEach(z=>F.push({a:ZA[z]*mir,f:0.9,c:"#ffd23f"}));
 zs.slice(3,5).forEach(z=>F.push({a:ZA[z]*mir,f:0.72,c:"#4ade80"}));
 zs.slice(5,8).forEach(z=>F.push({a:ZA[z]*mir,f:0.46,c:"#4ade80"}));
 const wl={"OUTSIDE_OFFSTUMP":75,"ON_THE_STUMPS":10,"DOWN_LEG":-80,"WIDE_OUTSIDE_OFFSTUMP":110}[p.weak_dir_pace];
 if(wl!==undefined)F.push({a:wl*mir,f:0.30,c:"#f87171"});
 // dismissal dots (catch-map layer), placed by zone with deterministic jitter
 if(document.getElementById("showOuts").checked && p.outs){
   for(let z=1;z<=8;z++){const n=p.outs[z]||0;
     for(let i=0;i<n;i++){const aj=ZA[z]*mir + (((i*67+z*31)%35)-17); const rf=0.44+(((i*53+z*19)%50)/100);
       const[x,y]=P2(aj,rf); s+=`<circle cx=${x.toFixed(1)} cy=${y.toFixed(1)} r=3.2 fill="#ff6b6b" stroke="#fff" stroke-width=0.4 opacity=0.85/>`;}}}
 F.forEach(f=>{const[x,y]=P2(f.a,f.f);s+=`<circle cx=${x.toFixed(0)} cy=${y.toFixed(0)} r=9 fill=${f.c} stroke="#0d1a2b" stroke-width=1.5/>`;});
 document.getElementById("fld").innerHTML=s;
 document.getElementById("hand").textContent=p.hand;
 document.getElementById("plan").innerHTML=`<b style="color:#f2b807">Bowling plan</b> &nbsp; vs pace: <span style="color:#fca5a5">${p.plan_pace}</span><br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; vs spin: <span style="color:#fca5a5">${p.plan_spin}</span>`;
}
const sel=document.getElementById("sel");
NAMES.forEach(n=>{const o=document.createElement("option");o.value=o.textContent=n;sel.appendChild(o);});
sel.value=NAMES.includes("Rohit Sharma")?"Rohit Sharma":NAMES[0];
sel.onchange=()=>draw(sel.value); document.getElementById("showOuts").onchange=()=>draw(sel.value); draw(sel.value);
</script>"""
HTML=HTML.replace("__DATA__",data).replace("__NAMES__",names)
open(E.OUT/"field_setting_tool.html","w").write(HTML)
print("wrote field_setting_tool.html")
