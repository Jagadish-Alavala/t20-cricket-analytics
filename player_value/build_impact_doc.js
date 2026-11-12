const fs = require("fs");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType } = d;

const NAVY="1F3864", ACCENT="2E5A88", GREY="595959", LIGHT="EAF0F7", RED="B23A3A", GREEN="2E7D46";
const H1=(t)=>new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:260,after:120},children:[new TextRun({text:t,bold:true,color:NAVY,size:30})]});
const H2=(t)=>new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:170,after:80},children:[new TextRun({text:t,bold:true,color:ACCENT,size:24})]});
const P=(runs,o={})=>new Paragraph({spacing:{after:120,line:276},...o,children:Array.isArray(runs)?runs:[new TextRun({text:runs,size:21})]});
const T=(t,o={})=>new TextRun({text:t,size:21,...o});
const bullet=(runs)=>new Paragraph({bullet:{level:0},spacing:{after:70,line:268},children:Array.isArray(runs)?runs:[new TextRun({text:runs,size:21})]});

function table(headers,rows,widths,hi){
  const tot=widths.reduce((a,b)=>a+b,0);
  const hr=new TableRow({tableHeader:true,children:headers.map((t,i)=>new TableCell({width:{size:widths[i],type:WidthType.DXA},shading:{type:ShadingType.CLEAR,fill:NAVY},margins:{top:40,bottom:40,left:70,right:70},children:[new Paragraph({children:[new TextRun({text:t,bold:true,color:"FFFFFF",size:17})]})]}))});
  const br=rows.map((r,ri)=>new TableRow({children:r.map((c,i)=>{
    let color="222222",bold=(i===0);
    if(hi&&hi.col===i){ if(String(c).startsWith("best")||hi.best===ri){} }
    return new TableCell({width:{size:widths[i],type:WidthType.DXA},shading:{type:ShadingType.CLEAR,fill:ri%2?"FFFFFF":LIGHT},margins:{top:34,bottom:34,left:70,right:70},children:[new Paragraph({children:[new TextRun({text:String(c),size:17,bold})]})]});
  })}));
  return new Table({columnWidths:widths,width:{size:tot,type:WidthType.DXA},rows:[hr,...br],
    borders:["top","bottom","left","right","insideHorizontal","insideVertical"].reduce((o,k)=>{o[k]={style:BorderStyle.SINGLE,size:2,color:"D0D7E2"};return o;},{})});
}

const doc=new Document({creator:"Jagadish Alavala",styles:{default:{document:{run:{font:"Calibri",size:21,color:"222222"}}}},sections:[{
  properties:{page:{size:{width:12240,height:15840},margin:{top:1100,bottom:1100,left:1200,right:1200}}},
  children:[
    new Paragraph({spacing:{after:40},children:[new TextRun({text:"Cricket Impact Analytics: what actually measures first-innings value",bold:true,color:NAVY,size:36})]}),
    new Paragraph({spacing:{after:200},border:{bottom:{style:BorderStyle.SINGLE,size:8,color:NAVY,space:6}},children:[new TextRun({text:"An evaluation of impact metrics against winning.  IPL + 7 leagues, 458,746 balls, test season 2024.",color:GREY,size:19})]}),

    H1("Summary"),
    P([T("The goal was a first-innings batting impact metric better than runs and strike rate. We built and tested six: DL closed-form impact, runs above average (RAA), an ML expected-runs impact, venue-adjusted impact, an empirical resource-curve impact, and a state-model RAA. Each was judged on two axes that matter: "),T("validity",{bold:true}),T(" (does it predict winning) and "),T("reliability",{bold:true}),T(" (does a player repeat it). The result was decisive and against intuition.")]),
    bullet([T("The simplest metric won. ",{bold:true}),T("Runs above the over-by-over league average (RAA) had the highest validity (AUC 0.660) and the best reliability of the context metrics. Every more sophisticated version scored lower on one or both axes.")]),
    bullet([T("DL impact is dominated. ",{bold:true}),T("It is less reliable than strike rate (0.35 vs 0.46) and no more predictive of winning than raw runs (0.625 vs 0.626). It should be retired for first-innings value.")]),
    bullet([T("Sophistication added noise, not signal, three times. ",{bold:true}),T("The DL power-law beat the ML win-probability model, the fixed DL curve beat the refit, and the over-average RAA beat the gradient-boosted RAA. In low signal-to-noise T20 data, flexible models overfit faster than they de-bias.")]),
    bullet([T("The expected-runs model is sound. ",{bold:true}),T("It matches a commercial vendor's model to within 3 to 4 percent (MAE 18.8 vs 18.3), so the foundation is not the weak link.")]),
    bullet([T("Settled metric set: ",{bold:true}),T("RAA plus strike rate for first-innings batting, WPA (via the DL power-law) for chases. Two currencies, reported separately, never fused under invented weights.")]),

    H1("1. How each metric was judged"),
    P([T("Validity is measured at the batter-innings level (first innings, minimum 10 balls, 1,450 innings): can the metric predict whether the batting team won the match, by AUC. Reliability is split-half: a player's per-ball metric in a random half of their innings versus the other half. A good metric is both, high validity means it tracks winning, high reliability means it measures the player and not the day.")]),

    H1("2. The verdict"),
    H2("Validity: AUC predicting the batting team's win"),
    table(["Metric","AUC","Reads"],[
      ["RAA (runs above over-average)","0.660","best. simple, exogenous baseline"],
      ["Strike rate","0.657","strong rate metric"],
      ["Raw runs","0.626","volume baseline"],
      ["DL closed-form impact","0.625","no better than raw runs"],
      ["DL impact per ball","0.617","worse than strike rate"],
      ["RAA (state ML model)","0.579","the fancy version, lower"],
    ],[4200,1200,3300]),
    H2("Reliability: does the player repeat it"),
    table(["Metric","Split-half r","Reads"],[
      ["Strike rate","0.46","most repeatable"],
      ["DL impact (global curve)","0.35","noisier than strike rate"],
      ["DL impact (empirical refit)","0.16","refit made it worse"],
    ],[4200,1500,3000]),
    P([T("The two tables point the same way. The metric with the best validity is a simple over-average RAA, and the sophisticated impacts are both less reliable and no more valid.")],{spacing:{before:100}}),

    H1("3. Why simple keeps winning"),
    P([T("A flexible model splits the expectation into many small cells, this over, this wicket count, this venue, and estimates each from too little data, so its expected value is noisier than a smooth average. One specific trap surfaced clearly: an early state-model RAA conditioned on the current score and run rate, which are caused by the batter's own scoring. The baseline absorbed the very contribution RAA is meant to isolate, and validity collapsed to a coin flip (0.501). Removing those endogenous features lifted it to 0.579, still below the simple 0.660. The lesson is to condition the baseline only on what the batter cannot cause: over, wickets in hand, venue.")]),

    H1("4. The expected-runs model is not the weak link"),
    table(["Expected final-score model","MAE","R squared"],[
      ["Naive run-rate projection","28.22","-0.50"],
      ["Ours (LightGBM)","18.81","0.499"],
      ["Data-provider predscore","18.26","0.569"],
    ],[4600,1900,2200]),
    P([T("Our from-scratch model lands within half a run of MAE of a commercial model in every phase. Early-innings error near 28 is the floor, not a fault: first-innings totals have a standard deviation of 37.8, so at ball one no model beats predicting the average.")],{spacing:{before:100}}),

    H1("5. Settled leaderboards, IPL 2024"),
    H2("First-innings batting, by RAA (runs above over-average)"),
    table(["Batter","Inns","Balls","Runs","SR","RAA"],[
      ["Travis Head","10","231","433","187.4","103.9"],
      ["Sunil Narine","8","213","393","184.5","73.9"],
      ["Jake Fraser-McGurk","5","70","169","241.4","70.8"],
      ["Rajat Patidar","8","153","285","186.3","57.6"],
      ["Phil Salt","8","134","242","180.6","54.5"],
      ["Abhishek Sharma","10","128","228","178.1","52.0"],
      ["Heinrich Klaasen","10","190","337","177.4","48.2"],
    ],[3000,1080,1080,1080,1200,1260]),
    H2("Chase batting, by WPA (wins added)"),
    table(["Batter","Inns","Balls","Runs","SR","WPA"],[
      ["Shashank Singh","10","182","314","172.5","1.44"],
      ["Jos Buttler","7","195","308","157.9","1.38"],
      ["Shimron Hetmyer","6","50","81","162.0","0.93"],
      ["Marcus Stoinis","6","147","238","161.9","0.93"],
      ["Yashasvi Jaiswal","10","216","354","163.9","0.75"],
      ["Sanju Samson","10","208","333","160.1","0.73"],
    ],[3000,1080,1080,1080,1200,1260]),
    H2("Chase bowling, by WPA prevented"),
    table(["Bowler","Inns","Balls","WPA prevented"],[
      ["Sunil Narine","8","192","0.90"],
      ["Pat Cummins","10","222","0.78"],
      ["Mayank Yadav","3","54","0.64"],
      ["Kuldeep Yadav","6","139","0.57"],
      ["Jasprit Bumrah","5","122","0.37"],
    ],[3400,1400,1400,2500]),

    H1("6. Limits, left open"),
    bullet("All first-innings validity AUCs sit in a narrow 0.58 to 0.66 band. One batter only weakly predicts a team result, because bowling and the chase decide most of the game. None of these is a strong standalone predictor, only a relative ranking."),
    bullet("RAA here uses an over-by-over average. A lightly richer baseline (over plus wickets plus venue, as a smoothed lookup, not a boosted model) may add a little without the overfitting that sank the ML version. Untested."),
    bullet("Chase WPA rests on the DL power-law, which beat our ML classifier. If a future model beats the DL law out of sample, swap it in, not before."),
    bullet("The expected-runs model can close its small gap to the vendor by adding batting-lineup and bowling-quality features."),

    H1("7. Reproducibility"),
    P([T("Every result regenerates from the raw ball-by-ball files. Key scripts: "),T("validity_test.py",{font:"Consolas",bold:true}),T(" (the AUC verdict), "),T("improve_impact2.py",{font:"Consolas",bold:true}),T(" (reliability), "),T("raa_model.py",{font:"Consolas",bold:true}),T(" (the RAA baselines), "),T("combined_leaderboard.py",{font:"Consolas",bold:true}),T(" (the settled boards). Stack: pandas, numpy, scikit-learn, lightgbm. Open-source throughout.")]),
  ],
}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(require("path").join(__dirname,"..","outputs","Impact_Analytics_Findings.docx"),b);console.log("written",b.length);});
