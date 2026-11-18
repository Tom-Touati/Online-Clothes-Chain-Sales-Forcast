"""Atlas template patch (D132): adds section 11 'Evaluation & error analysis' (data: J["eval"] from eda/124), renumbers the later
sections, adds the nav link, and fixes the decision-log parser in eda/19 (ids like D125b, and the indented sub-bullets of an entry are
kept as its body). Idempotent: does nothing if the section is already there."""
import re
p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
if 'id="evaluation"' not in s:
    s = s.replace('<a href="#forecast">Forecast</a>', '<a href="#evaluation">Evaluation</a><a href="#forecast">Forecast</a>', 1)
    s = s.replace('<div class="eyebrow">13 · Files &amp; how to run</div>', '<div class="eyebrow">14 · Files &amp; how to run</div>')
    s = s.replace('<div class="eyebrow">12 · Decision log</div>', '<div class="eyebrow">13 · Decision log</div>')
    s = s.replace('<div class="eyebrow">11 · The forecast</div>', '<div class="eyebrow">12 · The forecast</div>')
    SEC = '''<section class="block" id="evaluation"><div class="wrap">
  <div class="eyebrow">11 · Evaluation &amp; error analysis</div>
  <h2>Judged against recency, then taken apart error by error</h2>
  <p class="muted">Every system is scored as a ratio to R3 (each item's last 3 months × 4): below 1 beats recency. R3 + rules is the rule-agent system (recent rate × explicit rules, rewritten monthly by isolated agents). Same 13 origins and items throughout; "recent" = the nine origins June 2025 – February 2026.</p>
  <div class="panel">
    <h3>Scoring table, relative to R3</h3>
    <div class="controls"><div class="seg" id="ev-scope" role="group" aria-label="Scope"><button aria-pressed="true" data-v="table_recent">Nine recent origins</button><button aria-pressed="false" data-v="table_all">All 13 origins</button><button aria-pressed="false" data-v="table_targets">Recent, 100 targets</button></div></div>
    <div class="tscroll"><table id="ev-table"><thead><tr><th>System</th><th class="r">WAPE / R3</th><th class="r">Typical item / R3</th><th class="r">Wins vs R3</th><th class="r">Wins vs R3 + rules</th><th class="r">(MAE + |bias|) / R3</th><th class="r">Pinball / R3</th><th class="r">Buying loss / R3</th><th class="r">Bias</th></tr></thead><tbody></tbody></table></div>
    <p class="small muted">Typical item = geometric mean of the item error ratio (Davydenko–Fildes). Pinball over P10–P90. Buying loss = profit lost against perfect foresight in the daily buying simulation (11 windows, production buying policy), recent scope only.</p>
  </div>
  <div class="grid2">
    <div class="panel"><h3>What each step adds, from R3</h3><div class="tscroll"><table id="ev-fva"><thead><tr><th>Stage</th><th class="r">WAPE / R3</th><th class="r">Typical item</th><th class="r">Pinball / R3</th></tr></thead><tbody></tbody></table></div>
      <p class="small muted">Seasonality alone removes about 38% of recency's error; the quantile model's other learning another 15%; the NN and the rules about 1.5 points each.</p></div>
    <div class="panel"><h3>The 2026 forecast, step by step (100 targets)</h3><div id="ev-wf"></div>
      <p class="small muted">Rules + quantile + NN candidate. Model steps are TreeSHAP groups of the June 2026 quantile model, in log ratio to the recent-rate anchor.</p></div>
  </div>
  <div class="panel"><h3>Where the error lives</h3>
    <div class="grid2"><div class="tscroll"><table id="ev-arch"><thead><tr><th>Archetype (recent)</th><th class="r">Item-origins</th><th class="r">Units</th><th class="r">Abs. error</th><th class="r">Bias</th><th class="r">WAPE / R3</th></tr></thead><tbody></tbody></table></div>
    <ul class="takeaways small" id="ev-notes"></ul></div>
  </div>
  <div class="panel"><h3>Bias by target month</h3><div class="legend" id="ev-mb-leg"></div><div id="ev-mb"></div>
    <p class="small muted">Members' monthly forecasts, all origins and horizons pooled. November 2025 (Black Friday) is under-forecast by every member.</p></div>
  <div class="panel"><h3>Bias by origin and product group (rules + quantile + NN)</h3><div class="tscroll"><table id="ev-om"></table></div></div>
  <div class="panel"><h3>Probes and fixes tried</h3><div class="tscroll"><table id="ev-probes"><thead><tr><th>Probe</th><th>Decision</th><th>Verdict</th><th>Why</th></tr></thead><tbody></tbody></table></div></div>
</div></section>
'''
    s = s.replace('<section class="block" id="forecast">', SEC + '<section class="block" id="forecast">', 1)
    JS = '''
/* ---------- evaluation & error analysis ---------- */
(function evaluation() {
  const EV = D.eval; if (!EV) return;
  const f3 = x => x == null ? "–" : x.toFixed(3), sg = x => x == null ? "–" : (x > 0 ? "+" : "") + (x * 100).toFixed(1) + "%";
  const best = (rows, k) => Math.min(...rows.filter(r => r.name !== "R3 + rules (rule agents)" && r[k] != null).map(r => r[k]));
  function renderEv(scope) {
    const rows = EV[scope]; const b = best(rows, "WAPE / R3");
    $("#ev-table tbody").innerHTML = rows.map(r => `<tr${r["WAPE / R3"] === b ? ' style="background:var(--accent-soft)"' : ""}><td>${r.name === "R3 + rules (rule agents)" ? "<b>R3 + rules (rule agents)</b>" : esc(r.name)}</td><td class="r">${f3(r["WAPE / R3"])}</td><td class="r">${f3(r["relMAE vs R3"])}</td><td class="r">${pct(r["win vs R3"], 1)}</td><td class="r">${pct(r["win vs R3+rules"], 1)}</td><td class="r">${f3(r["(MAE+|bias|) / R3"])}</td><td class="r">${f3(r["pinball / R3"])}</td><td class="r">${f3(r["sim loss / R3"])}</td><td class="r">${sg(r.bias)}</td></tr>`).join("");
  }
  segCtl($("#ev-scope"), renderEv); renderEv("table_recent");
  $("#ev-fva tbody").innerHTML = EV.fva.filter(r => r.name !== "R3 + rules (rule agents)").map(r => `<tr><td>${esc(r.name)}</td><td class="r">${f3(r["WAPE / R3"])}</td><td class="r">${f3(r["relMAE vs R3"])}</td><td class="r">${f3(r["pinball / R3"])}</td></tr>`).join("");
  const wf = EV.waterfall; const mx = Math.max(...wf.map(w => w.total));
  bars($("#ev-wf"), wf.map((w, i) => ({ k: w.step.replace("model: ", "· "), v: w.total, c: i === 0 || i === wf.length - 1 ? "var(--s1)" : "var(--s2)",
    tip: `<b>${esc(w.step)}</b><br>${fmt(w.total)} units${i ? ` (${w.total >= wf[i - 1].total ? "+" : ""}${fmt(w.total - wf[i - 1].total)}, ×${(w.total / wf[i - 1].total).toFixed(3)})` : ""}` })), { max: mx });
  $("#ev-arch tbody").innerHTML = EV.archetypes.map(a => `<tr><td>${esc(a.name)}</td><td class="r">${pct(a.items)}</td><td class="r">${pct(a.units)}</td><td class="r">${pct(a.error)}</td><td class="r">${sg(a.bias)}</td><td class="r">${f3(a.rel)}</td></tr>`).join("");
  $("#ev-notes").innerHTML = [`<b>Concentration:</b> ${pct(EV.concentration.half)} of item-origins carry half of all absolute error, ${pct(EV.concentration.eighty)} carry 80%.`,
    "<b>Black Friday:</b> the average item lift is stable year to year (median 5.7×, ratio to last year 1.00) but an item's own lift barely repeats (r = 0.18).",
    "<b>Seasonal turns:</b> warm items were under-forecast as winter arrived and over-forecast as it ended; swimwear's own last-year pattern called its turns where the models did not.",
    "<b>Disagreement:</b> where the members disagree most, the error is twice as large; the actual lands near the middle, so equal weights are right on average.",
    "<b>Fine granularity:</b> forecasting per store / channel and summing was worse — TikTok and new dropship partners ramped, store assortments turned over, and those errors do not cancel."].map(x => `<li>${x}</li>`).join("");
  const mb = EV.month_bias, cols = { lgb: "var(--s3)", nn: "var(--s4)", rules: "var(--s2)", geo: "var(--s1)" }, labs = { lgb: "LightGBM", nn: "NN", rules: "Rules", geo: "Three-way shape" };
  legend($("#ev-mb-leg"), Object.keys(cols).map(k => [cols[k], labs[k]]));
  lineChart($("#ev-mb"), { labels: mb.months, series: Object.keys(cols).map(k => ({ values: mb[k], color: cols[k], w: k === "geo" ? 2.6 : 1.4, dots: k === "geo" })), yMin: -0.4, yMax: 0.55, ref: 0,
    yFmt: v => (v * 100).toFixed(0) + "%", label: "Bias by target month", tip: i => `<b>${mb.months[i]}</b><br>` + Object.keys(cols).map(k => `${labs[k]}: ${sg(mb[k][i])}`).join("<br>") });
  const om = EV.origin_macro; const cell = v => { if (v == null) return "<td>–</td>"; const a = Math.min(1, Math.abs(v) / 0.5); const c = v < 0 ? `rgba(80,120,220,${a * 0.55})` : `rgba(220,90,70,${a * 0.55})`; return `<td class="r" style="background:${c}">${sg(v)}</td>`; };
  $("#ev-om").innerHTML = `<thead><tr><th>Origin</th>${om.groups.map(g => `<th class="r">${esc(g)}</th>`).join("")}</tr></thead><tbody>` + om.origins.map((o, i) => `<tr><td>${o}</td>${om.bias[i].map(cell).join("")}</tr>`).join("") + "</tbody>";
  $("#ev-probes tbody").innerHTML = EV.probes.map(p => `<tr><td>${esc(p[0])}</td><td><button class="chip" type="button" onclick="openLog('${p[1].split(/[-–]/)[0]}')">${esc(p[1])}</button></td><td>${esc(p[2])}</td><td class="small">${esc(p[3])}</td></tr>`).join("");
})();
'''
    k = s.rfind("})();\n</script>")
    assert k > 0
    s = s[:k] + JS + s[k:]
    open(p, "w", encoding="utf-8").write(s); print("template patched")
q = "eda/19_project_map_data.py"; t = open(q, encoding="utf-8").read()
old = '''for line in log.splitlines():
    if line.startswith("## Turn"): turn = line[3:].strip()
    mm = re.match(r"- \\*\\*(D\\d{3}) (.+?)\\*\\* (.*)", line)
    if mm:
        body = mm.group(3); sm = re.search(r"Status: ([^.]+)\\.?\\s*$", body)
        J["log"].append(dict(id=mm.group(1), title=mm.group(2).rstrip("."), body=re.sub(r"\\s*Status: [^.]+\\.?\\s*$", "", body), status=sm.group(1) if sm else "", turn=turn))'''
new = '''entries = []
for line in log.splitlines():
    if line.startswith("## Turn"): turn = line[3:].strip(); continue
    mm = re.match(r"- \\*\\*(D\\d{3}[a-z]?) (.+?)\\*\\*\\s*(.*)", line)
    if mm: entries.append([mm.group(1), mm.group(2), mm.group(3), turn]); continue
    if entries and line.startswith("  ") and line.strip(): entries[-1][2] += " " + re.sub(r"^\\s*-\\s*", "", line.strip())   # sub-bullets belong to the entry
for eid, title, body, tn in entries:
    body = body.strip(); sm = re.search(r"Status: ([^.]+)\\.?\\s*$", body)
    J["log"].append(dict(id=eid, title=title.rstrip("."), body=re.sub(r"\\s*Status: [^.]+\\.?\\s*$", "", body), status=sm.group(1) if sm else "", turn=tn))'''
if old in t:
    t = t.replace(old, new); open(q, "w", encoding="utf-8").write(t); print("log parser fixed")
else:
    print("log parser: pattern not found (already fixed?)", "entries = []" in t)
