"""Fork 2 (user: "let me select which of the models in our ensemble to watch, and show all back tests available" and "add an evaluation
metric - correlation or R2 per item"): patches eda/item_explorer_template.html.
  - model chips (system, rule agents, tree, NN, TabPFN conformal, TabPFN 32, R3) and origin chips (13 backtest origins, presets
    'back-to-back 4' / 'all 13' / 'none') above the history chart; both the history chart and the past-forecasts chart follow them;
  - a per-item scorecard by model over the item's scored origins: error / R3's, bias, within +-20%, correlation of log forecast with
    log actual, and R2 (1 - SS_res / SS_tot, units) of forecast against actual.
Idempotent (refuses if already applied). Rebuild with eda/212c_item_past.py."""
import re
p = "eda/item_explorer_template.html"; s = open(p, encoding="utf-8").read()
if 'id="pm"' in s: raise SystemExit("already patched")
lines = s.split("\n")
i0 = [k for k, l in enumerate(lines) if l.startswith("  // past forecasts: the four backtest origins")][0]
i1 = [k for k, l in enumerate(lines) if l.startswith('  leg($("#pj-leg")')][0]
NEW_PJ = r'''  // past forecasts: the selected backtest origins and models, each 4-month forecast drawn flat at its monthly average
  const PW = (x.past || []).filter((q) => PO.has(q.o)).map((q) => ({q, ix: [1,2,3,4].map((k) => MON.indexOf(addM(q.o, k))).filter((i) => i >= 0)}));
  const pastSeries = [], pastBands = [];
  PW.forEach((w) => { PMODELS.filter(([k]) => PM.has(k)).forEach(([k, , c]) => { if (w.q[k] == null) return; const v = Array(X.length).fill(null); w.ix.forEach((i) => v[i] = w.q[k] / 4); pastSeries.push({v, c: css(c), w: k === "F" ? 2.5 : 1.8, dash: k === "R3"}); });
    if (PM.has("F") && w.q.p10 != null) pastBands.push({idx: w.ix, lo: w.ix.map(() => w.q.p10 / 4), hi: w.ix.map(() => w.q.p90 / 4), fill: css("--band")}); });
  chart($("#pj"), {x: X, xl: MLAB, h: 300, label: "Monthly units, past forecasts and projection", shade: [n0, n0+3],
    band: pastBands.concat([{idx, lo: band(p.P10), hi: band(p.P90), fill: css("--band")}, {idx, lo: band(p.P25), hi: band(p.P75), fill: css("--band2")}]),
    series: [{v: hist, c: css("--ink2"), w: 2}].concat(pastSeries, [{v: ly, c: css("--ink2"), dash: true, w: 1.5}, {v: rulesM, c: css("--s2"), w: 2, dash: true}, {v: proj, c: css("--s1"), w: 2.5, dots: [n0, n0+1, n0+2, n0+3]}]),
    tip: (i) => { const t = []; if (hist[i] != null) t.push(["Actual", fmt(hist[i])]);
      PW.filter((w) => w.ix.includes(i)).forEach((w) => { const q = w.q; PMODELS.filter(([k]) => PM.has(k) && q[k] != null).forEach(([k, n]) => t.push([`${n}, made end of ${MLAB(q.o)}`, fmt(q[k] / 4) + "/mo · " + pcs(q.y ? q[k] / q.y - 1 : null)])); });
      if (i >= n0) { t.push(["Projection", fmt(p.v[i-n0])]); t.push(["P25–P75", fmt(p.v[i-n0]*sc(p.P25)) + "–" + fmt(p.v[i-n0]*sc(p.P75))]); if (rulesM[i] != null) t.push(["Rule agents", fmt(rulesM[i])]); if (ly[i] != null) t.push(["Same month 2025", fmt(ly[i])]); } return t; }});
  leg($("#pj-leg"), [{n:"Actual units", c: css("--ink2")}].concat(PW.length ? PMODELS.filter(([k]) => PM.has(k)).map(([k, n, c]) => ({n: "Past: " + n, c: css(c), cls: k === "R3" ? "dash" : ""})) : [], [{n:"Projection (submitted)", c: css("--s1")}, {n:"Rule agents, 2026", c: css("--s2"), cls:"dash"}, {n:"Same months last year", c: css("--ink2"), cls:"dash"}, {n:"P25–P75", c: css("--band2"), cls:"box"}, {n:"P10–P90", c: css("--band"), cls:"box"}]));'''
lines[i0:i1 + 1] = NEW_PJ.split("\n")
s = "\n".join(lines)

a = s.index("// ---------- past forecasts (backtest)"); b = s.index("function spark(v)")
NEW_PAST = r'''// ---------- past forecasts (backtest): pickers shared with the history chart
const PMODELS = [["F","System (submitted)","--s1"],["rules","Rule agents","--s2"],["qnx","Tree","--s3"],["nn","Neural net","--s6"],["pf2c","TabPFN conformal","--s4"],["pf2d32","TabPFN 32","--s5"],["R3","R3 (recent rate × 4)","--muted"]];
const ORIGS = ["2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12","2026-01","2026-02"];
const B2B = ["2025-02","2025-06","2025-10","2026-02"];
let PM = new Set(["F"]), PO = new Set(B2B);
try { const a = JSON.parse(localStorage.getItem("ie-pm") || "null"); if (Array.isArray(a)) PM = new Set(a); const b = JSON.parse(localStorage.getItem("ie-po") || "null"); if (Array.isArray(b)) PO = new Set(b); } catch(e) {}
const savePick = () => { try { localStorage.setItem("ie-pm", JSON.stringify([...PM])); localStorage.setItem("ie-po", JSON.stringify([...PO])); } catch(e) {} };
const addM = (o, k) => { const y = +o.slice(0,4), m = +o.slice(5) - 1 + k; return `${y + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, "0")}`; };
const pcs = (v) => v == null || !isFinite(v) ? "–" : (v >= 0 ? "+" : "") + (v * 100).toFixed(0) + "%";
function drawPickers(x) {
  const have = new Set((x.past || []).map((q) => q.o));
  $("#pm").innerHTML = PMODELS.map(([k, n, c]) => `<button class="chip" data-k="${k}" aria-pressed="${PM.has(k)}"><i class="sw" style="background:${css(c)}"></i>${esc(n)}</button>`).join("");
  $("#po").innerHTML = [["b2b","Back-to-back 4"],["all","All 13"],["none","None"]].map(([k, n]) => `<button class="chip preset" data-k="${k}">${n}</button>`).join("")
    + ORIGS.map((o) => `<button class="chip" data-o="${o}" aria-pressed="${PO.has(o)}" ${have.has(o) ? "" : 'disabled title="the item was not scored at this origin"'}>${MLAB(o)}</button>`).join("");
}
function onPick(e) { const b = e.target.closest(".chip"); if (!b || b.disabled) return;
  if (b.parentElement.id === "pm") { const k = b.dataset.k; PM.has(k) ? PM.delete(k) : PM.add(k); }
  else if (b.dataset.k === "b2b") PO = new Set(B2B); else if (b.dataset.k === "all") PO = new Set(ORIGS); else if (b.dataset.k === "none") PO = new Set();
  else if (b.dataset.o) { const o = b.dataset.o; PO.has(o) ? PO.delete(o) : PO.add(o); }
  savePick(); draw(); }
// per-item scorecard of one model over the item's scored origins
function itemScore(P, k) {
  const r = P.filter((q) => q[k] != null && q.y != null); if (!r.length) return null;
  const y = r.map((q) => q.y), f = r.map((q) => q[k]), n = r.length, sum = (a) => a.reduce((s, v) => s + v, 0);
  const ae = sum(f.map((v, i) => Math.abs(v - y[i]))), aeR3 = sum(r.map((q) => Math.abs(q.R3 - q.y))), ys = sum(y);
  const ly = y.map((v) => Math.log(v + 1)), lf = f.map((v) => Math.log(Math.max(v, 0) + 1)), my = sum(ly) / n, mf = sum(lf) / n;
  const cov = sum(ly.map((v, i) => (v - my) * (lf[i] - mf))), vy = sum(ly.map((v) => (v - my) ** 2)), vf = sum(lf.map((v) => (v - mf) ** 2));
  const ym = ys / n, sst = sum(y.map((v) => (v - ym) ** 2)), ssr = sum(f.map((v, i) => (v - y[i]) ** 2));
  return {n, wr3: aeR3 > 0 ? ae / aeR3 : null, bias: ys > 0 ? sum(f) / ys - 1 : null, w20: f.filter((v, i) => Math.abs(v - y[i]) <= 0.2 * y[i]).length / n,
          corr: n >= 3 && vy > 0 && vf > 0 ? cov / Math.sqrt(vy * vf) : null, r2: n >= 3 && sst > 0 ? 1 - ssr / sst : null};
}
function past(x) {
  const P = x.past || [], S = x.past_stats, el = $("#past");
  if (!P.length) { $("#past-kpi").innerHTML = ""; $("#past-leg").innerHTML = ""; $("#past-sc").innerHTML = ""; el.innerHTML = '<p class="muted">No backtest forecasts for this item: it launched too recently or sold too little at the 2025–26 origins to be scored.</p>'; $("#past-tb").hidden = true; return; }
  $("#past-tb").hidden = false;
  const win = (o) => MLAB(addM(o, 1)).split(" ")[0] + "–" + MLAB(addM(o, 4));
  const sysS = itemScore(P, "F");
  $("#past-kpi").innerHTML = [["Origins scored", `${S.n} of 13`], ["System error ÷ R3 (R3 = 1)", S.wape_r3 != null ? S.wape_r3.toFixed(2) : "–"], ["Bias", pcs(S.bias)], ["Within ±20%", (S.w20 * 100).toFixed(0) + "%"],
    ["Correlation (logs)", sysS && sysS.corr != null ? sysS.corr.toFixed(2) : "–"], ["R²", sysS && sysS.r2 != null ? sysS.r2.toFixed(2) : "–"], ["Actual inside P10–P90", (S.inside * 100).toFixed(0) + "%"]]
    .map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
  const sc = PMODELS.map(([k, n, c]) => [k, n, c, itemScore(P, k)]).filter((r) => r[3]);
  const best = (f, lowGood) => { const v = sc.map((r) => r[3][f]).filter((x) => x != null && isFinite(x)); return v.length ? (lowGood ? Math.min(...v) : Math.max(...v)) : null; };
  const bw = best("wr3", true), bc = best("corr", false), br = best("r2", false), bb = (() => { const v = sc.map((r) => r[3].bias).filter((x) => x != null); return v.length ? Math.min(...v.map(Math.abs)) : null; })();
  const cell = (v, d, isBest) => `<td class="x">${v == null || !isFinite(v) ? "–" : (isBest ? "<b>" : "") + v.toFixed(d) + (isBest ? "</b>" : "")}</td>`;
  $("#past-sc").innerHTML = `<thead><tr><th>Model, over this item's ${S.n} origins</th><th style="text-align:right">Error ÷ R3's</th><th style="text-align:right">Bias</th><th style="text-align:right">Within ±20%</th><th style="text-align:right">Correlation (logs)</th><th style="text-align:right">R²</th></tr></thead><tbody>`
    + sc.map(([k, n, c, m]) => `<tr${PM.has(k) ? "" : ' class="dim"'}><td><i class="sw" style="background:${css(c)}"></i>${esc(n)}</td>${k === "R3" ? '<td class="x">1.00</td>' : cell(m.wr3, 2, m.wr3 === bw)}<td class="x">${m.bias != null && Math.abs(m.bias) === bb ? "<b>" + pcs(m.bias) + "</b>" : pcs(m.bias)}</td><td class="x">${(m.w20 * 100).toFixed(0)}%</td>${cell(m.corr, 2, m.corr === bc)}${cell(m.r2, 2, m.r2 === br)}</tr>`).join("") + "</tbody>";
  const M = PMODELS.filter(([k]) => PM.has(k));
  const X = P.map((q) => q.o), idx = P.map((_, i) => i), rec = X.indexOf("2025-06");
  chart(el, {x: X, xl: (o) => MLAB(o), h: 260, label: "Past four-month forecasts against actual units",
    band: PM.has("F") ? [{idx, lo: P.map((q) => q.p10), hi: P.map((q) => q.p90), fill: css("--band")}] : [],
    marks: rec > 0 ? [{i: rec, t: "recent 9"}] : [],
    series: [{v: P.map((q) => q.y), c: css("--ink2"), w: 2, dots: idx}].concat(M.map(([k, , c]) => ({v: P.map((q) => q[k]), c: css(c), w: k === "F" ? 2.5 : 1.8, dash: k === "R3", dots: k === "F" ? idx : null}))),
    tip: (i) => { const q = P[i]; return [["Window", win(q.o)], ["Actual", fmt(q.y)]].concat(M.map(([k, n]) => [n, fmt(q[k]) + " (" + pcs(q.y ? q[k] / q.y - 1 : null) + ")"]), PM.has("F") ? [["System P10–P90", fmt(q.p10) + "–" + fmt(q.p90)]] : []); }});
  leg($("#past-leg"), [{n: "Actual units", c: css("--ink2")}].concat(M.map(([k, n, c]) => ({n, c: css(c), cls: k === "R3" ? "dash" : ""})), PM.has("F") ? [{n: "System P10–P90", c: css("--band"), cls: "box"}] : []));
  $("#past-tb thead").innerHTML = `<tr><th>Data through</th><th>Forecast window</th><th style="text-align:right">Actual</th>${M.map(([, n]) => `<th style="text-align:right">${esc(n)}</th>`).join("")}${PM.has("F") ? '<th style="text-align:right">System P10–P90</th>' : ""}</tr>`;
  $("#past-tb tbody").innerHTML = P.slice().reverse().map((q) => { const ins = q.y >= q.p10 && q.y <= q.p90;
    return `<tr><td>${MLAB(q.o)}</td><td>${win(q.o)}</td><td class="x">${fmt(q.y)}</td>${M.map(([k]) => `<td class="x">${fmt(q[k])} <span class="muted small">${pcs(q.y ? q[k] / q.y - 1 : null)}</span></td>`).join("")}${PM.has("F") ? `<td class="x">${fmt(q.p10)}–${fmt(q.p90)}${ins ? "" : ' <span class="muted" title="actual outside the range">·out</span>'}</td>` : ""}</tr>`; }).join("");
}
$("#pm").addEventListener("click", onPick); $("#po").addEventListener("click", onPick);
'''
s = s[:a] + NEW_PAST + s[b:]

a = '      <div class="legend" id="pj-leg"></div>'; assert s.count(a) == 1
s = s.replace(a, '''      <div class="pick"><span class="pk">Models to watch</span><div class="chips" id="pm" role="group" aria-label="Models to show"></div></div>
      <div class="pick"><span class="pk">Past forecasts made at the end of</span><div class="chips" id="po" role="group" aria-label="Backtest origins to show"></div></div>
''' + a, 1)
a = "  past(x);\n"; assert s.count(a) == 1; s = s.replace(a, "  drawPickers(x); past(x);\n", 1)
a = '      <div class="legend" id="past-leg"></div>'; assert s.count(a) == 1
s = s.replace(a, '      <div class="tscroll"><table class="nb scard" id="past-sc"></table></div>\n      <p class="note">Bold = best of the models for this item. Correlation is between log forecast and log actual across the item\'s origins (does the forecast move with the item?); R² = 1 − squared error ÷ variance of the actuals (can be negative: worse than always forecasting the item\'s average). Both need at least three origins and say little below about six.</p>\n' + a, 1)
a = '.note{font-size:12px;color:var(--muted);max-width:90ch}\n'; assert s.count(a) == 1
s = s.replace(a, a + '.pick{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:4px 0 8px}.pick .pk{font-size:12px;color:var(--muted);min-width:150px}.pick .chips{display:flex;flex-wrap:wrap;gap:6px}\n'
              '.sw{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}.chip.preset{border-style:dashed}.chip[disabled]{opacity:.35;cursor:default}\n'
              'table.scard tr.dim td{color:var(--muted)} table.scard{margin-bottom:6px}\n', 1)
old = re.search(r"In purple, what the same system forecast in the past:.*?Every origin is listed in the next section\.", s).group(0)
s = s.replace(old, "Past forecasts are drawn flat at their monthly average (only four-month totals are stored), with the system's P10–P90. Choose the models and which of the 13 backtest origins (Feb 2025–Feb 2026; the scoreboard's 'recent' figure uses the 9 from Jun 2025) to show; the default is the four whose windows follow each other. Neighbouring origins' windows overlap by three months.")
open(p, "w", encoding="utf-8").write(s)
print("patched")
