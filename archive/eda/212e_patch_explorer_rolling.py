"""Fork 2 (user: "show it as a time series of rolling average of 4 months, both true and prediction"): the history chart of the explorer
shows the actual units as a trailing 4-month average (month m = mean of m-3 .. m) and each model's past forecasts as a line through the
window ends: the forecast made with data through month o covers o+1 .. o+4 and is plotted at o+4 as total / 4, the same quantity as the
actual's rolling average at that month. All 13 origins form one line per model, so the origin picker is removed. The 2026 projection
is added at Oct 2026 the same way. Raw monthly units stay as a light line. Idempotent."""
p = "eda/item_explorer_template.html"; s = open(p, encoding="utf-8").read()
if "roll4" in s: raise SystemExit("already patched")
lines = s.split("\n")
i0 = [k for k, l in enumerate(lines) if l.startswith("  // past forecasts: the selected backtest origins and models")][0]
i1 = [k for k, l in enumerate(lines) if l.startswith('  leg($("#pj-leg")')][0]
NEW = r'''  // rolling view: actual as a trailing 4-month average; each past forecast (data through o, months o+1..o+4) plotted at o+4 as total / 4
  const roll4 = X.map((_, i) => i >= 3 && i < n0 ? (x.monthly[i] + x.monthly[i-1] + x.monthly[i-2] + x.monthly[i-3]) / 4 : null);
  const mm = x.monthly.concat(p.v), rollP = X.map((_, i) => i >= n0 - 1 ? (mm[i] + mm[i-1] + mm[i-2] + mm[i-3]) / 4 : null);
  const P4 = x.past || [], endIx = (o) => X.indexOf(addM(o, 4)), Msel = PMODELS.filter(([k]) => PM.has(k));
  const fser = Msel.map(([k, n, c]) => { const v = Array(X.length).fill(null); P4.forEach((q) => { const i = endIx(q.o); if (i >= 0 && q[k] != null) v[i] = q[k] / 4; });
    if (k === "F") v[n0 + 3] = p.total / 4;
    if (k === "rules" && p.rules_m) v[n0 + 3] = p.rules_m.reduce((a, b) => a + b, 0) / 4;
    const d = v.map((u, i) => u != null ? i : -1).filter((i) => i >= 0); return {v, c: css(c), w: k === "F" ? 2.5 : 1.8, dash: k === "R3", dots: k === "F" ? d : null, k, n}; });
  const fb = PM.has("F") ? (() => { const r = P4.filter((q) => q.p10 != null && endIx(q.o) >= 0); return r.length ? [{idx: r.map((q) => endIx(q.o)), lo: r.map((q) => q.p10 / 4), hi: r.map((q) => q.p90 / 4), fill: css("--band")}] : []; })() : [];
  chart($("#pj"), {x: X, xl: MLAB, h: 320, label: "Four-month rolling average: actual units and past forecasts", shade: [n0, n0+3],
    band: fb.concat(PM.has("F") ? [{idx: [n0+3], lo: [p.P10 / 4], hi: [p.P90 / 4], fill: css("--band")}] : []),
    marks: [{i: n0 - 1, t: "data end"}],
    series: [{v: hist, c: css("--axis"), w: 1.2}, {v: roll4, c: css("--ink2"), w: 2.5}, {v: rollP, c: css("--s1"), w: 2, dash: true}].concat(fser),
    tip: (i) => { const t = []; if (hist[i] != null) t.push(["Units this month", fmt(hist[i])]); if (roll4[i] != null) t.push(["Actual, last 4 months ÷ 4", fmt(roll4[i])]); if (i >= n0) t.push(["Rolling, with projected months", fmt(rollP[i])]);
      const q = P4.find((r) => endIx(r.o) === i);
      if (q) { t.push(["Forecast made end of", MLAB(q.o) + " (for " + MLAB(addM(q.o, 1)).split(" ")[0] + "–" + MLAB(addM(q.o, 4)) + ")"]); fser.forEach((s) => { if (s.v[i] != null) t.push([s.n, fmt(s.v[i]) + " · " + pcs(roll4[i] ? s.v[i] / roll4[i] - 1 : null)]); }); if (PM.has("F") && q.p10 != null) t.push(["System P10–P90", fmt(q.p10 / 4) + "–" + fmt(q.p90 / 4)]); }
      if (i === n0 + 3) { t.push(["2026 projection, Jul–Oct ÷ 4", fmt(p.total / 4)]); t.push(["P10–P90", fmt(p.P10 / 4) + "–" + fmt(p.P90 / 4)]); }
      return t; }});
  leg($("#pj-leg"), [{n: "Actual, 4-month rolling average", c: css("--ink2")}, {n: "Actual, single month", c: css("--axis")}, {n: "Rolling average continued with the 2026 projection", c: css("--s1"), cls: "dash"}].concat(fser.map((s) => ({n: s.n + " (forecast ÷ 4, at window end)", c: s.c, cls: s.k === "R3" ? "dash" : ""})), PM.has("F") ? [{n: "System P10–P90", c: css("--band"), cls: "box"}] : []));'''
lines[i0:i1 + 1] = NEW.split("\n")
s = "\n".join(lines)
# drop the origin picker (all 13 origins form one line now)
a = '      <div class="pick"><span class="pk">Past forecasts made at the end of</span><div class="chips" id="po" role="group" aria-label="Backtest origins to show"></div></div>\n'
assert s.count(a) == 1; s = s.replace(a, "")
a = '  $("#po").innerHTML ='; j = s.index(a); k = s.index("\n}", j); s = s[:j] + "  // origin picker removed (rolling view shows all 13 origins)" + s[k:]
a = '$("#pm").addEventListener("click", onPick); $("#po").addEventListener("click", onPick);'; assert s.count(a) == 1
s = s.replace(a, '$("#pm").addEventListener("click", onPick);')
import re
old = re.search(r'<h2>History and projection</h2><p class="sub small">.*?</p>', s).group(0)
s = s.replace(old, '<h2>History, past forecasts and projection</h2><p class="sub small">Four-month rolling averages, so forecasts and actuals compare like for like. The dark line is the actual: at each month, the average of that month and the three before. Each model\'s line joins its past forecasts: a forecast made with data through one month covers the next four and is plotted at the last of them as its total ÷ 4, so where it meets the dark line it was right. 13 backtest origins, data through Feb 2025 to Feb 2026; the scoreboard\'s "recent" score uses the 9 from Jun 2025. The point at Oct 2026 is the submitted Jul–Oct projection ÷ 4. The dashed blue line continues the rolling average through Jul–Oct 2026 with the projected months. The light line is single months.</p>')
open(p, "w", encoding="utf-8").write(s)
print("patched")
