"""Fork 2 (user: "add future projection of each model separately"): in the explorer's rolling chart every selected model gets its own
Jul-Oct 2026 projection, drawn dashed in its colour as the continuation of the 4-month rolling average (actual months + that model's
projected months). Monthly split: rule agents from their own months; R3 flat at R3 / 4; the tree, NN and both TabPFN members carry only a
total, so they use the submitted projection's monthly shape scaled to their total. The submitted projection is always shown. Idempotent."""
p = "eda/item_explorer_template.html"; s = open(p, encoding="utf-8").read()
if "fproj" in s: raise SystemExit("already patched")
for a in ['    if (k === "F") v[n0 + 3] = p.total / 4;\n', '    if (k === "rules" && p.rules_m) v[n0 + 3] = p.rules_m.reduce((a, b) => a + b, 0) / 4;\n']:
    assert s.count(a) == 1; s = s.replace(a, "")
a = '  const fb = PM.has("F") ?'; assert s.count(a) == 1
s = s.replace(a, r'''  // each selected model's own 2026 projection as the rolling average continued (actual months + its projected months)
  const projM = (k) => k === "rules" ? p.rules_m : k === "R3" ? [1,2,3,4].map(() => p.R3 / 4) : (p.members && p.members[k] != null && p.total ? p.v.map((m) => m * p.members[k] / p.total) : null);
  const projTot = (k) => k === "rules" ? (p.rules_m || []).reduce((a, b) => a + b, 0) : k === "R3" ? p.R3 : (p.members || {})[k];
  const fproj = Msel.filter(([k]) => k !== "F").map(([k, n, c]) => { const pm = projM(k); if (!pm) return null; const m2 = x.monthly.concat(pm);
    const v = X.map((_, i) => i >= n0 - 1 ? (m2[i] + m2[i-1] + m2[i-2] + m2[i-3]) / 4 : null); return {v, c: css(c), w: 1.8, dash: true, dots: [n0 + 3], k, n, tot: projTot(k)}; }).filter(Boolean);
''' + a, 1)
a = '{v: rollP, c: css("--s1"), w: 2, dash: true}].concat(fser),'; assert s.count(a) == 1
s = s.replace(a, '{v: rollP, c: css("--s1"), w: 2.5, dash: true, dots: [n0 + 3]}].concat(fser, fproj),')
a = '      if (i === n0 + 3) { t.push(["2026 projection, Jul–Oct ÷ 4", fmt(p.total / 4)]); t.push(["P10–P90", fmt(p.P10 / 4) + "–" + fmt(p.P90 / 4)]); }\n'; assert s.count(a) == 1
s = s.replace(a, '''      if (i >= n0) fproj.forEach((f) => t.push([f.n + ", rolling", fmt(f.v[i])]));
      if (i === n0 + 3) { t.push(["Submitted, Jul–Oct total", fmt(p.total) + " (P10–P90 " + fmt(p.P10) + "–" + fmt(p.P90) + ")"]); fproj.forEach((f) => t.push([f.n + ", Jul–Oct total", fmt(f.tot) + " · " + pcs(p.total ? f.tot / p.total - 1 : null) + " vs submitted"])); }
''')
a = '{n: "Rolling average continued with the 2026 projection", c: css("--s1"), cls: "dash"}]'; assert s.count(a) == 1
s = s.replace(a, '{n: "Submitted 2026 projection (rolling)", c: css("--s1"), cls: "dash"}].concat(fproj.map((f) => ({n: f.n + ", 2026 (rolling)", c: f.c, cls: "dash"})))')
a = "The dashed blue line continues the rolling average through Jul–Oct 2026 with the projected months."; assert s.count(a) == 1
s = s.replace(a, "Dashed lines continue the rolling average through Jul–Oct 2026 with each selected model's own projected months (blue = submitted). The rule agents have their own months and R3 is flat; the tree, NN and TabPFN members store only a Jul–Oct total, so their months follow the submitted shape.")
open(p, "w", encoding="utf-8").write(s)
print("patched")
