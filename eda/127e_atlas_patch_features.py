"""D343 (fork 2; user: "update the atlas for final inspection. put in all the features used of course"): a 'Features' section in the atlas
listing every input of every model of the candidate and submitted systems (data/feature_inventory.json from eda/206), with a model selector,
a search box and per-block counts; eda/127_build_atlas.py adds the inventory to the page data. Idempotent."""
p = "eda/127_build_atlas.py"; s = open(p, encoding="utf-8").read()
if "feature_inventory" not in s:
    a = 'J = json.load(open("data/project_map.json", encoding="utf-8"))'; assert a in s
    s = s.replace(a, a + '\nJ["features"] = json.load(open("data/feature_inventory.json", encoding="utf-8"))   # D343: every model input (eda/206)'); open(p, "w", encoding="utf-8").write(s)
p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
if 'id="features"' in s: raise SystemExit("template already patched")
a = '<a href="#decisions">Decision log</a>'; assert s.count(a) == 1; s = s.replace(a, '<a href="#features">Features</a>' + a)
SEC = '''<section class="block" id="features"><div class="wrap">
  <div class="eyebrow">Features</div>
  <h2>Every input of every model</h2>
  <p class="muted">Taken from the code that builds each model (<code>eda/206_feature_inventory.py</code>), not retyped. Every input is known on the forecast date. Pick a model to see its inputs; the search box filters by name, block or description.</p>
  <div class="controls"><select id="feat-model" aria-label="Model"></select><input id="feat-q" type="search" placeholder="Search inputs" aria-label="Search inputs"><span class="small muted" id="feat-n"></span></div>
  <div class="grid2">
    <div class="panel"><h3 id="feat-title"></h3><p class="small muted" id="feat-src"></p><div class="bars" id="feat-blocks"></div></div>
    <div class="tscroll" style="max-height:520px;overflow:auto"><table id="feat-tab"><thead><tr><th>Input</th><th>Block</th><th>What it is</th></tr></thead><tbody></tbody></table></div>
  </div>
  <p class="small muted" id="feat-note"></p>
</div></section>

'''
a = '<section class="block" id="decisions">'; assert s.count(a) == 1; s = s.replace(a, SEC + a)
JS = r'''
// ---- features (D343)
(function () {
  const FI = D.features; if (!FI) return; const sel = document.getElementById("feat-model"), q = document.getElementById("feat-q"), tb = document.querySelector("#feat-tab tbody");
  FI.models.forEach((m, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${m.name} (${m.n})`; sel.appendChild(o); });
  document.getElementById("feat-note").textContent = FI.note;
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function draw() {
    const m = FI.models[+sel.value], k = q.value.trim().toLowerCase(), rows = m.features.filter((f) => !k || (f.name + " " + f.block + " " + f.desc).toLowerCase().includes(k));
    document.getElementById("feat-title").textContent = m.name; document.getElementById("feat-src").textContent = `${m.role} · ${m.source}`;
    document.getElementById("feat-n").textContent = `${rows.length} of ${m.n} inputs`;
    tb.innerHTML = rows.map((f) => `<tr><td><code>${esc(f.name)}</code></td><td class="small">${esc(f.block)}</td><td class="small">${esc(f.desc)}</td></tr>`).join("");
    const cnt = {}; m.features.forEach((f) => { cnt[f.block] = (cnt[f.block] || 0) + 1; }); const mx = Math.max(...Object.values(cnt));
    document.getElementById("feat-blocks").innerHTML = Object.entries(cnt).sort((a, b) => b[1] - a[1]).map(([b, n]) => `<div class="bar"><span class="k" title="${esc(b)}">${esc(b)}</span><div class="track"><div class="fill" style="width:${(n / mx) * 100}%"></div></div><span class="v">${n}</span></div>`).join("");
  }
  sel.addEventListener("change", draw); q.addEventListener("input", draw); sel.value = 1; draw();
})();
'''
a = "</script>\n</body>"
if a not in s: a = "</script>"; idx = s.rindex(a)
else: idx = s.rindex(a)
s = s[:idx] + JS + s[idx:]
open(p, "w", encoding="utf-8").write(s); print("patched")
