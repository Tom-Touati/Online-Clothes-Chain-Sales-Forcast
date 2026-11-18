"""Fork: one-off patch after the user's request in the session ("add an indication of the item volume"): every table that lists
single items (worst misses, the 2026 watch list, the flagged 2026 targets) gets a 'Units a month' column with a small bar.
Volume = R3 / 4, the average of the last three months at that origin. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if "const volCell" in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80]); s = s.replace(a, b, 1)


rep("const mon = o => [", "const volCell = (v, mx) => h('td', {}, h('div', { style: 'display:flex;gap:8px;align-items:center;min-width:120px' }, h('div', { class: 'pair', style: 'flex:1' }, h('div', { class: 'e', style: 'width:' + Math.max(3, Math.sqrt(v / mx) * 100) + '%;height:9px' })), h('span', { text: int(v) })));\nconst mon = o => [")
rep('<table id="t-worst"><thead><tr><th>Item</th><th>Origin</th>', '<table id="t-worst"><thead><tr><th>Item</th><th>Units a month at the origin</th><th>Origin</th>')
rep('<table id="t-watch"><thead><tr><th>Item</th><th class="r">Forecast</th>', '<table id="t-watch"><thead><tr><th>Item</th><th>Units a month now</th><th class="r">Forecast</th>')
rep('<table id="t-flag26"><thead><tr><th>2026 target</th><th>Type</th>', '<table id="t-flag26"><thead><tr><th>2026 target</th><th>Type</th><th>Units a month now</th>')
rep("{ const tb = $('#t-worst tbody'); J.worst.forEach(w => tb.append(h('tr', {}, h('td', { html: `${w.name}<br><span class=\"small\">${w.item}</span>` }), h('td', { text: mon(w.origin) })",
    "{ const tb = $('#t-worst tbody'), VW = Math.max(...J.worst.map(w => w.vol)); J.worst.forEach(w => tb.append(h('tr', {}, h('td', { html: `${w.name}<br><span class=\"small\">${w.item}</span>` }), volCell(w.vol, VW), h('td', { text: mon(w.origin) })")
rep("const tb = $('#t-watch tbody'); J.watch.forEach(w => tb.append(h('tr', {}, h('td', { html: `${w.name}<br><span class=\"small\">${w.item}</span>` }), h('td', { class: 'r', text: int(w.F) })",
    "const tb = $('#t-watch tbody'), VT = Math.max(...J.watch.map(w => w.vol)); J.watch.forEach(w => tb.append(h('tr', {}, h('td', { html: `${w.name}<br><span class=\"small\">${w.item}</span>` }), volCell(w.vol, VT), h('td', { class: 'r', text: int(w.F) })")
rep("const t26 = $('#t-flag26 tbody'); FI.t26.top.forEach(d => t26.append(h('tr', {}, h('td', { html: d.name + '<br><span class=\"small\">' + d.item + '</span>' }), h('td', { text: d.ptype }), h('td', { class: 'r', text: int(d.F) })",
    "const t26 = $('#t-flag26 tbody'), VF = Math.max(...FI.t26.top.map(d => d.vol)); FI.t26.top.forEach(d => t26.append(h('tr', {}, h('td', { html: d.name + '<br><span class=\"small\">' + d.item + '</span>' }), h('td', { text: d.ptype }), volCell(d.vol, VF), h('td', { class: 'r', text: int(d.F) })")
rep("Shown: the twelve with the highest risk.</p>", "Shown: the twelve with the highest risk. Units a month is the average of the last three months; the bar compares the items in the table.</p>")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
y = 'T = dict(\n'; assert y in t
pre = '''# item volume for the tables that list single items (user: "add an indication of the item volume"): R3 / 4 = the average of the last three months
_r26 = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color").R3
for _w in J["worst"]: _w["vol"] = round(_w["R3"] / 4, 1)
for _w in J["watch"]: _w["vol"] = round(_w["R3"] / 4, 1)
for _w in J["flaginfo"]["t26"]["top"]: _w["vol"] = round(float(_r26[_w["item"]]) / 4, 1)
'''
t = t.replace(y, pre + y, 1); open(b, "w", encoding="utf-8").write(t); print("patched")
