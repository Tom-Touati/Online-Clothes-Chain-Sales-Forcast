import sys, json
D = sys.argv[1]; s = open(f"{D}/files/index.html", encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b, 1)
rep('            <th data-field="bias">Bias</th>\n',
    '            <th data-field="bias">Bias</th>\n'
    '            <th data-field="within20">Items within ±20%</th>\n'
    '            <th data-field="within20_units">Units within ±20%</th>\n'
    '            <th data-field="median_ape">Median % error</th>\n'
    '            <th data-field="r2_units">R² (units)</th>\n'
    '            <th data-field="r2_log">R² (log)</th>\n'
    '            <th data-field="abs_wape">Error ÷ units</th>\n'
    '            <th data-field="wape_sea">WAPE ÷ seasonal naive</th>\n'
    '            <th data-field="mdrae">Median item ÷ R3</th>\n'
    '            <th class="sb-l" data-field="blk_rules">Origin-block 95% interval vs R3 + rules</th>\n'
    '            <th data-field="early_wape_r3">Four early origins: WAPE ÷ R3</th>\n')
rep('          <option value="bf">Black Friday origins (Jul–Oct 2025)</option>\n',
    '          <option value="bf">Black Friday origins (Jul–Oct 2025)</option>\n'
    '          <option value="early">Four early origins (Feb–May 2025)</option>\n'
    '          <option value="s_spring">Spring origins (Feb–Apr)</option>\n'
    '          <option value="s_summer">Summer origins (May–Jul)</option>\n'
    '          <option value="s_autumn">Autumn origins (Aug–Oct)</option>\n'
    '          <option value="s_winter">Winter origins (Nov–Jan)</option>\n')
rep("      if (r.licence === 'non-commercial') pill('non-commercial licence');",
    "      if (r.licence === 'non-commercial') pill('non-commercial licence');\n"
    "      if (r.gate_early === false) pill('fails the early-origin gate');")
rep("    for (const [k, dir] of cols) {\n",
    "    for (const [k, dir] of cols) {\n      if (dir === 'none') continue;\n")
rep('<th data-field="relmae_r3">Typical item ÷ R3</th>', '<th data-field="relmae_r3">Typical item ÷ R3 (every item counts equally)</th>')
rep('<th data-field="win_r3">Wins vs R3</th>', '<th data-field="win_r3">Items closer than R3</th>')
rep("['mab_r3', 'low', F.ratio], ['pinball_r3', 'low', F.ratio], ['bias', 'zero', F.spct], ['sim_loss', 'low', F.pct2], ['sim_loss_r3', 'low', F.ratio],",
    "['mab_r3', 'low', F.ratio], ['pinball_r3', 'low', F.ratio], ['bias', 'zero', F.spct],\n"
    "      ['within20', 'high', F.pct1], ['within20_units', 'high', F.pct1], ['median_ape', 'low', F.pct1], ['r2_units', 'high', F.ratio], ['r2_log', 'high', F.ratio],\n"
    "      ['abs_wape', 'low', F.pct1], ['wape_sea', 'low', F.ratio], ['mdrae', 'low', F.ratio], ['blk_rules', 'none', (v) => (v == null || v === '' ? '—' : String(v))], ['early_wape_r3', 'low', F.ratio],\n"
    "      ['sim_loss', 'low', F.pct2], ['sim_loss_r3', 'low', F.ratio],")
rep('<dt>Buying loss</dt><dd>Profit lost',
    '<dt>Within ±20%</dt><dd>Share of item forecasts whose 4-month total landed within ±20% of actual (the assignment\'s confidence-1 band); "units" weighs each item by its actual units.</dd>\n'
    '      <dt>Median % error</dt><dd>The middle item\'s |forecast − actual| ÷ actual: how far off a typical item is, in percent.</dd>\n'
    '      <dt>Error ÷ units</dt><dd>Total absolute error divided by total actual units (plain WAPE, not relative to R3).</dd>\n'
    '      <dt>WAPE ÷ seasonal naive</dt><dd>Error relative to a seasonal naive forecast: R3\'s monthly level shaped by the product type\'s like-for-like seasonal pattern.</dd>\n'
    '      <dt>Median item ÷ R3</dt><dd>The middle item\'s (|error| + 1) ÷ (|R3 error| + 1); below 1 means the typical item is closer than R3.</dd>\n'
    '      <dt>Origin-block interval</dt><dd>95% interval of the difference in WAPE ÷ R3 against R3 + rules when whole origins, not items, are resampled. Origins overlap, so it is wider and more honest than the item bootstrap.</dd>\n'
    '      <dt>Early-origin gate</dt><dd>A system that beats the submitted one on the nine recent origins must also beat it on the four early origins (Feb–May 2025), or it is marked as failing the gate. Removing last year\'s same window from training is checked separately.</dd>\n'
    '      <dt>R²</dt><dd>Share of the variation in actual 4-month totals that the forecast explains, in units (dominated by big items) and in logs (every item counts alike).</dd>\n'
    '      <dt>Buying loss</dt><dd>Profit lost')
open(f"{D}/files/index_new.html", "w", encoding="utf-8").write(s)
json.dump({"text": s}, open(f"{D}/index_new.json", "w", encoding="utf-8")); print(len(s))
