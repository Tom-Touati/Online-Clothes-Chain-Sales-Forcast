"""Fork 2 (sole page editor; D389i): the atlas follows the switch of the submission to the new nx5 (routed new-item tree, refitted
corrections, focus-weighted ranges) and lists candidate_nx6 (hybrid confidence, D389n). eda/19 reads the new nx5 cascade; the panel
'For final inspection: candidate_nx5 (D361)' becomes the submitted new nx5 with the previous systems beside it; the workbook, facts and
pipeline texts name the new nx5. Idempotent."""
import re
p = "eda/19_project_map_data.py"; s = open(p, encoding="utf-8").read()
s = s.replace('C = pd.read_csv("data/final_2026_nx4_cascade.csv", index_col=0)', 'C = pd.read_csv("data/final_2026_nx5_cascade.csv", index_col=0)')   # D389i
s = s.replace('H = pd.read_csv("data/final_2026_nx4_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx5_cascade_hierarchy.csv")')
open(p, "w", encoding="utf-8").write(s)
p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
if "D389i" in s: raise SystemExit("template already patched")
i = s.index('<div class="panel" id="candidate-nx5"'); j = s.index("\n  </div>", s.index('<p class="small muted">Files: <code>Edikted_Forecasting_Exercise_candidate_nx5.xlsx</code>', i)) + len("\n  </div>")
PANEL = '''<div class="panel" id="candidate-nx5" style="scroll-margin-top:60px;border-color:var(--accent)">
    <h3>Submitted since 18 Nov: the new nx5 (D389i)</h3>
    <div class="grid2">
      <ul class="takeaways small">
        <li><b>System:</b> nx5 of D361 (rules, Huber tree, NN, both TabPFN members, the per-location recommender at a quarter weight) where items under six months old get a Huber tree trained with new items' rows counted five times (D389h), the corrections refitted on that system, and ranges recalibrated per age and volume group with last July–October and the latest March–June weighted four times.</li>
        <li><b>Backtest, WAPE ÷ R3</b> (nx4 in brackets): recent 0.3806 (0.3851), held out 0.2926 (0.2948), all 13 0.4357 (0.4399), early 0.6593 (0.6623). Level with D361's nx5 (0.3811 / 0.2924 / 0.4358).</li>
        <li><b>The new-item lift:</b> the 24 targets under six months rise 10.6% (194,622 → 215,244). In the backtest routing moved new items by only 0.3% while the system under-forecast them by 7–12%: the direction holds, the size is new. Stated in the workbook's "Where it is weak".</li>
        <li><b>Workbook:</b> 16 explanations in plain words (what drove the number, what the models said, not adjusted); the Model explanation opens with the 15 data artefacts and their treatment.</li>
        <li><b>Overnight after it</b> (D389j–o): separate tuning of the new-item and old-item trees, a grid of simple versions, full-strength corrections, per-item weights, two-window calibration: none passes the held-out test and the gate. The monthly split was scored for the first time and kept. The one change that held is the confidence column (D389k): candidate_nx6 (D389n) puts level 3 only where the calibrated range gives less than even odds of landing within ±50% (claims met on the held-out origins 0.75 → 0.81); levels 10 · 86 · 4. For the user's decision.</li>
      </ul>
      <div class="tscroll"><table><thead><tr><th>Jul–Oct 2026</th><th class="r">new nx5 (submitted)</th><th class="r">nx5 D361</th><th class="r">nx4</th></tr></thead><tbody>
        <tr><td>July</td><td class="r">177,611</td><td class="r">172,308</td><td class="r">171,591</td></tr>
        <tr><td>August</td><td class="r">181,724</td><td class="r">176,664</td><td class="r">175,788</td></tr>
        <tr><td>September</td><td class="r">127,588</td><td class="r">124,533</td><td class="r">123,969</td></tr>
        <tr><td>October</td><td class="r">133,903</td><td class="r">131,333</td><td class="r">130,453</td></tr>
        <tr style="background:var(--accent-soft)"><td><b>Total</b></td><td class="r"><b>620,826</b></td><td class="r"><b>604,838</b></td><td class="r"><b>601,801</b></td></tr>
        <tr><td>Buying quantity</td><td class="r">634,349</td><td class="r">637,791</td><td class="r">636,953</td></tr>
        <tr><td>Confidence 1 · 2 · 3</td><td class="r">10 · 82 · 8</td><td class="r">10 · 82 · 8</td><td class="r">10 · 82 · 8</td></tr>
        <tr><td>Range of the 99 modelled targets, summed: P10 · P25 · P75 · P90</td><td class="r" colspan="3">385,955 · 477,424 · 705,984 · 886,834</td></tr>
        <tr><td>Item change against nx4 (10th / 50th / 90th percentile)</td><td class="r" colspan="3">−1.0% / +1.3% / +10.8%; largest +20.1% (S18184_LIGHT-BLUE, a launch)</td></tr>
      </tbody></table></div>
    </div>
    <p class="small muted">Files: <code>Edikted_Forecasting_Exercise_filled.xlsx</code> = <code>…_candidate_nx5.xlsx</code> (md5 53784513…), <code>data/final_2026_nx5_cascade.csv</code>, builder <code>eda/181l_final_nx5r_2026.py</code> (base <code>eda/218_base_nx5r.py</code>, corrections <code>eda/218c_geo_nx5r.py</code>, ranges <code>eda/219_ranges_nx5r.py</code>). Kept: <code>…_candidate_nx4.xlsx</code> (the previous submission), <code>…_candidate_nx5_d361.xlsx</code>; candidate <code>…_candidate_nx6.xlsx</code> (<code>eda/181m_final_nx6_2026.py</code>).</p>
  </div>'''
s = s[:i] + PANEL + s[j:]
def rep(a, b):
    global s; assert s.count(a) == 1, (a[:60], s.count(a)); s = s.replace(a, b)
rep("The deliverable is <code>…_filled.xlsx</code>, a copy of <code>…_candidate_nx4.xlsx</code> since D350 (candidate_nx2 with the World Cup hand rule, the D318 confidence rule and the D317 ranges);",
    "The deliverable is <code>…_filled.xlsx</code>, a copy of <code>…_candidate_nx5.xlsx</code> since 18 Nov (D389i: the new nx5 with the routed new-item tree, refitted corrections and focus-weighted ranges; before that candidate_nx4, D350);")
rep('"units forecast, Jul–Oct 2026, 100 items: the submitted workbook (candidate_nx4, D350)"', '"units forecast, Jul–Oct 2026, 100 items: the submitted workbook (new nx5, D389i)"')
rep('what: "A copy of candidate_nx4 since D350: candidate_nx2 with the World Cup hand rule restored (D303), confidence by the validated rule (D318) and ranges widened upward (D317).',
    'what: "A copy of the new nx5 since 18 Nov (D389i): nx5 with the routed new-item tree, refitted corrections and focus-weighted ranges, plain explanations and the data artefacts in the Model explanation; before that candidate_nx4 (D350).')
rep('s: "181a_base2026 · 181g_final_nx2_2026.py · 181h_final_nx3_2026.py · 181i_final_nx4_2026.py"', 's: "181a_base2026 · 181i_final_nx4_2026.py · 181k_final_nx5_2026.py · 218_base_nx5r · 219_ranges_nx5r · 181l_final_nx5r_2026.py"')
s = s.replace("<!--D389i-->", "") + "<!--D389i-->"
open(p, "w", encoding="utf-8").write(s); print("patched")
