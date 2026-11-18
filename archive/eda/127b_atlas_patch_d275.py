"""Atlas template patch (D275): brings eda/atlas_template.html from the D132 state to the submitted system of D274.
Header facts, the project map (current members, stack, corrections, buying column, final chain, explanation layer), a 'current
system' panel and the rounds after D132 in Modelling, the experiment ledger (rounds 6-12), the confidence rule, the evaluation
section's captions and data hooks (J["eval"] from eda/124, J["system"] from eda/19), the forecast table (cascade per item, buying
quantity), unique ids in the decision log, the run order and links. Idempotent: does nothing if the marker is already present."""
import re, sys
sys.stdout.reconfigure(encoding="utf-8")
p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
if "atlas-d275" in s: print("already patched"); sys.exit(0)
N = [0]
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new); N[0] += 1
def between(a, b, new, keep_a=True, keep_b=True):
    global s
    i = s.index(a); j = s.index(b, i + len(a)); assert s.count(a) == 1, a[:80]
    s = s[:i] + (a if keep_a else "") + new + (b if keep_b else "") + s[j + len(b):]; N[0] += 1
chip = lambda *ids: " ".join(f'<button class="chip dl" type="button" data-d="{d}">{d}</button>' for d in ids)

# ---------------- header ----------------
rep("state as of 16 Nov 2026", "state as of decision D274, 17 Nov 2026")
rep('<p class="lede">Everything you need to pick this project up: where the data comes from, what the EDA found, which modelling ideas were tested and dropped, how the 100-item forecast was built, and where every file lives.',
    '<p class="lede" data-v="atlas-d275">Everything you need to pick this project up: where the data comes from, what the EDA found, which modelling ideas were tested and dropped, how the submitted 100-item forecast is built from five models, and where every file lives.')
between("const facts = [", "];\n$(\"#facts\")", '''[fmt(tot), "units forecast, Jul–Oct 2026, 100 items: the submitted workbook (candidate_nx2, D274)"], [SY.recent.toFixed(3) + " × R3", "backtest error against recency (last 3 months × 4), nine recent origins; " + SY.all13.toFixed(3) + " on all 13"], [fmt(SY.buy), "buying quantity for the same items (quantile members at P65)"], [cc, "items at confidence 1 · 2 · 3"], [fmt(SY.orig), "the original 12 Nov submission, kept as a backup file"], [String(D.log.length), "logged decisions in log.md"]''')
rep("const tot = D.targets.reduce((a, t) => a + t.fin, 0), raw = D.targets.reduce((a, t) => a + t.raw, 0);", "const SY = D.system || {}; const tot = D.targets.reduce((a, t) => a + t.fin, 0), raw = D.targets.reduce((a, t) => a + (t.raw || 0), 0);")

# ---------------- project map ----------------
rep('''outs: ["data/targets.csv", "The filled copy"], nums: ["6 campaigns planned, 28 Jun – 18 Oct", "Confidence scale: ±20% / ±50% / worse on the 4-month total"], dec: ["D040"], go: "#campaigns" } ]],''',
    '''outs: ["data/targets.csv", "The filled copy"], nums: ["6 campaigns planned, 28 Jun – 18 Oct", "Confidence scale: ±20% / ±50% / worse on the 4-month total"], dec: ["D040"], go: "#campaigns" },
    { id: "outside", kind: "file", t: "Outside and derived signals", s: "Temperature, events, carts, LLM scores, photos", what: "Signals added after the first submission: NOAA statewide temperature normals and anomalies; a US calendar of 31 event windows; cart co-purchase pairs from the order lines, fitted as a network at every origin; 14 LLM-scored fashion fields per item; the public product feed (photos and text) for the content embedding.", ins: ["NOAA NCEI", "eda/161_events.py · eda/180_newfeats.py", "Dataset.parquet (baskets)", "edikted.com product feed"], outs: ["data/weather_state_month.parquet", "data/events_month.parquet", "data/fk_dyn_feats.parquet", "data/llm/scores_part*.csv", "data/content_emb.npz"], nums: ["Temperature + events: three-way 0.4365 → 0.4234 (D159)", "Dynamic co-purchase neighbours: 0.4222 → 0.4202 (D203)", "Embeddings must predict carts before use: content v4 0.431 vs 0.180 for raw product features", "As tree features the content neighbours add nothing (D205)"], dec: ["D153", "D159", "D160", "D164", "D170", "D193", "D198", "D205"], go: "#modelling" } ]],''')
between('  ["Model", [', '];\nconst flow = $("#flow")', '''
    { id: "feat", t: "Feature tables", s: "06_features · 162 · 182 · 180_newfeats", what: "One row per item × month-end origin, built only from data up to the origin: momentum lags, like-for-like seasonality, occasion timing, statewide temperature, 31 event windows, dynamic co-purchase neighbours and launch curves. The tree member drops the static descriptors (attributes, type, LLM scores); the TabPFN members keep them.", ins: ["monthly_item.parquet", "daily_item_chan.parquet", "Outside and derived signals"], outs: ["data/features.parquet", "data/fk_explain_table.parquet", "data/fk_dyn_feats.parquet", "data/nf2_feats.parquet"], nums: ["28 origins, Mar 2024 – Jun 2026", "Target: log((4-month units + 1) / (4 × level + 1))", "Rebuilt from data cut at the origin: every feature matches (D254)"], dec: ["D009", "D135", "D159", "D198", "D252", "D254"], go: "#modelling" },
    { id: "bt", t: "Backtest harness", s: "13 origins · ratios to R3 · buying simulation", what: "Every system forecasts the next four months at 13 month-end origins (Feb 2025 – Feb 2026) from data known at the time; training rows need their outcome known by the origin. Scores are ratios to R3 (last 3 months × 4), with R3 + rules as a row and paired item bootstraps for small differences. A daily buying simulation measures profit lost against perfect foresight.", ins: ["Feature tables"], outs: ["data/q_backtest_*.parquet", "data/dec_base_*.parquet", "The scoreboard"], nums: ["11,727 item-origins; 9,031 on the nine recent origins", "R3 itself: WAPE 0.699 on the recent origins", "R3 loses 19.5% of perfect-foresight profit in the buying simulation"], dec: ["D010", "D109", "D110", "D111", "D154"], go: "#evaluation" },
    { id: "m_rules", t: "Member 1: rule agents", s: "R3 × explicit rules, rewritten monthly", what: "Isolated agents write explicit rules on top of each item's recent rate at every origin, from data up to that origin only. It is also the reference row of every scoring table.", ins: ["Data up to the origin"], outs: ["agent_eval/<origin>/predictions_verified.csv"], nums: ["0.488 × R3 alone (recent)", "Its monthly profile, with the NN's, splits the total into months (D121)"], dec: ["D105", "D109", "D121"], go: "#evaluation" },
    { id: "m_tree", t: "Member 2: LightGBM", s: "180q_models.py ec_ns", what: "Quantile LightGBM on the 4-month total, base and full feature sets averaged: the lean model with occasion timing, temperature and events, dynamic co-purchase neighbours, 15 more event windows and launch curves. Static descriptors removed.", ins: ["Feature tables"], outs: ["data/q_backtest_nx_ec_ns.parquet"], nums: ["0.430 × R3 alone (recent)", "A model that never saw the item's style: +1% only", "Its volume-weighted version at P65 feeds the buying column"], dec: ["D135", "D159", "D198", "D253", "D254", "D264"], go: "#modelling" },
    { id: "m_nn", t: "Member 3: sales-curve NN", s: "57_embed_nn.py · 197b_nn_ec.py", what: "Neural quantile forecaster with hierarchical embeddings (group, type, style, item) and a GRU over the last 26 weeks of sales, with the event and launch-curve inputs added. Six seeds.", ins: ["features.parquet", "daily_item_chan.parquet", "nf2_feats.parquet"], outs: ["data/nn_backtest_preds_S13ecx6.parquet"], nums: ["0.459 × R3 alone (0.484 before the new inputs)", "In the stack: 0.3961 → 0.3939"], dec: ["D076", "D096", "D272", "D273"], go: "#modelling" },
    { id: "m_pfn", t: "Members 4–5: TabPFN v2", s: "196b_tabpfn.py · 199c_conformal.py", what: "A tabular foundation model used in context on the rows whose outcome is known: one version with 32 estimators, one with conformally calibrated quantiles. TabPFN v2's licence allows production use; TabPFN 3.5 is non-commercial and is not used.", ins: ["fk_explain_table.parquet", "fk_dyn_feats.parquet"], outs: ["data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet", "data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"], nums: ["0.463 and 0.429 × R3 alone (recent)", "Rebuilt on the new features: better alone, level in the stack (D270)"], dec: ["D204", "D206", "D207", "D270"], go: "#modelling" },
    { id: "stack", t: "Stack and corrections", s: "arithmetic mean · 179x2_bias · 179b_breakout · 180g_geo", what: "The arithmetic mean of the five members in units, then two item-level corrections combined by geometric mean with each origin's total kept: past segment biases (volume × age × discount) and a breakout model (weekly momentum, look-peer growth, falling-item signals).", ins: ["The five members"], outs: ["data/dec_base_ecnsnn.parquet", "data/geo_ecnsnn_x_preds.parquet"], nums: ["Stack 0.394 → corrected 0.385 × R3 (recent)", "0.440 on all 13 origins · 0.371 on the 100 targets", "Bias −6.2% recent, −1.7% on all 13"], dec: ["D230", "D231", "D245", "D247", "D248", "D259", "D273"], go: "#evaluation" },
    { id: "buy", t: "Buying column", s: "newsvendor ladder at P65", what: "The same stack with the quantile members (volume-weighted LightGBM, both TabPFN versions) at their 65th percentile. Buying loss follows bias more than WAPE, and the loss curve bottoms at P65–P70.", ins: ["The five members' quantiles"], outs: ["Buying quantity per item"], nums: ["Buying loss 10.4%, against 10.7% for the point forecast and 19.5% for R3", "2026: 655,940 units against 620,608 forecast"], dec: ["D233", "D236", "D260", "D265"], go: "#forecast" } ]],
  ["Deliver", [
    { id: "final", t: "Final 2026 chain", s: "181a_base2026 · 181g_final_nx2_2026.py", what: "Members refitted at the June 2026 origin on every row with a known outcome; corrections re-fitted on all 13 backtest origins; the total split into months with the rule agent's and the NN's monthly profiles. One override: the out-of-stock item keeps its in-stock forecast of 130 units.", ins: ["June 2026 members", "Corrections"], outs: ["data/final_2026_nx2_cascade.csv", "Edikted_Forecasting_Exercise_candidate_nx2.xlsx"], nums: ["Five-model average 626,076 → forecast 620,602 before monthly rounding", "Confidence 1 · 2 · 3 = 20 · 67 · 13", "No campaign, World Cup or segment rules on top"], dec: ["D034", "D121", "D235", "D237", "D262", "D273"], go: "#forecast" },
    { id: "explain", t: "Explanation layer", s: "198b_explain_final.py · cascade sheets", what: "TreeSHAP of the tree member's deviation from R3, grouped into 11 named categories by an LLM categoriser and checked by three LLM judges. The workbook's cascade sheets show every layer from R3 to the buying quantity by item, style, type and total.", ins: ["Tree member", "The cascade"], outs: ["data/fk_2026_explain.parquet", "Cascade (items) · Cascade (hierarchy)"], nums: ["The categories explain 94% of the stack's deviation from R3", "Direction right for 92% of items", "16 written explanations in the workbook"], dec: ["D191", "D194", "D195", "D210", "D235"], go: "#forecast" },
    { id: "out", kind: "deliv", t: "Submitted workbook", s: "Edikted_Forecasting_Exercise_filled.xlsx", what: "A copy of candidate_nx2 since D274. Forecast sheet: 100 items × 4 months, confidence, explanations; the Model explanation sheet; two cascade sheets. The original 12 Nov submission is kept as Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx.", ins: ["Final 2026 chain"], outs: ["The submission"], nums: ["620,608 units: Jul 176,453 · Aug 186,211 · Sep 126,019 · Oct 131,925", "Original submission: 580,968", "Switched at D271 (candidate_nx) and D274 (candidate_nx2)"], dec: ["D040", "D271", "D273", "D274"], go: "#forecast" },
    { id: "docs", kind: "deliv", t: "Reports, log, scoreboard", s: "log.md · reports · scoreboard · this page", what: "The append-only decision log; the reports (EDA, error analysis, M5 round, decomposition, fixable errors, explainability); the live scoreboard of every scored system; and this atlas.", ins: ["All of the above"], outs: ["Hand-over"], nums: [D.log.length + " logged decisions", "Scoreboard: every system against R3, with buying loss"], dec: ["D019", "D041", "D132", "D154", "D275"], go: "#decisions" } ]]
''', keep_b=True)
rep("sel(nodes.eda);", "sel(nodes.stack);")

# ---------------- targets profile ----------------
rep("Confidence comes from each item's calibrated range, not from age (D046); young items mostly land at 2 or 3 because their ranges are wide.",
    "Confidence comes from the width of each item's calibrated range and from how far the five models disagree, not from age (D237, D266).")

# ---------------- modelling ----------------
rep("<h2>Five rounds of testing, one quantile model plus checks</h2>", "<h2>From one quantile model to a stack of five</h2>")
rep("Scoring is on the four-month total per item, the unit the confidence scale uses. Differences under about 0.02 WAPE are treated as noise.</p>",
    "Scoring is on the four-month total per item, the unit the confidence scale uses. Since D109 every system is scored as a ratio to R3 (each item's last 3 months × 4), and small differences are judged by a paired item bootstrap.</p>" + f'''
  <div class="panel" id="current" style="scroll-margin-top:60px">
    <h3>The submitted system (D274)</h3>
    <div class="grid2">
      <ul class="takeaways small">
        <li><b>Anchor:</b> R3, then the product type's like-for-like seasonality.</li>
        <li><b>Five members, each forecasting the 4-month total:</b> the rule agents; a quantile LightGBM with events, co-purchase neighbours and launch curves; a sales-curve neural net; TabPFN v2 with 32 estimators; TabPFN v2 with conformally calibrated quantiles.</li>
        <li><b>Stack:</b> the arithmetic mean of the five, in units. The geometric mean forecast low and bought worse (D230).</li>
        <li><b>Corrections:</b> a segment re-allocation (volume × age × discount) and a breakout model, combined by geometric mean with each origin's total kept (D245, D247, D259).</li>
        <li><b>Months:</b> the total is split with the rule agent's and the NN's monthly profiles (D121).</li>
        <li><b>Buying quantity:</b> the same stack with the quantile members at P65 (D236).</li>
        <li><b>Confidence:</b> the conformal TabPFN member's P25–P75 width, one level worse where the members differ by more than 2× (D237).</li>
      </ul>
      <div class="tscroll"><table id="sys-tab"><thead><tr><th>Submitted system, backtest</th><th class="r">Value</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>
  <div class="panel" id="rounds">
    <h3>What changed after the first submission</h3>
    <p class="small muted">The 12 Nov submission was the quantile model below plus judgement rules (580,968 units). Each round was tested on the same 13 origins; decision ids open the log.</p>
    <ul class="takeaways small">
      <li><b>Rule agents, NN and the three-way stack.</b> Explicit rules on the recent rate, a neural net with hierarchical embeddings and a sales-curve encoder, combined with the quantile model: 0.440 × R3. A lean feature set, occasion timing and a Black Friday fix followed. {chip("D076", "D096", "D105", "D134", "D135", "D141")}</li>
      <li><b>Weather and events.</b> Statewide temperature and a US events calendar, kept alongside seasonality: three-way 0.4234; with the Black Friday fix 0.4222 and 11.05% buying loss. Replacing seasonality outright was worse. {chip("D155", "D159", "D161")}</li>
      <li><b>Carts, style embeddings and the cart gate.</b> Co-purchase neighbours and LLM fashion scores helped (0.4326). Rule from D160: an embedding must predict cart co-occurrence better than raw product features before it is used. Content embeddings v4 and v6 (with photos) pass; a graph network is level with them; as tree features they add nothing. A Poisson co-purchase network re-fitted at every origin does help: 0.4202. {chip("D150", "D153", "D160", "D164", "D168", "D180", "D193", "D198", "D199", "D203", "D208")}</li>
      <li><b>Explainability.</b> SHAP of each item's deviation from R3, grouped into 11 categories by an LLM and scored by three LLM judges; simple surrogates on the categories recover 72–85% of the lift. For the submitted stack the categories explain 94% of the deviation. {chip("D191", "D194", "D195", "D202", "D210")}</li>
      <li><b>Foundation models.</b> Time-series models (Chronos-2, Bolt, TimesFM) rejected at 0.63–0.76 alone. TabDPT, Mitra, Nori and TabICL rejected or conditional. TabPFN v2 adopted as a member: 0.4100 recent, 10.78% buying loss. {chip("D192", "D196", "D197", "D201", "D204")}</li>
      <li><b>The M5 round.</b> Rejected: hierarchical reconciliation, campaign-calendar features, learned combination weights, Google Trends, multi-horizon neural nets, augmentation, weekly origins, other boosting libraries, tuning, bagging, cluster and trend features. Adopted: conformal TabPFN quantiles, the arithmetic stack, and the P65 buying column (buying loss follows bias, not WAPE). {chip("D172", "D173", "D175", "D177", "D178", "D183", "D205", "D206", "D207", "D230", "D231", "D236", "D237")}</li>
      <li><b>Decomposition and errors.</b> Looks hold twice the shared error of product types, but no group-level model beats the sum of items. The error is concentrated in young launches that break out into the peak. Two corrections with totals kept moved the system to 0.3902. {chip("D238", "D240", "D244", "D245", "D247", "D248", "D250")}</li>
      <li><b>Fixable errors.</b> The tree member had no age feature. Adding 15 event windows and launch curves and dropping the static descriptors: 0.3961 before corrections, 0.3871 after. Leakage and memorisation checks clean; a survivorship limit found. A launch plan at look level would be worth about 1% but needs Edikted's calendar. {chip("D252", "D253", "D254", "D259", "D260", "D264", "D265")}</li>
      <li><b>Rebuilds and switches.</b> TabPFN rebuilt on the new features is better alone but level in the stack; the NN rebuilt on them is better: 0.3851 recent, 0.4399 on all 13. The submission was switched to candidate_nx and then to candidate_nx2. {chip("D262", "D270", "D271", "D272", "D273", "D274")}</li>
    </ul>
  </div>''')
rep("<h3>Current model: quantiles of the 4-month total</h3>", "<h3>History: the quantile model behind the 12 Nov submission</h3>")
rep("<h3>Monthly model (now only splits totals into months)</h3>", "<h3>History: the monthly model of rounds 1–3</h3>")
rep("Round 4: quantiles, with the answer chosen from bias and confidence from the IQR.", "Round 4: quantiles, with the answer chosen from bias and confidence from the IQR. Round 5: the common-sense audit. Rounds 6–12: everything after the first submission, scored as WAPE ÷ R3 on the nine recent origins.")
between("    <h3>How confidence levels are set: from the calibrated IQR</h3>\n", "  </div>\n</div></section>\n\n<section class=\"block\" id=\"evaluation\">", '''    <div class="grid2">
      <div class="panel">
        <p class="small">The assignment's scale: 1 = the 4-month total lands within ±20% of actual, 2 = within ±50%, 3 = off by more. The submitted rule uses the conformal TabPFN member's range around its median, w = (P75 − P25) ÷ (2 × P50):</p>
        <ul class="takeaways small">
          <li><span class="pill c1">1</span> w ≤ 0.2: 20 targets.</li>
          <li><span class="pill c2">2</span> w ≤ 0.5: 67 targets.</li>
          <li><span class="pill c3">3</span> wider, or one level worse where the five members differ by more than 2×: 13 targets.</li>
        </ul>
        <p class="small muted">The stated levels rank items correctly but only weakly (quadratic kappa +0.11). The member's P10–P90 range, put on the final forecast, covers 80.6% of actuals (D266, D273).</p>
      </div>
      <div class="tscroll"><table id="conf"><thead><tr><th>Backtest, by stated level</th><th class="r">Share</th><th class="r">Within ±20%</th><th class="r">Within ±50%</th><th class="r">Off &gt; 50%</th></tr></thead><tbody>
        <tr><td><span class="pill c1">1</span> nine recent origins</td><td class="r">2.9%</td><td class="r">53.5%</td><td class="r">92.2%</td><td class="r">7.8%</td></tr>
        <tr><td><span class="pill c2">2</span> nine recent origins</td><td class="r">73.2%</td><td class="r">44.2%</td><td class="r">82.6%</td><td class="r">17.4%</td></tr>
        <tr><td><span class="pill c3">3</span> nine recent origins</td><td class="r">23.9%</td><td class="r">32.6%</td><td class="r">68.1%</td><td class="r">31.9%</td></tr>
        <tr><td colspan="5" class="small muted">Submitted system, 9,031 item-origins (eda/confidence_output_ecnsnn.txt). On all 13 origins the three levels land within ±20% 56%, 42% and 29% of the time.</td></tr>
      </tbody></table></div>
    </div>
''')
rep("    <h3>How confidence levels are set: from the calibrated IQR</h3>", "    <h3>How confidence levels are set, and how they score</h3>")

# ---------------- evaluation ----------------
rep('<p class="small muted">Seasonality alone removes about 38% of recency\'s error; the quantile model\'s other learning another 15%; the NN and the rules about 1.5 points each.</p>', '<p class="small muted" id="ev-fva-note"></p>')
rep('<p class="small muted">Rules + quantile + NN candidate. Model steps are TreeSHAP groups of the June 2026 quantile model, in log ratio to the recent-rate anchor.</p>',
    '<p class="small muted">The submitted cascade (D273). The five members are alternatives, not steps: each bar is that layer\'s total for the 100 targets. Hover a bar for its ratio to R3.</p>')
rep("<th>Archetype (recent)</th>", "<th>Segment (recent, submitted system)</th>")
rep("<h3>Bias by target month</h3>", "<h3>Bias by target month (the three members of the D132 system)</h3>")
rep("November 2025 (Black Friday) is under-forecast by every member.</p>", "November 2025 (Black Friday) is under-forecast by every member. On the five-member stack the monthly split over-forecasts September by 28% and under-forecasts November by 19% and January by 14% (D244).</p>")
rep("<h3>Bias by origin and product group (rules + quantile + NN)</h3>", "<h3>Bias by origin and product group (submitted system)</h3>")
rep("<h3>What each kind of information is worth (quantile model, one group removed at a time)</h3>", "<h3>What each kind of information is worth (quantile model at D134, one group removed at a time)</h3>")
rep("Seasonality carries most of the value; channel mix, discount, occasion flags and lifecycle add nothing, so the lean model drops them.</p>",
    "Seasonality carries most of the value; channel mix, discount, occasion flags and lifecycle add nothing, so the lean model drops them. On the later tree member, 15 more event windows and launch curves helped; age, type trends and peak features were level; removing the static descriptors was level on recent origins and better on all 13 (D253).</p>")
rep("Buying loss = profit lost against perfect foresight in the daily buying simulation (11 windows, production buying policy), recent scope only.</p>",
    "Buying loss = profit lost against perfect foresight in the daily buying simulation (11 windows, production buying policy), recent scope only. The highlighted row is the lowest WAPE: the submitted system. Its buying quantity (P65 ladder) loses <span id=\"ev-ladder\"></span>.</p>")

# ---------------- forecast ----------------
between("      <h3>Judgement on top of the model</h3>\n", "    </div>\n  </div>\n  <div class=\"sub\">\n    <div class=\"controls\">\n      <input type=\"search\" id=\"t-q\"", '''      <ul class="takeaways small" id="fc-notes">
        <li><b>Point forecast:</b> the arithmetic mean of the five members at the June 2026 origin, times the segment and breakout corrections (D273). <span id="fc-corr"></span></li>
        <li><b>Months:</b> each total is split with the geometric mean of the rule agent's and the NN's monthly profiles (D121).</li>
        <li><b>Buying quantity:</b> the stack with the quantile members at P65, shown per item in the table (D236).</li>
        <li><b>One override:</b> <code>S16079_DARK-GRAY</code>, out of stock since April, keeps its in-stock forecast of 130 units; no model member covers it (D034).</li>
        <li><b>No judgement rules after the stack.</b> The campaign uplift, World Cup caps and segment caps of the 12 Nov submission (D031, D033, D051–D053) are not applied. Event effects enter only through the June 2026 rule agent and the event windows.</li>
        <li><b>World Cup items:</b> the three targets total <span id="fc-wc"></span> here, against <span id="fc-wc0"></span> under the tournament-end rule of the 12 Nov submission. The largest item, <code>S24840_LIGHT-GRAY-MELANGE</code>, is forecast higher in August than in July although the final is on 19 July.</li>
        <li><b>Largest single-item risks:</b> Vikkey (<code>S23794_BLACK</code>) at about 30k, with members from 22k to 43k; the items where the members differ by more than 2× (<span id="fc-sp"></span> targets).</li>
      </ul>
''', keep_a=False)
rep('<div class="panel">\n      <ul class="takeaways small" id="fc-notes">', '<div class="panel">\n      <h3>What the submitted numbers contain</h3>\n      <ul class="takeaways small" id="fc-notes">')
rep('<th class="sortable r" data-k="fin">Total</th><th class="sortable r" data-k="adj">vs raw</th>', '<th class="sortable r" data-k="fin">Total</th><th class="sortable r" data-k="buy">Buy (P65)</th><th class="sortable r" data-k="vr3">vs R3</th>')
rep("Hover or focus a row for the item's history, the structured baseline and the explanation written into the workbook. Click a column header to sort.", "Hover or focus a row for the item's history, its cascade from R3 through the five members to the forecast, and the explanation written into the workbook. Click a column header to sort.")
rep('''$("#fc-scale").textContent = `For scale: June 2026 × 4 for the same items is ${fmt(junTot * 4)}. Raw model answer ${fmt(raw)}; adjustments take it to ${fmt(tot)} (${spct(tot / raw - 1)}). The version before the common-sense audit (61st percentile) was ${fmt(prevTot)}.`;''',
    '''$("#fc-scale").textContent = `For scale: June 2026 × 4 for the same items is ${fmt(junTot * 4)}; R3 (last 3 months × 4) is ${fmt(sumq("r3"))}. Five-model average ${fmt(sumq("avg5"))}; the corrections take it to ${fmt(tot)}. Buying quantity ${fmt(SY.buy)}. The original 12 Nov submission was ${fmt(prevTot)}.`;
const q10 = a => { const v = a.filter(x => x != null).sort((x, y) => x - y); return [0.1, 0.5, 0.9].map(q => v[Math.round(q * (v.length - 1))].toFixed(2)).join(" / "); };
$("#fc-corr").textContent = `Correction multiplier on the targets, 10th / 50th / 90th percentile: ${q10(TG.map(t => t.corr))}.`;
const WC = TG.filter(t => /World Cup/.test(t.occ)); $("#fc-wc").textContent = fmt(WC.reduce((a, t) => a + t.fin, 0)); $("#fc-wc0").textContent = fmt(WC.reduce((a, t) => a + t.prev, 0)); $("#fc-sp").textContent = TG.filter(t => t.spread > 2).length;''')
rep("TG.forEach((t, i) => { t.i = i + 1; t.f0 = t.f[0]; t.f1 = t.f[1]; t.f2 = t.f[2]; t.f3 = t.f[3]; t.adj = t.fin / t.raw - 1; });",
    "TG.forEach((t, i) => { t.i = i + 1; t.f0 = t.f[0]; t.f1 = t.f[1]; t.f2 = t.f[2]; t.f3 = t.f[3]; t.vr3 = t.r3 > 20 ? t.fin / t.r3 - 1 : 0; t.buy = t.buy == null ? t.fin : t.buy; });")
between("function ttip(t) {\n", "function renderT() {", '''  const row = (k, v) => `<tr><td>${k}</td><td>${v == null ? "–" : fmt(v)}</td></tr>`;
  return `<div class="tt">${esc(t.name)}</div><span class="mono">${t.sc}</span> · ${esc(t.type)} · published ${t.pub}${t.occ ? " · " + esc(t.occ) : ""}${t.cn > 1 ? " · colour " + t.cn : ""}
  <table>${row("June 2026 actual", t.jun)}${row("Last 12 months", t.l12)}${row("Jul–Oct 2025 actual", t.jo25)}${row("R3: last 3 months × 4", t.r3)}${row("× type seasonality", t.season)}
  ${row("Rule agents", t.rules)}${row("LightGBM", t.tree)}${row("Sales-curve NN", t.nnm)}${row("TabPFN v2, conformal", t.pf2c)}${row("TabPFN v2, 32 estimators", t.pf2d)}${row("Average of the five", t.avg5)}
  <tr><td>Correction</td><td>${t.corr == null ? "–" : "× " + t.corr.toFixed(3)}</td></tr><tr><td><b>Forecast</b></td><td><b>${fmt(t.fin)}</b>${t.r3 > 20 ? " (" + spct(t.vr3) + " vs R3)" : ""}</td></tr>${row("Buying quantity (P65)", t.buy)}
  <tr><td>P10–P90</td><td>${t.p10 == null ? "–" : fmt(t.p10) + " – " + fmt(t.p90)}</td></tr>${row("12 Nov submission", t.prev)}</table><div style="margin-top:4px">Confidence ${t.conf}${t.spread != null ? ` · members differ by ×${t.spread.toFixed(2)}` : ""}</div>${t.expl ? `<div style="margin-top:6px">${esc(t.expl)}</div>` : ""}`;
}
''')
rep('''<td class="r"><b>${fmt(t.fin)}</b></td><td class="r small ${Math.abs(t.adj) > 0.05 ? "" : "muted"}">${spct(t.adj)}</td>''', '''<td class="r"><b>${fmt(t.fin)}</b></td><td class="r">${fmt(t.buy)}</td><td class="r small">${t.r3 > 20 ? spct(t.vr3) : "–"}</td>''')

# ---------------- experiments: rounds 6-12 ----------------
rep('''  ["Blend toward the exit pace for fading items", "5", "—", "drop", "Recent origins show no over-forecast for them (bias about 0)", "D056"]];''',
'''  ["Blend toward the exit pace for fading items", "5", "—", "drop", "Recent origins show no over-forecast for them (bias about 0)", "D056"],
  ["R3 (last 3 months × 4) as the reference for every score", "6", "1.000", "baseline", "From D109 every score is WAPE ÷ R3 on the nine recent origins; R3's own WAPE is 0.699", "D109"],
  ["Rule agents: R3 × explicit rules rewritten at each origin", "6", "0.488", "keep", "Member and reference row of every table", "D105"],
  ["NN with hierarchical embeddings and a sales-curve encoder", "6", "0.484 alone", "keep", "Third member; small, consistent gain from the 26-week GRU", "D096"],
  ["Rules × quantile × NN, geometric mean", "6", "0.440", "keep", "The three-way system; lean features, occasion timing and the Black Friday fix took it to 0.433", "D141"],
  ["Temperature + events calendar alongside seasonality", "7", "0.4234", "keep", "With the Black Friday fix 0.4222, buying loss 11.05%; replacing seasonality outright is worse", "D159"],
  ["Cart co-purchase neighbours + LLM fashion scores", "7", "0.4326", "keep", "Later superseded by the dynamic network", "D153"],
  ["Multi-view clusters; per-style embedding; residual-distribution member", "7", "level", "drop", "No gain as features or members", "D158"],
  ["Dynamic co-purchase network, re-fitted at each origin", "8", "0.4202", "keep", "Deviance explained on next-month carts 0.50 vs 0.27 static", "D203"],
  ["Time-series foundation models (Chronos-2, Bolt, TimesFM)", "8", "0.63–0.76 alone", "drop", "Worse as a fourth member too", "D192"],
  ["TabDPT, Mitra, Nori, TabICL", "8", "0.455 or worse alone", "drop", "TabDPT conditional; the rest worse than R3 at a 3k context", "D201"],
  ["TabPFN v2 as a fourth member", "8", "0.4100", "keep", "All 13 origins 0.4586; buying loss 10.78%; production licence", "D204"],
  ["Cluster-history, attribute-trend, trend-cycle, nearest-neighbour features", "9", "0.4427 vs 0.4315 alone", "drop", "All level or worse, in the trees and in TabPFN", "D205"],
  ["CatBoost, XGBoost, DART, Tweedie, tuning, bagging, TabPFN fine-tuning", "9", "level or worse", "drop", "Default LightGBM stays best", "D206"],
  ["Hierarchical reconciliation, campaign calendar, FFORMA weights, Trends, neural nets, augmentation", "9", "level or worse", "drop", "Main session's M5 directions", "D172"],
  ["Conformal quantiles for the TabPFN member", "9", "0.4027", "keep", "Becomes the fifth member", "D206"],
  ["Arithmetic mean of five members (p2c32a)", "9", "0.3977", "keep", "Geometric stacks forecast low; buying loss 10.74%", "D231"],
  ["Buying quantity: quantile members at P65", "9", "buying 10.38%", "keep", "The loss curve bottoms at P65–P70; a column for buyers, not the point forecast", "D236"],
  ["Middle-out through type × look cells; factor model", "10", "worse", "drop", "Cell totals cannot be forecast better than the sum of items", "D240"],
  ["Segment re-allocation + breakout model, totals kept", "10", "0.3902", "keep", "Accuracy gain; buying level", "D248"],
  ["Tree member: + 15 event windows + launch curves, − static descriptors", "11", "0.3961", "keep", "Alone 0.4326 → 0.4282; with corrections 0.3871", "D253"],
  ["Age block, type trends, peak features, low-volume and Black Friday models", "11", "level", "drop", "Level alone or in the system", "D253"],
  ["Launch plan at look level", "11", "0.3933 vs 0.3977", "drop", "Helps, but needs Edikted's launch calendar for Jul–Oct 2026", "D264"],
  ["TabPFN v2 rebuilt on the new features", "12", "0.3969 vs 0.3961", "drop", "Better alone, but loses the upward level the stack needed", "D270"],
  ["Sales-curve NN rebuilt on the new features", "12", "0.3939", "keep", "With re-fitted corrections 0.3851 recent, 0.4399 all 13: the submitted system", "D273"]];''')

# ---------------- decision log: unique keys, chips anywhere ----------------
rep('''function openLog(id) { const f = $("#log-filter button[aria-pressed='true']").dataset.v; if (f !== "all") { $("#log-filter button[data-v='all']").click(); } const d = $("#log-" + id); if (!d) return; d.open = true; d.scrollIntoView({ block: "center" }); d.classList.add("flash"); setTimeout(() => d.classList.remove("flash"), 1600); }''',
    '''function openLog(id) { const f = $("#log-filter button[aria-pressed='true']").dataset.v; if (f !== "all") { $("#log-filter button[data-v='all']").click(); } const d = document.getElementById("log-" + id); if (!d) return; d.open = true; d.scrollIntoView({ block: "center" }); d.classList.add("flash"); setTimeout(() => d.classList.remove("flash"), 1600); }
document.querySelectorAll(".chip.dl").forEach(c => { const e = D.log.find(l => l.id === c.dataset.d); if (e) tipOn(c, `<div class="tt">${e.id} · ${esc(e.title)}</div>${esc(e.body.slice(0, 260))}${e.body.length > 260 ? "…" : ""}`); c.addEventListener("click", () => openLog(c.dataset.d)); });
const stb = $("#sys-tab tbody"); if (stb) stb.innerHTML = [["WAPE ÷ R3, nine recent origins", SY.recent.toFixed(4)], ["WAPE ÷ R3, all 13 origins", SY.all13.toFixed(4)], ["WAPE ÷ R3, the 100 targets (recent)", SY.targets.toFixed(4)], ["WAPE ÷ R3, windows with Black Friday", SY.bf.toFixed(4)], ["R3 + rules, for reference (recent)", SY.rules_recent.toFixed(4)], ["Bias, recent · all 13", spct(SY.bias_recent) + " · " + spct(SY.bias_all)], ["Buying loss: point forecast · P65 ladder · R3", pct(SY.loss_final, 1) + " · " + pct(SY.loss_ladder_geo, 1) + " · " + pct(SY.loss_r3, 1)], ["Item-origins scored, recent · all", fmt(SY.n_recent) + " · " + fmt(SY.n_all)]].map(([k, v]) => `<tr><td>${k}</td><td class="r">${v}</td></tr>`).join("");''')

# ---------------- evaluation script hooks ----------------
rep('''  $("#ev-notes").innerHTML = [`<b>Concentration:</b> ${pct(EV.concentration.half)} of item-origins carry half of all absolute error, ${pct(EV.concentration.eighty)} carry 80%.`,''',
    '''  if ($("#ev-fva-note")) $("#ev-fva-note").textContent = EV.fva_note || ""; if ($("#ev-ladder")) $("#ev-ladder").textContent = pct(SY.loss_ladder_geo, 1) + " against " + pct(SY.loss_final, 1) + " for the point forecast";
  $("#ev-notes").innerHTML = EV.notes ? EV.notes.map(x => `<li>${x}</li>`).join("") : [`<b>Concentration:</b> ${pct(EV.concentration.half)} of item-origins carry half of all absolute error, ${pct(EV.concentration.eighty)} carry 80%.`,''')
rep('''  bars($("#ev-wf"), wf.map((w, i) => ({ k: w.step.replace("model: ", "· "), v: w.total, c: i === 0 || i === wf.length - 1 ? "var(--s1)" : "var(--s2)",
    tip: `<b>${esc(w.step)}</b><br>${fmt(w.total)} units${i ? ` (${w.total >= wf[i - 1].total ? "+" : ""}${fmt(w.total - wf[i - 1].total)}, ×${(w.total / wf[i - 1].total).toFixed(3)})` : ""}` })), { max: mx });''',
    '''  bars($("#ev-wf"), wf.map((w, i) => ({ k: w.step.replace("member: ", "· "), v: w.total, c: /^member/.test(w.step) ? "var(--s2)" : /forecast$/.test(w.step) ? "var(--accent)" : "var(--s1)",
    tip: `<b>${esc(w.step)}</b><br>${fmt(w.total)} units${i ? `, ×${(w.total / wf[0].total).toFixed(3)} of R3` : ""}` })), { max: mx });''')

# ---------------- files & how to run ----------------
rep('''  "# 8  numbers for this page", ".venv/Scripts/python eda/19_project_map_data.py"].join("\\n");''',
    '''  "# 8  later members and features (each script's docstring gives its inputs)", ".venv/Scripts/python eda/162_weather_event_feats.py      # temperature + events", ".venv/Scripts/python eda/180_cooc_prep.py; eda/181_cooc_dyn.py (GPU); eda/182_cooc_feats.py   # dynamic co-purchase neighbours", ".venv/Scripts/python eda/187_explain_table.py; eda/180_newfeats.py                         # model table, event windows, launch curves", ".venv/Scripts/python eda/180q_models.py ec_ns w03ec                                        # tree member (+ volume-weighted, for buying)", ".venv/Scripts/python eda/197b_nn_ec.py seq=1 seeds=3 tag=_S13ec                             # NN member (GPU)", ".venv/Scripts/python eda/196b_tabpfn.py v2 est=32 feat=dyn; eda/199c_conformal.py           # TabPFN members (GPU)", "",
  "# 9  stack, corrections, buying simulation", ".venv/Scripts/python eda/180d_newbase.py; eda/179x2_bias.py; eda/179b_breakout.py; eda/180g_geo.py", ".venv/Scripts/python eda/175n_sim_nn.py; eda/175o_sim_ladder_nn.py", "",
  "# 10 final 2026 chain, workbook, explanation layer", ".venv/Scripts/python eda/181a_base2026.py; eda/181g_final_nx2_2026.py", ".venv/Scripts/python eda/198b_explain_final.py", "",
  "# 11 scoreboard and this page", ".venv/Scripts/python eda/159_scoreboard.py", ".venv/Scripts/python eda/127_build_atlas.py   # runs eda/19 and eda/124, then fills eda/atlas_template.html"].join("\\n");''')
rep('''<li><b>Workbook:</b> the original <code>Edikted_Forecasting_Exercise.xlsx</code> is untouched. The deliverable is <code>…_filled.xlsx</code>.''',
    '''<li><b>Workbook:</b> the original <code>Edikted_Forecasting_Exercise.xlsx</code> is untouched. The deliverable is <code>…_filled.xlsx</code>, a copy of <code>…_candidate_nx2.xlsx</code> since D274; the first submission is kept as <code>…_filled_original_2026-11-12.xlsx</code>. Replacing the submission is the user's decision.''')
rep('''<li><b>Long write-up:</b> <code>EDA_REPORT.md</code> (Parts 1–6) holds the narrative; the printed outputs sit beside each script in <code>eda/*_output.txt</code>.</li>''',
    '''<li><b>Long write-ups:</b> <code>EDA_REPORT.md</code> (Parts 1–6), <code>ERROR_ANALYSIS_REPORT.md</code>, <code>OVERNIGHT_M5_REPORT.md</code>, <code>DECOMPOSITION_ERRORS_REPORT.md</code>, <code>FIXABLE_ERRORS_REPORT.md</code> and <code>SUBMISSION_EXPLAINABILITY.md</code>; the printed outputs sit beside each script in <code>eda/*_output.txt</code>.</li>
        <li><b>Two sessions:</b> <code>SYNC.md</code> holds the rules for working in parallel: decision-id ranges, script-number ranges and who owns which artifact.</li>
        <li><b>Related pages:</b> <span id="links"></span></li>''')
between('    <h3>Open questions worth your first day</h3>\n', '  </div>\n</div></section>\n</main>', '''    <ul class="takeaways small">
      <li><b>Summer 2026 actuals.</b> The backtest has one window like the deliverable (the June 2025 origin). Score the frozen submission on July–October 2026 as the months arrive: it is the only untouched test.</li>
      <li><b>World Cup items.</b> The submitted chain has no tournament-end rule; the three targets are forecast 55% above the 12 Nov submission, which capped August–October after the 19 July final (D033).</li>
      <li><b>Breakouts.</b> Young launches that break out into the peak are the largest error and are weakly visible at the origin; half show within a month. A monthly re-forecast of young items is the operational fix (D248).</li>
      <li><b>Missing inputs.</b> Per-item markdown and promotion plans, the launch calendar (worth about 1% at look level, D264), inventory and receipts, and the full historical catalogue: the 1,500 items are survivors (D254).</li>
      <li><b>Buying economics.</b> The P65 quantity comes from the simulation's assumed margins; with per-item cost, price and salvage the order quantile can be set per item.</li>
    </ul>
''')
rep("const DDOC = {", '''{ const L = D.links || {}, a = []; if (L.scoreboard) a.push(`<a href="${L.scoreboard}">Forecast scoreboard</a> (every scored system against R3)`); if (L.error_analysis) a.push(`<a href="${L.error_analysis}">Deep error analysis</a>`); const le = $("#links"); if (le) le.innerHTML = a.join(" · ") || "–"; }
const DDOC = {''')
# ---------------- modelling charts: the submitted system per origin; relabel the round-3 model ----------------
rep('''const PO = D.per_origin, origins = Object.keys(PO["Final model (two median-loss LightGBMs)"]), PC = { "Final model (two median-loss LightGBMs)": "var(--s1)", "Structured baseline": "var(--s3)", "Tweedie on units": "var(--s2)", "Flat 91-day run-rate": "var(--fg3)" };''',
    '''const PO = D.per_origin, origins = Object.keys(PO["Monthly LightGBM pair (rounds 2-3)"]), PC = { "Submitted system (five-member stack + corrections)": "var(--accent)", "Monthly LightGBM pair (rounds 2-3)": "var(--s1)", "Structured baseline": "var(--s3)", "Tweedie on units": "var(--s2)", "Flat 91-day run-rate": "var(--fg3)" };''')
rep('''dots: true, w: k.startsWith("Final") ? 2.4 : 1.6 })),''', '''dots: true, w: k.startsWith("Submitted") ? 2.8 : 1.6 })),''')
rep("<h3>What drives the final all-feature model</h3>", "<h3>What drives the monthly all-feature model (round 3)</h3>")
rep("Hover an origin for WAPE, bias and hit rates. The structured baseline needs a year of history, so it starts in July 2025.", "Hover an origin for WAPE, bias and hit rates. The submitted system is the thick line; the others are the round 2-3 references. The structured baseline needs a year of history, so it starts in July 2025.")
open(p, "w", encoding="utf-8", newline=chr(10)).write(s); print("patched", N[0], "blocks;", len(s), "chars")
