"""Fork 2 (sole page editor, D336): one-off patch of the error study for the overnight work (user: "/goal hyperparameter optimization on all
models through the night"; artifact-editor note of 18 Nov). Adds a 'Tuning' section after 'Overfit':
  - every model tuned on 2025-02/03..09 and judged on the held-out origins 2025-10..2026-02 inside the five-member system (D325, D326,
    D362, D363, D381, D389 / D389b);
  - the bias of each member, recent against older origins (D384), and the remedies for under-forecasting tried tonight (D382, D383,
    D385-D388, D389c, D395, D396).
Numbers are copied from log.md (static text, not tokens). Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="tuning"' in s: raise SystemExit("already patched")
a = '<a href="#confounds">Confounds</a>'; assert s.count(a) == 1; s = s.replace(a, '<a href="#tuning">Tuning</a>' + a)
SEC = '''<section id="tuning">
  <div class="eyebrow">Tuning</div>
  <h2>Every model was tuned overnight; none moved the system</h2>
  <p>Each model's settings were searched on the origins 2025-02/03 to 2025-09 and then judged on five held-out origins (2025-10 to 2026-02) inside the five-member system, with the submitted correction multiplier. Scores are WAPE ÷ R3; lower is better. A tuned model is adopted only if the system gains beyond the 95% interval and the last-year-removed gate still passes.</p>
  <div class="tscroll"><table id="t-tuning"><thead><tr><th>Model</th><th>What was searched</th><th class="r">Alone, held out</th><th class="r">System, held out</th><th>Verdict</th><th>Log</th></tr></thead><tbody>
    <tr><td>Tree member (LightGBM)</td><td>Leaves, depth, learning rate, regularisation, loss</td><td class="r">no gain</td><td class="r">no gain</td><td>Rejected</td><td>D362</td></tr>
    <tr><td>Corrections (segment and breakout)</td><td>Shrinkage, segment sizes, caps</td><td class="r">–</td><td class="r">no gain</td><td>Rejected</td><td>D363</td></tr>
    <tr><td>Neural net</td><td>Width, depth, dropout, learning rate, epochs</td><td class="r">no gain</td><td class="r">no gain</td><td>Rejected</td><td>D325</td></tr>
    <tr><td>Per-location recommender</td><td>Embedding size, layers, learning rate, the tree twin</td><td class="r">–</td><td class="r">0.3031 → 0.3031</td><td>Rejected</td><td>D381</td></tr>
    <tr><td>Rule agents</td><td>16 knobs (weights of the recent and long rates, ramps, shrinkage, seasonal anchor and power, age fade)</td><td class="r">0.3595 → 0.3532</td><td class="r">0.2922 → 0.2917<br><span class="small">95%: −0.0012 to +0.0003</span></td><td>Level; current rules stand</td><td>D389, D389b</td></tr>
    <tr><td>TabPFN (pf2d32)</td><td>Estimators, softmax temperature, context cap, averaging</td><td class="r">0.3432 → 0.3418</td><td class="r">0.2948 → 0.2947<br><span class="small">95%: −0.0002 to −0.0000</span></td><td>Level; stored member stands</td><td>D326</td></tr>
  </tbody></table></div>
  <p class="small">Two models improved on their own (rules and TabPFN) and passed the held-out test alone, but averaging with the four other members absorbs the change. TabPFN's gain in the system (0.0001) is smaller than the noise from simply rerunning it (0.0004). Its tuned ranges cover more (P25–P75 inside 47.6% → 54.2%) but pinball loss got worse (0.3797 → 0.3842), so the ranges do not improve either. Not tuned: the window pf2c uses to calibrate its ranges.</p>
  <div class="grid2">
    <div class="panel"><h3>Who under-forecasts</h3><p>Bias of each member (forecast ÷ actual − 1), on the recent origins against the older ones (D384):</p>
      <div class="tscroll"><table><thead><tr><th>Member</th><th class="r">Recent</th><th class="r">Older</th></tr></thead><tbody>
        <tr><td>Tree (absolute error)</td><td class="r">−10%</td><td class="r">+20%</td></tr>
        <tr><td>Tree (Huber)</td><td class="r">−9%</td><td class="r">+19%</td></tr>
        <tr><td>Rule agents</td><td class="r">−6%</td><td class="r">−6%</td></tr>
        <tr><td>Neural net</td><td class="r">−10%</td><td class="r">+9%</td></tr>
        <tr><td>TabPFN conformal (pf2c)</td><td class="r">−14%</td><td class="r">+23%</td></tr>
        <tr><td>TabPFN 32 (pf2d32)</td><td class="r">+9%</td><td class="r">+20%</td></tr>
        <tr><td><b>Five-member mean</b></td><td class="r"><b>−6%</b></td><td class="r"><b>+13%</b></td></tr>
      </tbody></table></div>
      <p>The trees are not singled out: every learned member swung from over-forecasting the older origins to under-forecasting the recent ones, as growth slowed. The bias follows the period, not the model type.</p></div>
    <div class="panel"><h3>Remedies tried tonight</h3><ul>
      <li><b>Linear-leaf trees</b> (a linear model inside each leaf, to extrapolate beyond the training range): worse (D382).</li>
      <li><b>Stretching the ensemble</b> away from the mean: the shrinkage is real but small; a stretch hurts (0.399 against 0.385) (D382).</li>
      <li><b>Time since the data start, and the last year weighted ×2</b> in the loss: no gain (D383).</li>
      <li><b>Absolute (log) units as the target</b>, and a linear model's total as an input: no gain.</li>
      <li><b>Business plan missing as NaN</b> (augmented training), or <b>imputed from the past</b>: no gain over the stored members (D385, D386).</li>
      <li><b>The recommender's tree twin at full weight</b>: best backtest, but fails the last-year-removed gate (0.4168 → 0.7656 alone); kept only as the value of Edikted's plan (D388).</li>
      <li><b>TabPFN first, trees and NN on its residual</b>: the first layer alone is the best no-plan model (0.406 / 0.462) but level in the system (D387, D389c).</li>
      <li><b>Copying last year</b>, and a tree on the residual from last year: far worse on the Jul–Oct 2025 analog (2.09 × R3 against 0.66, bias +67%); too few training rows have a last year (D395, D396).</li>
    </ul></div>
  </div>
</section>

'''
a = '<section id="confounds">'; assert s.count(a) == 1; s = s.replace(a, SEC + a)
open(p, "w", encoding="utf-8").write(s)
b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
t = t.replace('    built="17 Nov 2026",', '    built="18 Nov 2026",'); open(b, "w", encoding="utf-8").write(t)
print("patched")
