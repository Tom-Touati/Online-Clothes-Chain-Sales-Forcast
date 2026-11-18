"""Fork 2 (sole page editor): the error study gets a 'Submission' section for the new nx5 (D389i) and the overnight optimisation after it
(D389j-o), placed after 'Tuning'. Numbers copied from log.md (static text). Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="submission"' in s: raise SystemExit("already patched")
a = '<a href="#confounds">Confounds</a>'; assert s.count(a) == 1; s = s.replace(a, '<a href="#submission">Submission</a>' + a)
SEC = '''<section id="submission">
  <div class="eyebrow">Submission</div>
  <h2>The new nx5 is the submission; tonight's optimisation kept it as it is</h2>
  <p>On 18 November the submitted workbook moved from nx4 to the new nx5: the six-member system with the Huber tree and the recommender at a quarter weight, where items under six months old get a tree trained with new items counted five times, the corrections refitted on that system, and ranges recalibrated per age and volume group with last July–October and the latest March–June weighted four times. Scores are WAPE ÷ R3 on the time-honest backtest.</p>
  <div class="tscroll"><table id="t-submission"><thead><tr><th>System</th><th class="r">Recent 9</th><th class="r">Held out</th><th class="r">All 13</th><th class="r">Early 4</th><th class="r">2026 total</th></tr></thead><tbody>
    <tr><td>nx4 (previous submission)</td><td class="r">0.3851</td><td class="r">0.2948</td><td class="r">0.4399</td><td class="r">0.6623</td><td class="r">601,801</td></tr>
    <tr><td>nx5 (D361, Huber tree + recommender)</td><td class="r">0.3811</td><td class="r">0.2924</td><td class="r">0.4358</td><td class="r">0.6581</td><td class="r">604,838</td></tr>
    <tr><td><b>new nx5 (submitted)</b></td><td class="r"><b>0.3806</b></td><td class="r"><b>0.2926</b></td><td class="r"><b>0.4357</b></td><td class="r"><b>0.6593</b></td><td class="r"><b>620,826</b></td></tr>
  </tbody></table></div>
  <p class="small">Almost all of the 2026 difference (+15k) is the 24 targets under six months old: the launch-weighted tree lifts them 10.6%. In the backtest the same routing moved new items' totals by only 0.3%, while the system under-forecast them by 7–12%; the 2026 lift goes the historical way but is larger than anything the backtest saw. The ranges: left one origin out, 82.5% of actuals inside P10–P90 and 52.4% inside P25–P75 on the held-out origins; on last July–October only 70%, a season that ran below the forecast, while March–June 2026 ran above it.</p>
  <div class="grid2">
    <div class="panel"><h3>What the night tried after it</h3><ul>
      <li><b>New-item and old-item trees tuned separately</b> (333 trials): both beat the current trees on the tuning origins and lose on the five held-out ones (new 0.3463 → 0.3492, old 0.2653 → 0.2678), as every earlier tuning round (D389l).</li>
      <li><b>A fixed grid of simple versions:</b> weighting the youngest launches (under 3 months) ×8–12 helps the held-out origins (0.3463 → 0.3418) but fails the last-year-removed gate (0.4505 → 0.4651); kept for the July re-test (D389m).</li>
      <li><b>Corrections at full strength:</b> worse held out (0.2926 → 0.2938–0.2992) (D389o).</li>
      <li><b>The monthly split</b> (rules × network monthly shapes), scored for the first time: best of nine on both focus windows; monthly error 0.336 on Jul–Oct 25 against 0.404 for a flat split (D389j).</li>
      <li><b>Per-item weights by correlation</b> at item, cluster and type level, and <b>everything calibrated to the two focus windows</b>: level, or better there and worse everywhere else (D389e, D389f).</li>
    </ul></div>
    <div class="panel"><h3>The one change that held: confidence</h3><p>Judged by the brief's own scale (1 = within ±20%, 2 = within ±50%, 3 = could miss by more than 50%), the submitted rule's level 3 (the widest fifth of ranges) lands within ±50% 63% of the time, so most of its claims fail. Putting level 3 only where the calibrated range gives less than even odds of landing within ±50% raises the share of claims met from 0.75 to 0.81 on the held-out origins, 0.70 to 0.74 on last July–October and 0.76 to 0.84 on March–June 2026 (D389k). In 2026 that moves four model-forecast items from 3 to 2 (levels 10 / 86 / 4). It is built as candidate_nx6 (D389n), for the user to decide.</p></div>
  </div>
</section>

'''
a = '<section id="confounds">'; assert s.count(a) == 1; s = s.replace(a, SEC + a)
open(p, "w", encoding="utf-8").write(s); print("patched")
