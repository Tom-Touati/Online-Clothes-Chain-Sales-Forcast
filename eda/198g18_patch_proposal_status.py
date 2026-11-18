"""D336 (fork 2, now the sole editor of the pages; user: "make sure that ... the error analysis, atlas and scoreboard are up to date"):
one-off patch of the error study for the work done since version 18:
  - a 'Status' line under each of the 14 proposals: owner (SYNC.md, 'the 14 proposals', D314) and what is done, with D293-D296 (Main)
    and D315-D318 (fork) results;
  - proposal 8 rewritten as done (the World Cup cap is in the submitted workbook since D303; D295: no other event rule needed);
  - the 'What to expect' note gets the frozen expectation of D316 (80% band for a 100-item score).
Tokens filled from data/fk_err_analysis.json keys lyremoved, expect26, intervals, confrule (read only). Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if "{{ps_1}}" in s: raise SystemExit("already patched")
import re
heads = [(m.start(), int(m.group(1))) for m in re.finditer(r'<div class="prop"><h3>(\d+)\. ', s)]; assert [n for _, n in heads] == list(range(1, 15)), [n for _, n in heads]
for pos, n in sorted(heads, reverse=True):
    j = s.index("</dl></div>", pos); s = s[:j] + f'\n      <dt>Status</dt><dd class="status">{{{{ps_{n}}}}}</dd>' + s[j:]
a = "<dt>Evidence</dt><dd>The World Cup items: {{wc_now}} units in the submitted workbook against {{wc_rule}} under the earlier rule.</dd>"; assert a in s
s = s.replace(a, "<dt>Evidence</dt><dd>The World Cup items: the models' {{wc_now}} units against {{wc_rule}} under the hand rule, now in the submitted workbook.</dd>")
a = "<dt>Worth</dt><dd>About {{wc_diff}} units on three targets, {{wc_pct}} of the submitted total.</dd>"; assert a in s
s = s.replace(a, "<dt>Worth</dt><dd>About {{wc_diff}} units on three targets, {{wc_pct}} of the total; taken.</dd>")
a = "In that season the five models score 0.77–0.90 alone and the stack 0.74.</div>"; assert a in s
s = s.replace(a, "In that season the five models score 0.77–0.90 alone and the stack 0.74. <b>The expectation is frozen</b> on the submitted workbook (md5 {{ex_md5}}, {{ex_total}} units): central absolute error {{ex_abs}}, {{ex_r3}} × R3, bias {{ex_bias}}, {{ex_hit}} within ±20%. For a 100-item score the 80% band is {{ex_b_abs_lo}}–{{ex_b_abs_hi}} absolute error and {{ex_b_r3_lo}}–{{ex_b_r3_hi}} × R3 (FORECAST_EXPECTATION.md; eda/score_2026_actuals.py scores the workbook once the actuals exist).</div>")
s = s.replace("#t-rules td:first-child,", "dd.status{color:var(--fg2);font-size:13px} #t-rules td:first-child,", 1)
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    ex_md5=EX["md5"][:8], ex_total=it(EX["total"]), ex_abs=pc(EX["central"]["abs_wape"], 1), ex_r3=f2(EX["central"]["wape_r3"]), ex_bias=sp(EX["central"]["bias"]), ex_hit=pc(EX["central"]["hit20"]),
    ex_b_abs_lo=pc(EX["band100"]["abs_wape"][0]), ex_b_abs_hi=pc(EX["band100"]["abs_wape"][1]), ex_b_r3_lo=f2(EX["band100"]["wape_r3"][0]), ex_b_r3_hi=f2(EX["band100"]["wape_r3"][1]),
    **PS,'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
pre = '''# proposal status lines (D336): owners from SYNC.md 'the 14 proposals' (D314); results of D293-D296 (Main) and D315-D318 (fork)
EX = J["expect26"]; LY = {(d["member"], d["mode"]): d for d in J["lyremoved"]["rows"]}; IV = J["intervals"]; CR = [d for d in J["confrule"]["rows"] if d["rule"].startswith("level 1 = narrowest")]; CT = J["confrule"]["targets"]["rule"]
_lo = IV["loo_summary"]; _rec = _lo["lower x1.0, upper x1.4"]
PS = {f"ps_{k}": v for k, v in {
  1: "Owner: Fork (data request, plan template). Tested as a known future: own stores 0.3752 and every business decision 0.3706 with corrections (D308–D310). Waiting for Edikted's 2026 plan.",
  2: "Owner: Fork 2 (partner channel modelled apart, inside the per-location recommender); data request by the Fork. In progress.",
  3: "Owner: Main. Done as a no-op layer in the workbook build (D296): forecast capped at stock + inbound once the data exists.",
  4: "Owner: Fork 2 when the data arrives; data request by the Fork.",
  5: "Owner: Main when the data arrives; data request by the Fork.",
  6: "Owner: Fork. Queued: month-1 update model and a script Edikted can run on July 2026 actuals.",
  7: "Owner: Fork. Queued: three conditional forecasts per young target from the known-future tree model.",
  8: "Done. The World Cup cap is in the submitted workbook (D291, D303). Other dated tags among the targets (4th of July, Halloween, Homecoming) showed no bias that needs a rule (D295).",
  9: "Owner: Fork 2 (per-location recommender: store sales per door × doors). Running.",
  10: f"Owner: Fork, then Main. Done for the tree member (D315): without last year's window it loses {LY[('ec_ns', 'no LY')]['alone_rec'] - LY[('ec_ns', 'full')]['alone_rec']:.2f} alone and {LY[('ec_ns', 'no LY')]['sys_rec'] - LY[('ec_ns', 'full')]['sys_rec']:.3f} in the system. The 'sig' candidate is mixed: better on all 13 origins without last year ({LY[('sig', 'no LY')]['sys_all']:.4f} against {LY[('ec_ns', 'no LY')]['sys_all']:.4f}), worse on recent ones ({LY[('sig', 'no LY')]['sys_rec']:.4f} against {LY[('ec_ns', 'no LY')]['sys_rec']:.4f}). Neural net and TabPFN next; shrunk seasonality (D297) prepared.",
  11: f"Owner: Fork computes, Main applies. Computed (D317): keep the lower side, multiply the upper log width by 1.4. Recent coverage {_lo['as stated']['inside']:.1%} → {_rec['inside']:.1%}, interval score level ({_lo['as stated']['interval_score']:.3f} → {_rec['interval_score']:.3f}); 2026 ranges in data/fk_intervals_2026.csv, for the next candidate.",
  12: f"Owner: Fork (a), Main (b, c). Confidence rule (D318): level 1 = narrowest fifth of the range + rising + sold last year. Levels 1 / 2 / 3 land within ±20% {CR[0]['hit_recent']:.1%} / {CR[1]['hit_recent']:.1%} / {CR[2]['hit_recent']:.1%} of the time; the 2026 targets fall {CT['1']} / {CT['2']} / {CT['3']}. Unit-weighted hit rate on the scoreboard (D294).",
  13: "Owner: Main. Done (D294): the scoreboard carries absolute error, the ratio to the seasonal naive, the ratio to R3 by season, the typical-item ratio (GMRAE) and origin-block intervals.",
  14: f"Owner: Fork (a), Main (b). Done: expectation frozen on the submitted workbook (D316, md5 {EX['md5'][:8]}); early-origin gate on the scoreboard (D294).",
}.items()}
'''
t = t.replace(y, pre + y, 1); open(b, "w", encoding="utf-8").write(t); print("patched")
