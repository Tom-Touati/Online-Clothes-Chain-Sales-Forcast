"""D337 (fork 2, sole page editor): the atlas follows Main's D299 / D350: the submission is candidate_nx4 (candidate_nx3's units, the
D318 confidence rule, the D317 upper-widened ranges). eda/19 reads the nx4 cascade; the template's workbook texts name nx4. Idempotent."""
p = "eda/19_project_map_data.py"; s = open(p, encoding="utf-8").read()
if "final_2026_nx4_cascade" not in s:
    for a, b in [('C = pd.read_csv("data/final_2026_nx3_cascade.csv"', 'C = pd.read_csv("data/final_2026_nx4_cascade.csv"'), ('H = pd.read_csv("data/final_2026_nx3_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx4_cascade_hierarchy.csv")'),
                 ("# 11 targets table: the SUBMITTED workbook (D303: a copy of candidate_nx3 = candidate_nx2 + the World Cup hand rule, D291)", "# 11 targets table: the SUBMITTED workbook (D350: a copy of candidate_nx4 = candidate_nx3 + the D318 confidence rule and the D317 ranges, D299)")]:
        assert a in s, a[:60]; s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
if "candidate_nx4" in s: raise SystemExit("template already patched")
R = [("On 17 November the World Cup hand rule was restored on top (candidate_nx3). <button class=\"chip dl\" type=\"button\" data-d=\"D291\">D291</button> <button class=\"chip dl\" type=\"button\" data-d=\"D303\">D303</button>",
      "On 17 November the World Cup hand rule was restored on top (candidate_nx3), then the confidence column was set by a validated rule (levels 1 / 2 / 3 land within ±20% 58% / 44% / 28% of the time in the backtest) and the ranges widened upward to 80% coverage, with units unchanged (candidate_nx4). <button class=\"chip dl\" type=\"button\" data-d=\"D291\">D291</button> <button class=\"chip dl\" type=\"button\" data-d=\"D303\">D303</button> <button class=\"chip dl\" type=\"button\" data-d=\"D299\">D299</button> <button class=\"chip dl\" type=\"button\" data-d=\"D350\">D350</button>"),
     ("a copy of <code>…_candidate_nx3.xlsx</code> since D303 (candidate_nx2 with the World Cup hand rule)", "a copy of <code>…_candidate_nx4.xlsx</code> since D350 (candidate_nx2 with the World Cup hand rule, the D318 confidence rule and the D317 ranges)"),
     ("the submitted workbook (candidate_nx3, D303)", "the submitted workbook (candidate_nx4, D350)"),
     ("181a_base2026 · 181g_final_nx2_2026.py · 181h_final_nx3_2026.py", "181a_base2026 · 181g_final_nx2_2026.py · 181h_final_nx3_2026.py · 181i_final_nx4_2026.py"),
     ('outs: ["data/final_2026_nx3_cascade.csv", "Edikted_Forecasting_Exercise_candidate_nx3.xlsx"]', 'outs: ["data/final_2026_nx4_cascade.csv", "Edikted_Forecasting_Exercise_candidate_nx4.xlsx"]'),
     ("A copy of candidate_nx3 since D303: candidate_nx2 with the World Cup hand rule restored.", "A copy of candidate_nx4 since D350: candidate_nx2 with the World Cup hand rule restored (D303), confidence by the validated rule (D318) and ranges widened upward (D317)."),
     ("Switched at D271 (candidate_nx), D274 (candidate_nx2) and D303 (candidate_nx3)", "Switched at D271 (candidate_nx), D274 (candidate_nx2), D303 (candidate_nx3) and D350 (candidate_nx4)"),
     ("eda/181g_final_nx2_2026.py; eda/181h_final_nx3_2026.py\"", "eda/181g_final_nx2_2026.py; eda/181h_final_nx3_2026.py; eda/181i_final_nx4_2026.py\"")]
for a, b in R: assert s.count(a) == 1, a[:70]; s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s); print("patched")
