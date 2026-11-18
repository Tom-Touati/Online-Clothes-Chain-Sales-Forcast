"""D303 (fork): one-off patch bringing the Forecast Atlas to the submission of 17 Nov: candidate_nx3 = candidate_nx2 with the World Cup
hand rule restored (Main D291; switched by the fork on the user's "Yes, restore", D303). Edits eda/19_project_map_data.py (nx3 cascade
files, the new total, one more cascade step) and eda/atlas_template.html (texts that named candidate_nx2). Idempotent."""
b = "eda/19_project_map_data.py"; t = open(b, encoding="utf-8").read()
if "final_2026_nx3_cascade.csv" in t: raise SystemExit("already patched")


def rp(txt, a, c, n=1):
    assert txt.count(a) == n, (txt.count(a), a[:70]); return txt.replace(a, c)


t = rp(t, "the SUBMITTED workbook (D274: a copy of candidate_nx2, D273) with its cascade (data/final_2026_nx2_cascade.csv)", "the SUBMITTED workbook (D303: a copy of candidate_nx3 = candidate_nx2 + the World Cup hand rule, D291) with its cascade (data/final_2026_nx3_cascade.csv)")
t = rp(t, 'pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0)', 'pd.read_csv("data/final_2026_nx3_cascade.csv", index_col=0)')
t = rp(t, 'pd.read_csv("data/final_2026_nx2_cascade_hierarchy.csv")', 'pd.read_csv("data/final_2026_nx3_cascade_hierarchy.csv")')
t = rp(t, '== 620608, sum(t["fin"] for t in J["targets"])', '== 601801, sum(t["fin"] for t in J["targets"])')
t = rp(t, '("x segment and breakout corrections = forecast", "forecast"), ("buying quantity (quantile members at P65)", "buying")', '("x segment and breakout corrections = forecast", "forecast"), ("with the two hand rules (out of stock, World Cup) = submitted", "final"), ("buying quantity (quantile members at P65)", "buying")')
open(b, "w", encoding="utf-8").write(t)

p = "eda/atlas_template.html"; s = open(p, encoding="utf-8").read()
s = rp(s, "The submission was switched to candidate_nx and then to candidate_nx2. ", 'The submission was switched to candidate_nx and then to candidate_nx2. On 17 November the World Cup hand rule was restored on top (candidate_nx3). <button class="chip dl" type="button" data-d="D291">D291</button> <button class="chip dl" type="button" data-d="D303">D303</button> ')
s = rp(s, "<b>One override:</b> <code>S16079_DARK-GRAY</code>, out of stock since April, keeps its in-stock forecast of 130 units; no model member covers it (D034).</li>",
       "<b>Two hand rules:</b> <code>S16079_DARK-GRAY</code>, out of stock since April, keeps its in-stock forecast of 130 units; no model member covers it (D034). The three World Cup items (<code>S24840</code>, <code>S25039</code>, <code>S24847</code>) carry the hand cap of the first submission, 34,463 units in place of the models' 53,270: sales fall after the final on 19 July and no model has seen a World Cup (D033, restored in D303).</li>")
s = rp(s, "a copy of <code>…_candidate_nx2.xlsx</code> since D274", "a copy of <code>…_candidate_nx3.xlsx</code> since D303 (candidate_nx2 with the World Cup hand rule)")
s = rp(s, "the submitted workbook (candidate_nx2, D274)", "the submitted workbook (candidate_nx3, D303)")
s = rp(s, 's: "181a_base2026 · 181g_final_nx2_2026.py"', 's: "181a_base2026 · 181g_final_nx2_2026.py · 181h_final_nx3_2026.py"')
s = rp(s, "One override: the out-of-stock item keeps its in-stock forecast of 130 units.", "Two hand rules: the out-of-stock item keeps its in-stock forecast of 130 units, and the three World Cup items carry the cap of the first submission.")
s = rp(s, 'outs: ["data/final_2026_nx2_cascade.csv", "candidates/Edikted_Forecasting_Exercise_candidate_nx2.xlsx"]', 'outs: ["data/final_2026_nx3_cascade.csv", "candidates/Edikted_Forecasting_Exercise_candidate_nx3.xlsx"]')
s = rp(s, '"Confidence 1 · 2 · 3 = 20 · 67 · 13", "No campaign, World Cup or segment rules on top"], dec: ["D034", "D121", "D235", "D237", "D262", "D273"]', '"Confidence 1 · 2 · 3 = 20 · 65 · 15", "World Cup hand rule: 34,463 units in place of 53,270; no campaign or segment rules"], dec: ["D033", "D034", "D121", "D235", "D237", "D262", "D273", "D291"]')
s = rp(s, "A copy of candidate_nx2 since D274. Forecast sheet", "A copy of candidate_nx3 since D303: candidate_nx2 with the World Cup hand rule restored. Forecast sheet")
s = rp(s, 'nums: ["620,608 units: Jul 176,453 · Aug 186,211 · Sep 126,019 · Oct 131,925", "Original submission: 580,968", "Switched at D271 (candidate_nx) and D274 (candidate_nx2)"], dec: ["D040", "D271", "D273", "D274"]',
       'nums: ["601,801 units: Jul 171,591 · Aug 175,788 · Sep 123,969 · Oct 130,453", "Original submission: 580,968", "Switched at D271 (candidate_nx), D274 (candidate_nx2) and D303 (candidate_nx3)"], dec: ["D040", "D271", "D273", "D274", "D291", "D303"]')
s = rp(s, '".venv/Scripts/python eda/181a_base2026.py; eda/181g_final_nx2_2026.py"', '".venv/Scripts/python eda/181a_base2026.py; eda/181g_final_nx2_2026.py; eda/181h_final_nx3_2026.py"')
open(p, "w", encoding="utf-8").write(s); print("patched")
