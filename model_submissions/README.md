# Other workbooks

Every workbook other than the submission, one folder each: two early candidates that add one member to the LightGBM quantile model,
and the earlier versions of the submission.
Totals are July–October 2026 units for the 100 items; confidence counts are items at level 1 / 2 / 3. Decision ids refer to
[`docs/decision_log.md`](../docs/decision_log.md). The submission at the root is nx17 (D436): nx13 with a table in "Model explanation" of every metric for SYS and, as a ratio to it, for R3, SN,
AR and each of SYS's six models alone (deep blue = as good as SYS), under "How you tested it" and again below the boxes.

| Folder | What it is | Total | Confidence | Decision |
|---|---|---|---|---|
| `member_rule_agents` | The LightGBM quantile model, the neural network and the agents' planning rules, averaged (geometric mean), with the segment rules | 615,937 | 5 / 69 / 26 | D105 |
| `member_neural_network` | The LightGBM quantile model re-centred on a 50/50 blend with the neural network, with the segment rules | 627,253 | 6 / 68 / 26 | D085–D086, rebuilt D097 |
| `01_first_submission_2026-11-12` | The first submission: LightGBM quantile model blended with each item's own trend, plus segment rules (before the agents' planning rules) | 580,968 | 2 / 64 / 34 | D039–D063, final D073 |
| `02_p2c32a_stack` | Five-member stack (rules, LightGBM, NN, two TabPFN) | 629,372 | 20 / 65 / 15 | D237 |
| `03_nx_new_system` | New tree features (events, launch curves) and corrections | 623,502 | 20 / 65 / 15 | D262, submitted D271 |
| `04_nx2_new_nn` | The neural network with the same new inputs | 620,608 | 20 / 67 / 13 | D273, submitted D274 |
| `05_nx3_world_cup_rule` | The World Cup hand rule restored | 601,801 | 20 / 65 / 15 | D291, submitted D303 |
| `06_nx4_confidence_rule` | Confidence by the tested rule, wider ranges | 601,801 | 10 / 82 / 8 | D299, submitted D350 |
| `07_nx5_huber_tree_recommender` | Huber-loss tree and the recommender at a quarter weight | 604,838 | 10 / 82 / 8 | D361 |
| `08_nx5_routed_launch_tree` | Launch-weighted tree for items under six months, new ranges | 620,826 | 10 / 82 / 8 | D389i, submitted |
| `09_nx6_hybrid_confidence` | Confidence 3 only where the ranges give less than even odds | 620,826 | 10 / 86 / 4 | D389n, submitted D389t |
| `10_nx7_audited_text` | Corrected text, 18 chosen explanations, decisions sheet | 620,826 | 10 / 86 / 4 | D389v |
| `11_nx7c_campaign_uplift` | The two new site sales added explicitly | 629,779 | 10 / 86 / 4 | D389x, submitted |
| `12_nx8_combined` | nx7c's numbers with nx7's text | 629,779 | 10 / 86 / 4 | D421, submitted |
| `13_nx9_lookalike_term` | nx8 plus the look-alikes' last-year ramp in the correction | 630,254 | 10 / 86 / 4 | D422, submitted |
| `14_nx10_roadmap_text` | nx9's numbers with the repaired text and the improvement roadmap | 630,254 | 10 / 86 / 4 | D424, submitted D428 |
| `15_nx11_metrics_cascade` | The explanation coarse to fine, all the main metrics, the cascade as explainability | 630,254 | 10 / 86 / 4 | D429, submitted |
| `16_nx12_buying_metrics_table` | The buying simulation of the submitted system, the metrics-vs-baselines sheet, NOAA temperature, bold line summaries | 630,254 | 10 / 86 / 4 | D431, submitted |
| `17_nx13_abbreviations` | The abbreviations SYS / R3 / SN / AR in every list of figures | 630,254 | 10 / 86 / 4 | D432, submitted |
| `18_nx14_ratio_table_green` | The ratio table in green, metrics in their own directions | 630,254 | 10 / 86 / 4 | D433, submitted |
| `19_nx15_ratio_table_coolwarm` | The ratio table in cool-warm, every metric lower-is-better | 630,254 | 10 / 86 / 4 | D434, submitted |
| `20_nx16_models_table` | The all-models table (white at 1, red above) below the boxes | 630,254 | 10 / 86 / 4 | D435, submitted |
