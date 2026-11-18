# The pipeline

Every script that produced the submission, in stage order. All run from the repository root (`python pipeline/<script>`, with the arguments given below), read
and write `data/` (except `196b_tabpfn.py`, which writes to `out/`), and keep the numbered names they had during the project, so the decision log ([`docs/decision_log.md`](../docs/decision_log.md),
entries `D…`) can be searched for them; [`docs/FILE_MAP.tsv`](../docs/FILE_MAP.tsv) maps the log's `eda/…` paths to this repository.
"Laptop" jobs ran on 3 threads at low priority (`lowrun.ps1`, which has the project folder's path and Python written in: edit both lines to use it elsewhere); "GPU" jobs on a cloud T4 / A10; "CPU server" on an 8-core cloud machine.

Checked: the workbook builder rerun in this repository reproduces the submission in every cell of every sheet, and stage 3's
routing step, stage 7's baseline comparison, last-year test and monthly-split test reproduce their logged numbers here.
Stages 1–5 were run in the project folder; rerunning them from scratch needs `Dataset.parquet` at the root, the GPU steps a GPU,
and gives the same numbers up to the models' run-to-run noise (the neural network and the recommender are not bit-reproducible).

## 1 Base tables from the order lines

| Script | What it does |
|---|---|
| `02_build_aggs.py Dataset.parquet` | Reads the order lines (the path is its only argument) with DuckDB: units per item, day and channel (`daily_item_chan`), items (`items`), order types |
| `03_targets.py exercise/Edikted_Forecasting_Exercise.xlsx` | The 100 target items from the exercise workbook, its only argument (`targets.csv`) |
| `04_eda.py` | Units per item and month (`monthly_item`), the targets' overview, suspected stock-outs |
| `06_features.py` | One row per item × month-end × horizon from data up to the month-end only: recent rates, momentum, seasonal ratios (`features`) |

## 2 Feature tables

| Script | What it does |
|---|---|
| `152_orders_extract.py`, `153_basket_emb.py` | Items bought together, by month (`basket_pairs_month`) |
| `160_weather_fetch.py`, `160b_weather_noaa.py`, `161_events.py`, `162_weather_event_feats.py` | Temperature normals where each item sells; a calendar of about 30 US events and sales periods |
| `163_clusters.py` | Multi-view item embedding and clusters (baskets, sales co-movement, fashion descriptors, weather) |
| `177t_product_feed.py`, `177u_content_emb_server.py` (GPU) | Photo and text embeddings of each item (`content_emb`) |
| `181_cooc_dyn.py`, `182_cooc_feats.py` | The co-purchase network as of each month-end, and the seasonality of each item's basket neighbours |
| `182b_fashion_feats.py`, `182d_nn_content.py` | Fashion attributes and scores; photo / text look-alikes' outcomes (`fk_nnc_feats`, incl. the look-alikes' last-year ramp) |
| `182p_planlook.py`, `180_newfeats.py`, `180b_promo.py` | Launch curves, events per horizon month, promotion features (`nf2` / `nf3` / `nf4` tables) |
| `187_explain_table.py` | The model feature table (`fk_explain_table` + `fk_explain_meta.json`) |
| `209_tree_temporal.py` | Selling-rhythm features (`rc_temporal_feats`) |

## 3 The six members (backtest at 13 month-ends and the June 2026 forecast)

| Member | Script and run | Output |
|---|---|---|
| Planning rules | Written by AI agents per month-end in [`rule_agents/`](../rule_agents/) (`rules.py`); checked by `76_rules_harness.py` | `rule_agents/<origin>/predictions_verified.csv` |
| LightGBM tree | `187_losses.py huber huber_new5` (execs `180q_models.py` for the features); June 2026: `ORIGS=2026-06 OSUF=_2026`; last-year test: `LYDROP=1 OSUF=_lyd` (CPU server; the plain tree's last-year run is stored as `srv4/q_backtest_ls_huber_noly`) | `q_backtest_ls_huber[_new5][_2026].parquet`, `q_backtest_ls_huber_new5_lyd.parquet`, `srv4/q_backtest_ls_huber_noly.parquet` |
| … launch routing | `218a_route_tree_2026.py --write` (after stage 4's `218_base_nx5r.py`, whose age column it reads; without `--write` it only writes a check file): items under six months use the launch-weighted tree | `q_backtest_ls_huber_routed_2026.parquet` |
| Neural network | `197b_nn_ec.py seq=1 seeds=3 seed0=0 tag=_S13ec` and `seed0=10 tag=_S13ecs10`; June 2026 the same with `origins=2026-06 tag=_finalSec / _finalSec_s10` (GPU; execs `57_embed_nn.py`). The backtest file is the mean of the two seed sets (an inline step, D272) | `nn_backtest_preds_S13ecx6`, `…_finalSec[_s10]` |
| TabPFN 32 | `196b_tabpfn.py v2 feat=dyn est=32` and the same with `orig=2026-06` (GPU; the script writes `out/q_backtest_tabpfn_v2_dyn_e32.parquet` and `out/q_backtest_tabpfn_v2_o2026-06_dyn_e32.parquet`, so create `out/` first; they were copied into `data/` under the names on the right) | `q_backtest_fktab_tabpfn_v2_dyn_e32`, `fk_2026_pf2d32` |
| TabPFN calibrated | `196b_tabpfn.py v2 feat=dyn` (8 estimators) and the same with `orig=2026-06` (GPU; written to `out/`, the backtest copied to `data/q_backtest_fktab_tabpfn_v2_dyn.parquet`), then `199c_conformal.py pfn2dyn=data/q_backtest_fktab_tabpfn_v2_dyn.parquet` (per-segment conformal quantiles, backtest; the `seg_tails` variant is the member) and `199e_conformal_apply.py backtest=data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet target=out/q_backtest_tabpfn_v2_o2026-06_dyn.parquet out=data/fk_2026_pf2c.parquet` (June 2026) | `q_backtest_fkconf_pfn2dyn_seg_tails`, `fk_2026_pf2c` |
| Recommender | `202_rc_model.py itemfeat=1 loss=logl1 etaclip=3 mino=12 tag=_if_logl1_c3_m12` (DeepFM per item × selling place, summed; GPU / CPU server), then `205_rc_handover.py` (caps each forecast at 3 × the highest member and writes the columns `rc_deepfm`, `rc_gbm`, `rc_avg`), D338 / D339 / D354 / D358 | `rc_recsys_preds` (column `rc_deepfm`) |

## 4 Base table and corrections

| Script | What it does |
|---|---|
| `179d_base.py`, `180d_newbase.py <tag>`, `181a_base2026.py` | The table of member forecasts, actuals and descriptors per item × month-end (`dec_base…`): `179d` writes the five-member `dec_base`, `180d <tag>` swaps in a tree member from `q_backtest_nx_<tag>`. The six-member backtest base `dec_base_ecnsnnhubrq.parquet` (Huber tree, NN, recommender at 0.25; D355 / D358) was made by an inline step and is read as stored. `181a` appends June 2026: `SRC=ecnsnnhubrq NNS=nn_backtest_preds_finalSec,nn_backtest_preds_finalSec_s10 MEMF=data/q_backtest_ls_huber_2026.parquet RC=1 RCW=0.25` → `dec_base_ecnsnnhubrq26` (D361) |
| `218_base_nx5r.py` | Swaps in the routed tree, recomputes the weighted mean (`dec_base_nx5r26`) |
| `179x2_bias.py` | Bias by volume × age × discount group, learnt from earlier month-ends only: `DEC=_nx5r26 NOSCORE=1` (CPU server) |
| `179b_breakout.py` | Breakout / fading residual model: `XSIG=1 DEC=_nx5r26 NOSCORE=1` (CPU server) |
| `218c_geo_nx5r.py` | Combines the two (geometric mean, each month-end's total kept) |
| `228_breakout_fashion.py` | Adds the look-alikes' last-year ramp as a small linear term (within-origin ridge, D401): `OUT2026=1` (CPU server; without it, or with `LYGATE=1`, it only prints the backtest scores) | 

Final corrections: `geo_nx5r_fash_2026.parquet` (June 2026) and `geo_nx5r_fash_x_preds.parquet` (backtest).

## 5 Ranges and confidence

| Script | What it does |
|---|---|
| `219_ranges_nx5r.py` | P10 / P25 / P75 / P90 by age × volume group from the system's own backtest errors: `SYS=nx5r_fash` |
| `198d_error_master.py`, `198ac_confidence_rule.py` | The backtest error table; the rule for confidence 1 (`fk_confidence_2026.csv`) |
| `222_confidence_new.py` | Confidence 3 from the ranges, scored by the scale's own definitions: `SYS=nx5r_fash` |

## 6 The workbook

`181y_final_nx17_2026.py` writes the submission (`Edikted_Forecasting_Exercise_filled.xlsx`) and the cascade tables
(`data/final_2026_nx17_*`): members → weighted mean × correction → monthly split → campaign uplift → hand rules → ranges →
confidence → buying quantity → explanations → sheets. All words are in `181y_nx17_text.py`; it also writes the all-models table in "Model explanation" (twice) from
`data/model_metrics_table.csv`, and the "Metrics vs baselines" sheet from `data/metrics_table_nx12.csv`. (nx13 has
nx9's numbers with the rewritten text; `data/final_2026_nx9_*` … `nx16_*` are the same tables under earlier names.)

## 7 Evaluation quoted in the workbook

| Script | What it measures |
|---|---|
| `226_vs_baselines.py` (`SYS=nx5r_fash`) | Error against R3, a naive seasonal forecast, the agents' rules and the three together, with intervals |
| `224_gate_nx5r.py` | The system with last year's same season removed from training |
| `221_month_split.py` | Nine ways to split the 4-month total by month |
| `226_campaign_test.py` | The campaign uplift's backtest |
| `217_score_new5.py` | The launch-weighted tree, everywhere and routed to new items |
| `231_sim_nx11.py`, `231b_sim_naive.py` | The daily buying simulation of the submitted system, R3, the agents' rules and the seasonal naive (11 three-month windows; `data/sim_nx11_windows.csv`) |
| `232_metrics_table.py` | Every metric of the submitted system against R3, the seasonal naive and the agents' rules (`data/metrics_table_nx12.csv`), scored as the scoreboard |
| `239_model_metrics_table.py` | Every metric for SYS, R3, SN, AR and each of SYS's six models alone, as ratios to SYS with each cell's colour (`data/model_metrics_table.csv`) |
| `234_metric_ratio_table.py` | SYS's ratio to each baseline per metric, every metric turned lower-is-better, with each cell's cool-warm colour (`data/metric_ratio_table.csv`) |

Shared helpers: `score_lib.py`, `86_metrics_suite.py`, `20b_quantiles_lean.py`, `lowrun.ps1`; the buying simulator `47_decision_sim.py`,
`59_sim_v2.py`, `87_sim_systems.py`; the scoreboard's base table `157_stack_new.py`, `159_scoreboard.py` (read by `232`).
