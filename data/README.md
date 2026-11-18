# Data

The tables the pipeline reads or writes, at the paths the scripts use. The main ones:

| Table | What it holds | Written by |
|---|---|---|
| `monthly_item`, `items`, `daily_item_chan`, `features` | Units per item and month / day and channel; item master; model features per month-end | stage 1 |
| `fk_explain_table`, `nf2_feats` … `nf4_feats`, `fk_dyn_feats`, `fk_fash_feats`, `fk_nnc_feats`, `rc_temporal_feats` | Feature tables | stage 2 |
| `q_backtest_ls_huber*`, `nn_backtest_preds_*`, `q_backtest_fktab_*`, `q_backtest_fkconf_*`, `fk_2026_*`, `rc_recsys_preds` | Member forecasts: backtest (13 month-ends) and June 2026 | stage 3 |
| `dec_base_nx5r26`, `bias_seg_preds_nx5r26`, `breakout_preds_nx5r26_x`, `geo_nx5r_x_*`, `geo_nx5r_fash_*` | Base table and corrections; `geo_nx5r_fash_*` is the system in the submission | stage 4 |
| `ranges_nx5r_fash_*`, `fk_confidence_2026.csv`, `fk_err_master` | Ranges and confidence | stage 5 |
| `final_2026_nx9_cascade*.csv`, `final_2026_nx9_total_ranges.csv` | Every step of the submission per item, rolled up; ranges for totals | stage 6 |

`srv4/` holds two tree files copied back from a server (used by the last-year test); `llm/` and `weather_raw/` raw inputs of stage 2.
