"""Scoreboard data for the forecast-scoreboard dashboard (https://claude.ai/artifact/9ZPDJvYzHq2cUqELfcXdVY).
Every registered system is scored on the same 11,727 item-origins (13 backtest origins, Feb 2025 - Feb 2026) with the standard
R3-relative table (pipeline/score_lib.py: WAPE / R3, relMAE vs R3, win rate vs R3 and vs R3 + rules, (MAE + |bias|) / R3, pinball / R3
with the stored quantile model's spread, bias), on four subsets: the nine recent origins, all 13, the 100 targets (recent) and the
Black Friday origins (Jul-Oct 2025). Buying-simulation loss (production policy, 11 windows Mar 2025 - Jan 2026, all items) comes
from the stored simulation files, as the mean loss and as a ratio to R3's.
To add a system: give it a column in T below and an entry in SYSTEMS, then rerun and re-upload data/scoreboard.json.
Writes data/scoreboard.json ({rows, meta}) and data/scoreboard_sim.json (loss per simulation window)."""
import sys, json, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()):                      # T with r_q_nn, r_qlean_nn, best (D141), r_qnew_nn, best_new (D153)
    exec(open("pipeline/157_stack_new.py", encoding="utf-8").read().split("BF = T.origin.isin")[0].replace("sys.stdout.reconfigure(encoding=\"utf-8\"); ", ""))
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
rd = pd.read_parquet("data/q_backtest_rd_log.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["r_share_rd"] = G(T.rules.values, G(T.q_lean.values, rd), T.nn.values)
wxq = lambda v: pd.read_parquet(f"data/q_backtest_wx_{v}.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["q_wx"] = wxq("cpllm_we"); T["r_qwx_nn"] = G(T.rules.values, T.q_wx.values, T.nn.values); T["best_wx"] = T.r_qwx_nn * T.adj
# D198 / D203 (fork): D159 quantile model + neighbour features from the dynamic co-occurrence network
T["q_dyn"] = pd.read_parquet("data/q_backtest_fkdyn_dyn.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).q50.values
T["r_qdyn_nn"] = G(T.rules.values, T.q_dyn.values, T.nn.values); T["best_dyn"] = T.r_qdyn_nn * T.adj
# D204 (fork): TabPFN members, equal geometric mean with rules, the D198 quantile model and the NN (D169 simulation)
tabq = lambda v: pd.read_parquet(f"data/q_backtest_fktab_tabpfn_{v}.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).q50.values
T["q_tab2"] = tabq("v2"); T["q_tab35f"] = tabq("v35f")
T["tab_p2"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_tab2.values); T["tab_p35f"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_tab35f.values)
T["tab_p2_35f"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_tab2.values, T.q_tab35f.values)
# D206 (fork): TabPFN v2 with co-occurrence features (32 estimators) and the pruned LightGBM (D187 simulation)
fkq = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).q50.values
T["q_t2d"] = fkq("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet"); T["q_sel"] = fkq("data/q_backtest_fkgb_sel.parquet")
T["fk_p2d32"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_t2d.values); T["fk_sel_d32"] = G(T.rules.values, T.q_sel.values, T.nn.values, T.q_t2d.values)
# D205 / D206 (fork), simulated in D188
T["q_t2dyn"] = fkq("data/q_backtest_fktab_tabpfn_v2_dyn.parquet"); T["q_conf"] = fkq("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"); T["q_t35d"] = fkq("data/q_backtest_fktab_tabpfn_v35f_dyn.parquet")
T["fk_p2d"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_t2dyn.values); T["fk_p2c"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_conf.values)
T["fk_p2c_35fd"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.q_conf.values, T.q_t35d.values); T["fk_p2_selrep"] = G(T.rules.values, T.q_sel.values, T.nn.values, T.q_tab2.values)
AM = lambda *xs: np.nanmean(np.column_stack([np.clip(x, 0, None) for x in xs]), axis=1)   # arithmetic-mean stacks (D206, simulated in D190)
T["fk_sel_d32a"] = AM(T.rules.values, T.q_sel.values, T.nn.values, T.q_t2d.values); T["fk_p2d32a"] = AM(T.rules.values, T.q_dyn.values, T.nn.values, T.q_t2d.values)
T["q_w03"] = fkq("data/q_backtest_fkgb_wvol03.parquet")   # D231 (fork stacks, simulated by Main)
T["fk_p2c32a"] = AM(T.rules.values, T.q_dyn.values, T.nn.values, T.q_conf.values, T.q_t2d.values); T["fk_qsc32a"] = AM(T.rules.values, T.q_dyn.values, T.q_sel.values, T.nn.values, T.q_conf.values, T.q_t2d.values)
T["fk_qs_d32a"] = AM(T.rules.values, T.q_dyn.values, T.q_sel.values, T.nn.values, T.q_t2d.values); T["fk_w03d32a"] = AM(T.rules.values, T.q_w03.values, T.nn.values, T.q_t2d.values)
T["fk_w03c32a"] = AM(T.rules.values, T.q_w03.values, T.nn.values, T.q_conf.values, T.q_t2d.values); T["fk_w03_2a"] = AM(T.rules.values, T.q_w03.values, T.nn.values, T.q_tab2.values, T.q_t2d.values)   # D232
fk60 = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).q60.values   # D233 newsvendor ladder
w60, c60, d60 = fk60("data/q_backtest_fkgb_wvol03.parquet"), fk60("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"), fk60("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")
T["fk_w03c32a_ut"] = AM(T.rules.values, w60, T.nn.values, T.q_conf.values, T.q_t2d.values); T["fk_w03c32a_uc"] = AM(T.rules.values, T.q_w03.values, T.nn.values, c60, T.q_t2d.values)
T["fk_w03c32a_u"] = AM(T.rules.values, w60, T.nn.values, c60, d60)
T["r_qnosea_nn"] = G(T.rules.values, wxq("nosea_we"), T.nn.values)
fkx = lambda f, q: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))[q].values   # D236 ladder
for _q in ("q65", "q70"):
    T[f"fk_u{_q[1:]}"] = AM(T.rules.values, fkx("data/q_backtest_fkgb_wvol03_x.parquet", _q), T.nn.values, fkx("data/q_backtest_fkconf_pfn2dyn_seg_tails_x.parquet", _q), fkx("data/q_backtest_fktab_tabpfn_v2_dyn_e32_x.parquet", _q))
_bs = pd.read_parquet("data/bias_seg_preds.parquet").merge(pd.read_parquet("data/breakout_preds.parquet")[["origin", "style_color", "res_all_0.5n"]], on=["origin", "style_color"])   # D245 / D248
_bs["geo"] = np.sqrt(_bs["vol_age_disc_0.5n"].clip(lower=0) * _bs["res_all_0.5n"].clip(lower=0)); _bs["geo_n"] = _bs.geo * _bs.groupby("origin").F.transform("sum") / _bs.groupby("origin").geo.transform("sum")
_bs = _bs.set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
T["fk_p2c32a_seg"] = T.fk_p2c32a.values * (_bs["vol_age_disc_0.5n"] / _bs.F.clip(lower=1e-9)).fillna(1).values
T["fk_p2c32a_geo"] = T.fk_p2c32a.values * (_bs.geo_n / _bs.F.clip(lower=1e-9)).fillna(1).values
_nx = lambda f, c="q50": pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))[c].values   # D253-D260
T["q_ecns"] = _nx("data/q_backtest_nx_ec_ns.parquet"); T["nx_ecns"] = AM(T.rules.values, T.q_ecns.values, T.nn.values, T.q_conf.values, T.q_t2d.values)
for _t in ("ec_ns", "ec_ns_x"):
    _g = pd.read_parquet(f"data/geo_{_t}_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
    T[f"nx_{_t.replace('_', '')}_geo"] = (T.nx_ecns.values * (_g.geo / _g.F.clip(lower=1e-9)).fillna(1).values)
T["nx_u65ec"] = AM(T.rules.values, _nx("data/q_backtest_nx_w03ec.parquet", "q65"), T.nn.values, fkx("data/q_backtest_fkconf_pfn2dyn_seg_tails_x.parquet", "q65"), fkx("data/q_backtest_fktab_tabpfn_v2_dyn_e32_x.parquet", "q65"))
T["nx_w03ec"] = AM(T.rules.values, _nx("data/q_backtest_nx_w03ec.parquet"), T.nn.values, T.q_conf.values, T.q_t2d.values)   # D264
_g = pd.read_parquet("data/geo_w03ec_x_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
T["nx_w03ecx_geo"] = T.nx_w03ec.values * (_g.geo / _g.F.clip(lower=1e-9)).fillna(1).values
T["nx_w03ecpl"] = AM(T.rules.values, _nx("data/q_backtest_nx_w03ec_pl.parquet"), T.nn.values, T.q_conf.values, T.q_t2d.values)
_ni = pd.read_parquet("data/newitems_tot.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).q4.values   # D263
T["nx_newitems"] = np.where(np.isfinite(_ni), (T.nx_ecnsx_geo.values + np.nan_to_num(_ni)) / 2, T.nx_ecnsx_geo.values)
_gx = pd.read_parquet("data/geo_ec_ns_x_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
T["nx_u65ec_geo"] = T.nx_u65ec.values * (_gx.geo / _gx.F.clip(lower=1e-9)).clip(0.2, 5).fillna(1).values   # D265
_K = pd.MultiIndex.from_arrays([T.origin, T.style_color])   # D272 / D273: the fork's NN on the D252 event + launch-curve inputs
T["nn_ec"] = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet").groupby(["origin", "style_color"]).q50.sum().reindex(_K).values
T["nx_ecnsnn"] = AM(T.rules.values, T.q_ecns.values, np.where(np.isfinite(T.nn_ec.values), T.nn_ec.values, T.nn.values), T.q_conf.values, T.q_t2d.values)
_g = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(_K)
T["nx_ecnsnn_geo"] = T.nx_ecnsnn.values * (_g.geo / _g.F.clip(lower=1e-9)).fillna(1).values
T["q_ecsig"] = _nx("data/q_backtest_nx_ecsig.parquet")   # D293: fork D307 tree member (ec_ns + 11 origin-known signals)
T["nx_ecsig"] = AM(T.rules.values, T.q_ecsig.values, np.where(np.isfinite(T.nn_ec.values), T.nn_ec.values, T.nn.values), T.q_conf.values, T.q_t2d.values)
_g = pd.read_parquet("data/geo_ecnsnnsig_x_preds.parquet").set_index(["origin", "style_color"]).reindex(_K)
T["nx_ecsig_geo"] = T.nx_ecsig.values * (_g.geo / _g.F.clip(lower=1e-9)).fillna(1).values
_geoq = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(_K).geo.values   # D353-D355: corrected systems, geo column as stored
T["nx_hub_geo"] = _geoq("data/geo_ecnsnnhub_x_preds.parquet")       # D353: Huber (delta 0.5) tree member, 5 members + corrections
T["nx_rcd_geo"] = _geoq("data/geo_ecnsnnrcd2_x_preds.parquet")      # D354: + rc_deepfm sixth member + corrections
T["nx_hubrc_geo"] = _geoq("data/geo_ecnsnnhubrc_x_preds.parquet")   # D355: Huber tree member + rc_deepfm + corrections
T["nx_ecnsnnhubrq_geo"] = _geoq("data/geo_ecnsnnhubrq_x_preds.parquet")   # D358: Huber tree member + rc_deepfm at weight 0.25 + corrections
# D326 / D389b (overnight tuning): the submitted system with one member swapped for its tuned version, submitted correction multiplier kept
_gm = (_g0 := pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]).reindex(_K)).pipe(lambda g: (g.geo / g.F.clip(lower=1e-9)).fillna(1).values)
_nnx = np.where(np.isfinite(T.nn_ec.values), T.nn_ec.values, T.nn.values)
T["q_t2d_hpo"] = pd.read_parquet("data/fk_hpo_pf_final.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(_K).q50.values
T["nx5r_geo"] = _geoq("data/geo_nx5r_x_preds.parquet")   # D389i: the new nx5 (submission since 18 Nov): Huber tree routed to the x5 new-item tree for items under 6 months + rc at 1/4 + corrections
_cfx = pd.read_parquet("data/campaign_feats.parquet").set_index(["origin", "style_color"]).reindex(_K).x_cmp.fillna(0).values   # D389v-F2 / D411: campaign exposure
T["nx8_geo"] = T.nx5r_geo.values * np.exp(0.22 * _cfx)   # nx8 = nx7c: the new nx5 x the campaign uplift (half the x1.55 campaign-day lift, by change in campaign days vs last year x web share)
T["nx9_geo"] = _geoq("data/geo_nx5r_fash_x_preds.parquet") * np.exp(0.22 * _cfx)   # D412: nx9 = the new nx5 with the fashion fork's look-alike correction term (D402) x the campaign uplift
T["nx_hpo_pf_geo"] = AM(T.rules.values, T.q_ecns.values, _nnx, T.q_conf.values, T.q_t2d_hpo.values) * _gm
_rh = pd.read_parquet("data/rules_hpo_system.parquet").set_index(["origin", "style_color"]).reindex(_K)
T["rules_hpo"] = T.rules.values * (_rh.r_tuned / _rh.r_v14.clip(lower=1e-9)).fillna(1).clip(0.2, 5).values
T["nx_hpo_rules_geo"] = AM(T.rules_hpo.values, T.q_ecns.values, _nnx, T.q_conf.values, T.q_t2d.values) * _gm
# plain-language description, inputs and status of each system (shown on hover in the dashboard)
INFO = {
    "R3": ("Recency baseline: each item's last 3 months of sales x 4.", "Item's own last 3 months", "Reference"),
    "rules": ("Rule agents: isolated LLM agents write a site-level plan and per-item rule multipliers on the recent rate (v14 at the June 2026 origin).", "Recent rate, seasonal and event rules, site plan", "Reference"),
    "qmodel": ("LightGBM quantile model of the 4-month total as a ratio to the recent level; two feature sets averaged per quantile.", "Lags, level, seasonality (own, type, macro, occasion), channel, discount, attributes", "Source of the first submission (12 Nov); replaced on 18 Nov (D303)"),
    "nn": ("Neural quantile model with hierarchical embeddings (macro > type > style > item), a weekly sales-curve encoder and an item x month interaction; 6 seeds.", "Same numeric features as the trees, embeddings, 26 weeks of sales", "Member of the candidates"),
    "lgb": ("First pooled LightGBM point model of monthly sales, two feature sets averaged.", "Lags, level, seasonality, attributes", "Superseded"),
    "q_new": ("Lean quantile model (zero-value feature groups dropped, days to the item's occasion added) plus the cart neighbours' seasonal ramp and 14 LLM item scores.", "Lean features + cart co-purchase neighbours + LLM scores", "Candidate member"),
    "threeway": ("Equal geometric mean of the rule agents, the production quantile median and the NN.", "Rules, quantile model, NN", "Basis of the rules candidate workbook (615,937 units)"),
    "lean3": ("Three-way with the lean quantile model (occasion timing, no channel / discount / lifecycle groups).", "Rules, lean quantile model, NN", "Basis of the lean candidate (615,140 units)"),
    "lean3_bf": ("Lean three-way with the Black Friday fix: November level from last year's like-for-like ratio and item-lift allocation.", "Lean three-way + last-year November ratio + predicted item lift", "Best on buying loss; the fix only matters for windows with November"),
    "rd_share": ("Lean three-way where the quantile slot is shared with a member that samples the residual from R3 out of a predicted Normal distribution.", "Rules, lean quantile x residual-distribution member, NN", "Tested, not adopted (small gain)"),
    "new3": ("Three-way with the D153 quantile model (cart ramp + LLM scores).", "Rules, D153 quantile model, NN", "Tested, not adopted (worse buying loss)"),
    "new3_bf": ("D153 three-way with the Black Friday fix.", "Rules, D153 quantile model, NN, Black Friday fix", "Tested, not adopted (worse buying loss)"),
    "q_wx": ("D153 quantile model plus temperature (state normals weighted by each item's regional mix, its temperature preference) and a US events calendar with event x LLM-score interactions.", "D153 features + temperature + events", "New, not yet simulated"),
    "wx3": ("Three-way with the temperature + events quantile model.", "Rules, D159 quantile model, NN", "Simulated (D161): 11.30% buying loss, better than the plain three-way in 9 of 11 windows"),
    "q_dyn": ("D159 quantile model plus neighbour features from a time-aware neural co-occurrence network (Poisson model of cart pairs, refitted monthly).", "D159 features + 4 dynamic co-occurrence neighbour features", "Fork (D198); passed the cart co-occurrence gate (D199)"),
    "dyn3": ("Three-way with the dynamic co-occurrence quantile model.", "Rules, D198 quantile model, NN", "Fork (D203)"),
    "dyn3_bf": ("Dynamic co-occurrence three-way with the Black Friday fix.", "Rules, D198 quantile model, NN, Black Friday fix", "Fork (D203)"),
    "q_tab2": ("TabPFN v2 tabular foundation model on the D159 feature set (in-context learning, no training).", "D159 features", "Fork (D204); licence Apache 2.0 with attribution, usable in production"),
    "q_tab35f": ("TabPFN 3.5 (fast mode) tabular foundation model on the D159 feature set.", "D159 features", "Fork (D204); licence non-commercial only: research use, not for the production forecast"),
    "tab_p2": ("Four-member ensemble: rules, dynamic co-occurrence quantile model, NN and TabPFN v2 (equal geometric mean, no Black Friday fix).", "Rules, D198 quantile, NN, TabPFN v2", "Fork (D204); production-usable licence"),
    "tab_p35f": ("Four-member ensemble with TabPFN 3.5 fast instead of v2.", "Rules, D198 quantile, NN, TabPFN 3.5 fast", "Fork (D204); TabPFN 3.5 is non-commercial: research only"),
    "tab_p2_35f": ("Five-member ensemble: rules, dynamic co-occurrence quantile model, NN, TabPFN v2 and TabPFN 3.5 fast.", "Rules, D198 quantile, NN, TabPFN v2, TabPFN 3.5 fast", "Fork (D204); includes TabPFN 3.5, non-commercial: research only"),
    "q_sel": ("D198 LightGBM quantile model after feature pruning (null-importance selection on early origins).", "Pruned D198 features", "Fork (D206)"),
    "q_t2d": ("TabPFN v2 with the dynamic co-occurrence features, 32 estimators.", "D159 + dynamic co-occurrence features", "Fork (D206); licence Apache 2.0 with attribution"),
    "fk_p2d32": ("Rules, the D198 quantile model, the NN and TabPFN v2 with co-occurrence features (32 estimators), equal geometric mean.", "Rules, D198 quantile, NN, TabPFN v2 dyn 32", "Fork (D206); production-usable licence"),
    "fk_sel_d32": ("Rules, the pruned LightGBM quantile model, the NN and TabPFN v2 with co-occurrence features (32 estimators).", "Rules, pruned LightGBM, NN, TabPFN v2 dyn 32", "Fork (D206); production-usable licence; best licensable stack of the night"),
    "fk_p2d": ("Rules, the D198 quantile model, the NN and TabPFN v2 with co-occurrence features.", "Rules, D198 quantile, NN, TabPFN v2 dyn", "Fork (D205); production-usable licence"),
    "fk_p2c": ("p2d with the TabPFN member conformalised: each quantile shifted by earlier origins' residual quantiles per age x level segment.", "Rules, D198 quantile, NN, conformalised TabPFN v2 dyn", "Fork (D205); production-usable; lowest WAPE but bias -10% and higher buying loss"),
    "fk_p2c_35fd": ("p2c plus TabPFN 3.5 fast with co-occurrence features.", "p2c members + TabPFN 3.5 fast dyn", "Fork (D205); non-commercial licence: research only"),
    "fk_p2_selrep": ("Rules, the pruned LightGBM, the NN and TabPFN v2.", "Rules, pruned LightGBM, NN, TabPFN v2", "Fork (D206); production-usable licence"),
    "fk_sel_d32a": ("Arithmetic mean of rules, the pruned LightGBM, the NN and TabPFN v2 with co-occurrence features (32 estimators).", "Rules, pruned LightGBM, NN, TabPFN v2 dyn 32", "Fork (D206); production-usable; arithmetic mean removes most of the geometric mean's under-forecast"),
    "fk_p2d32a": ("Arithmetic mean of rules, the D198 quantile model, the NN and TabPFN v2 with co-occurrence features (32 estimators).", "Rules, D198 quantile, NN, TabPFN v2 dyn 32", "Fork (D206); production-usable; lowest buying loss of the licensable systems (D190)"),
    "q_w03": ("D198 LightGBM with training rows weighted by (4 x level + 1)^0.3.", "D198 features", "Fork"),
    "fk_p2c32a": ("Arithmetic mean of rules, D198 quantile, NN, conformalised TabPFN v2 dyn and TabPFN v2 dyn 32.", "5 members", "Fork; production-usable"),
    "fk_qsc32a": ("Arithmetic mean of rules, D198 quantile, pruned LightGBM, NN, conformalised TabPFN v2 dyn and TabPFN v2 dyn 32.", "6 members", "Fork; production-usable; best accuracy of the licensable systems (D231)"),
    "fk_qs_d32a": ("Arithmetic mean of rules, D198 quantile, pruned LightGBM, NN and TabPFN v2 dyn 32.", "5 members", "Fork; production-usable"),
    "fk_w03d32a": ("Arithmetic mean of rules, the volume-weighted LightGBM, NN and TabPFN v2 dyn 32.", "4 members", "Fork; production-usable; lowest buying loss of the licensable systems (D231)"),
    "fk_w03c32a": ("Arithmetic mean of rules, volume-weighted LightGBM, NN, conformalised TabPFN v2 dyn and TabPFN v2 dyn 32.", "5 members", "Fork; production-usable (D232)"),
    "fk_w03_2a": ("Arithmetic mean of rules, volume-weighted LightGBM, NN, TabPFN v2 and TabPFN v2 dyn 32; about zero bias on recent origins.", "5 members", "Fork; production-usable; lowest buying loss measured (D232)"),
    "fk_w03c32a_ut": ("w03c32a with the volume-weighted LightGBM at its P60 (newsvendor ladder).", "5 members, one at P60", "Fork; production-usable (D233)"),
    "fk_w03c32a_uc": ("w03c32a with the conformalised TabPFN at its P60.", "5 members, one at P60", "Fork; production-usable (D233)"),
    "fk_w03c32a_u": ("w03c32a with the LightGBM and both TabPFN members at their P60: bias about +2% recent.", "5 members, three at P60", "Fork; production-usable; lowest buying loss measured (D233)"),
    "fk_u65": ("w03c32a with the LightGBM and both TabPFN members at their P65 (newsvendor ladder; P65 interpolated by the fork).", "5 members, three at P65", "Fork + Main (D236); production-usable; lowest buying loss measured, with u70"),
    "fk_u70": ("w03c32a with the LightGBM and both TabPFN members at their P70 (newsvendor ladder).", "5 members, three at P70", "Fork + Main (D236); production-usable"),
    "fk_p2c32a_seg": ("p2c32a with a time-honest re-allocation between volume x age x discount segments (half of each segment's past bias, each origin's total kept).", "p2c32a + past segment biases", "Main (D245, D246); production-usable"),
    "fk_p2c32a_geo": ("p2c32a corrected by the geometric mean of the segment re-allocation and a time-honest breakout model (weekly momentum, look-peer growth, age, discount), each origin's total kept.", "p2c32a + segment biases + weekly / look-peer signals", "Main (D247-D249); production-usable; best accuracy of the licensable systems; buying loss level"),
    "q_ecns": ("The D198 LightGBM with 15 more event windows and launch curves, without the static item descriptors (median).", "D198 features + events + launch curves - static descriptors", "Main (D253); member"),
    "nx_ecns": ("p2c32a with the D198 LightGBM replaced by the events + launch-curves version (D253).", "5 members", "Main (D253); production-usable"),
    "nx_ecns_geo": ("nx_ecns with the segment re-allocation and breakout corrections re-fitted on it (geometric mean, totals kept).", "5 members + time-honest corrections", "Main (D257); production-usable"),
    "nx_ecnsx_geo": ("nx_ecns_geo with promotion sensitivity, recent spike, time since peak and launch-curve signals in the breakout correction.", "5 members + time-honest corrections", "Main (D259); production-usable; best accuracy of the licensable systems"),
    "nx_u65ec": ("Newsvendor ladder u65 with the volume-weighted LightGBM rebuilt on the events + launch-curves features (q65).", "5 members, three at P65", "Main (D260); production-usable"),
    "nx_w03ecx_geo": ("nx_ecnsx_geo with the volume-weighted version of the new tree member.", "5 members + time-honest corrections", "Main (D264); production-usable; level with nx_ecnsx_geo"),
    "nx_w03ecpl": ("System with the volume-weighted new tree member plus the look-level launch plan (new colours of the same style, look-alike launches in the window).", "5 members; needs the launch calendar", "Main (D264); research only: needs Edikted's launch plan (none known for Jul-Oct 2026)"),
    "nx_newitems": ("nx_ecnsx_geo, with items under 6 months averaged with a separate new-items 4-month model (learned imputation of missing history).", "5 members + corrections + new-items model", "Main (D263); level"),
    "nx_u65ec_geo": ("The u65ec newsvendor ladder with the segment and breakout corrections applied.", "5 members at P65 + corrections", "Main (D265); level on buying"),
    "nn_ec": ("The embedding NN retrained with the 15 extra event windows and launch-curve inputs (6 seeds).", "NN inputs + events + launch curves", "Fork (D272); member"),
    "nx_ecnsnn": ("nx_ecns with the NN replaced by the events + launch-curves NN.", "5 members", "Main (D273); production-usable"),
    "nx_ecnsnn_geo": ("nx_ecnsnn with the segment and breakout corrections re-fitted on it (geometric mean, totals kept).", "5 members + time-honest corrections", "Source of the previous submission (candidate_nx4, until 18 Nov 03:40: this system plus the World Cup and out-of-stock hand rules, the D318 confidence rule and the D317 ranges; D291, D303, D299, D350); replaced by the new nx5 (D389i)"),
    "q_ecsig": ("The new tree member refitted with 11 origin-known signals: stock-gap days, size run and concentration, newest colour, partner share and count, share of the style, live colours, web discount and its gap to the site's, largest day's share.", "D253 features + 11 signals", "Fork (D307); member"),
    "nx_ecsig": ("nx_ecnsnn with the tree member refitted on the 11 signals.", "5 members", "Main (D293); production-usable"),
    "nx_ecsig_geo": ("nx_ecsig with the segment and breakout corrections re-fitted on it.", "5 members + time-honest corrections", "Main (D293); production-usable; best accuracy of the licensable systems; not in the workbook yet"),
    "nx5r_geo": ("D361's nx5 (Huber tree, NN, rules, both TabPFN members, the recommender at a quarter weight) with the tree member for items under 6 months old replaced by the Huber tree trained with new items' rows x5, corrections re-fitted on it.", "6 members (routed tree) + time-honest corrections", "Submitted as nx6 (D389t) until the campaign uplift (nx7c, D389x) and nx8 (D421) took its place; same backtest as candidate_nx5 / nx6"),
    "nx8_geo": ("The new nx5 (routed new-item tree, recommender at a quarter, corrections) times the campaign uplift: exp(0.22 x change in campaign days against the same months last year / window days x the item's edikted.com share); 0.22 = half the measured x1.55 lift of a campaign day. In 2026 it raises August and September for the plan's two new sales (Clearance 12-19 Aug, 24-29 Sep).", "6 members (routed tree) + corrections + campaign uplift", "Submitted as nx8 (18 Nov, D421) until nx9 (D422): Edikted_Forecasting_Exercise_filled.xlsx = this system plus the World Cup and out-of-stock rules, the focus-weighted ranges, the hybrid confidence column and the audit's corrected texts; nx7c has the same numbers"),
    "nx9_geo": ("nx8 with the fashion fork's look-alike term in the correction layer (D402): the new nx5 (routed new-item tree, recommender at a quarter, corrections incl. look-alikes) times the campaign uplift for the plan's new sales.", "6 members (routed tree) + corrections with look-alikes + campaign uplift", "SUBMITTED as nx11 (18 Nov, D417 / D418): Edikted_Forecasting_Exercise_filled.xlsx = this system plus the World Cup and out-of-stock rules, the focus-weighted ranges, the hybrid confidence column, the repaired texts and the improvement roadmap; nx9 has the same numbers"),
    "nx_hpo_pf_geo": ("The submitted system with the TabPFN member (pf2d32) replaced by its overnight-tuned settings (32 estimators, temperature 1.23, no cap); submitted correction multiplier kept.", "5 members + submitted corrections", "Tuned overnight (D326): level in the system, not adopted"),
    "nx_hpo_rules_geo": ("The submitted system with the rules member scaled by the tuned/v14 ratio from the overnight rules search (agents' oracle-site harness, 16 knobs); submitted correction multiplier kept.", "5 members + submitted corrections", "Tuned overnight (D389 / D389b): level in the system, v14 stands"),
    "nx_hub_geo": ("nx_ecnsnn_geo with the tree member trained on Huber loss (delta 0.5) on the log ratio instead of absolute error; corrections re-fitted.", "5 members + time-honest corrections", "Tested (D353)"),
    "nx_rcd_geo": ("nx_ecnsnn_geo plus the per-location recommender (rc_deepfm, rc_gbm below 12 training origins) as a sixth member; corrections re-fitted.", "6 members + time-honest corrections", "Tested (D354)"),
    "nx_hubrc_geo": ("Huber-loss tree member plus the per-location recommender (rc_deepfm) as a sixth member; segment and breakout corrections re-fitted.", "6 members + time-honest corrections", "Tested (D355): best backtest, fails the last-year-removed gate (D358)"),
    "nx_ecnsnnhubrq_geo": ("Huber-loss tree member plus the per-location recommender (rc_deepfm) at a quarter of a member's weight; segment and breakout corrections re-fitted.", "5 members + recommender at weight 0.25 + time-honest corrections", "Candidate (D358): passes every gate, incl. last-year-removed; waits on the user's decision"),
    "wx3_bf": ("Temperature + events three-way with the Black Friday fix.", "Rules, D159 quantile model, NN, Black Friday fix", "Best on accuracy and buying loss (D161): 11.05% buying loss; deliverable not switched"),
    "nosea3": ("Three-way whose quantile model has no seasonal indices: temperature, events and LLM scores replace them.", "Rules, quantile model without seasonality + temperature + events, NN", "Experiment: temperature + events replace seasonality"),
}
# id, name, decision, role, column, simulation source (file, system/variant)
SYSTEMS = [
    ("R3", "R3 (last 3 months x 4)", "D109", "reference", "R3", ("sys", "R3")),
    ("rules", "R3 + rules (rule agents)", "D105", "reference", "rules", ("sys", "rules")),
    ("qmodel", "Quantile model (production)", "D042", "member", "qmodel", ("sys", "qmodel")),
    ("nn", "Embedding NN (6 seeds)", "D097", "member", "nn", ("sys", "nn")),
    ("lgb", "LightGBM ensemble", "D013", "member", "lgb", None),
    ("q_new", "Quantile model: lean + cart ramp + LLM scores", "D153", "member", "q_new", None),
    ("threeway", "Rules + quantile + NN", "D105", "ensemble", "r_q_nn", ("sys", "r_q_nn")),
    ("lean3", "Rules + lean quantile + NN", "D135", "ensemble", "r_qlean_nn", ("best", "lean three-way")),
    ("lean3_bf", "Lean three-way + Black Friday fix", "D141", "ensemble", "best", ("best", "lean three-way + BF fix")),
    ("rd_share", "Lean three-way, residual member shares quantile slot", "D148", "ensemble", "r_share_rd", None),
    ("new3", "Rules + new quantile + NN", "D153", "ensemble", "r_qnew_nn", ("new", "new three-way")),
    ("new3_bf", "New three-way + Black Friday fix", "D153", "ensemble", "best_new", ("new", "new three-way + BF fix")),
    ("q_wx", "Quantile model: D153 + temperature + events", "D159", "member", "q_wx", None),
    ("nosea3", "Three-way, seasonality replaced by temperature + events", "D159", "ensemble", "r_qnosea_nn", None),
    ("wx3", "Rules + temperature/events quantile + NN", "D159", "ensemble", "r_qwx_nn", ("wx", "d159 three-way")),
    ("wx3_bf", "Temperature/events three-way + Black Friday fix", "D159", "ensemble", "best_wx", ("wx", "d159 three-way + BF fix")),
    ("q_dyn", "Quantile model: D159 + dynamic co-occurrence neighbours", "D198", "member", "q_dyn", None),
    ("dyn3", "Rules + dynamic co-occurrence quantile + NN", "D203", "ensemble", "r_qdyn_nn", ("dyn", "dyn three-way")),
    ("dyn3_bf", "Dynamic co-occurrence three-way + Black Friday fix", "D203", "ensemble", "best_dyn", ("dyn", "dyn three-way + BF fix")),
    ("q_tab2", "TabPFN v2 (tabular foundation model)", "D204", "member", "q_tab2", None),
    ("q_tab35f", "TabPFN 3.5 fast (non-commercial licence)", "D204", "member", "q_tab35f", None),
    ("tab_p2", "Rules + dynamic quantile + NN + TabPFN v2", "D204", "ensemble", "tab_p2", ("tab", "p2")),
    ("tab_p35f", "Rules + dynamic quantile + NN + TabPFN 3.5 fast (non-commercial)", "D204", "ensemble", "tab_p35f", ("tab", "p35f")),
    ("tab_p2_35f", "Rules + dynamic quantile + NN + TabPFN v2 + 3.5 fast (non-commercial)", "D204", "ensemble", "tab_p2_35f", ("tab", "p2_35f")),
    ("q_sel", "Pruned LightGBM quantile model", "D206", "member", "q_sel", None),
    ("q_t2d", "TabPFN v2 + co-occurrence features (32 estimators)", "D206", "member", "q_t2d", None),
    ("fk_p2d32", "Rules + dynamic quantile + NN + TabPFN v2 dyn", "D206", "ensemble", "fk_p2d32", ("fkm5", "p2d32")),
    ("fk_sel_d32", "Rules + pruned LightGBM + NN + TabPFN v2 dyn", "D206", "ensemble", "fk_sel_d32", ("fkm5", "sel_d32")),
    ("fk_p2d", "Rules + dynamic quantile + NN + TabPFN v2 dyn (16 est.)", "D205", "ensemble", "fk_p2d", ("fk2", "p2d")),
    ("fk_p2c", "Rules + dynamic quantile + NN + conformalised TabPFN v2 dyn", "D205", "ensemble", "fk_p2c", ("fk2", "p2c")),
    ("fk_p2c_35fd", "p2c + TabPFN 3.5 fast dyn (non-commercial)", "D205", "ensemble", "fk_p2c_35fd", ("fk2", "p2c_35fd")),
    ("fk_p2_selrep", "Rules + pruned LightGBM + NN + TabPFN v2", "D206", "ensemble", "fk_p2_selrep", ("fk2", "p2_selrep")),
    ("fk_sel_d32a", "Arithmetic mean: rules + pruned LightGBM + NN + TabPFN v2 dyn", "D206", "ensemble", "fk_sel_d32a", ("fk3", "sel_d32a")),
    ("fk_p2d32a", "Arithmetic mean: rules + dynamic quantile + NN + TabPFN v2 dyn", "D206", "ensemble", "fk_p2d32a", ("fk3", "p2d32a")),
    ("q_w03", "Volume-weighted LightGBM quantile model", "D206", "member", "q_w03", None),
    ("fk_p2c32a", "Arithmetic: rules + dyn quantile + NN + conformal TabPFN + TabPFN dyn", "D206", "ensemble", "fk_p2c32a", ("fk4", "p2c32a")),
    ("fk_qsc32a", "Arithmetic: 6 members (+ pruned LightGBM)", "D206", "ensemble", "fk_qsc32a", ("fk4", "qsc32a")),
    ("fk_qs_d32a", "Arithmetic: rules + dyn quantile + pruned LightGBM + NN + TabPFN dyn", "D206", "ensemble", "fk_qs_d32a", ("fk4", "qs_d32a")),
    ("fk_w03d32a", "Arithmetic: rules + volume-weighted LightGBM + NN + TabPFN dyn", "D206", "ensemble", "fk_w03d32a", ("fk4", "w03d32a")),
    ("fk_w03c32a", "Arithmetic: rules + volume-weighted LightGBM + NN + conformal TabPFN + TabPFN dyn", "D206", "ensemble", "fk_w03c32a", ("fk5", "w03c32a")),
    ("fk_w03_2a", "Arithmetic: rules + volume-weighted LightGBM + NN + TabPFN v2 + TabPFN dyn (bias-neutral)", "D206", "ensemble", "fk_w03_2a", ("fk5", "w03_2a")),
    ("fk_w03c32a_ut", "w03c32a, LightGBM at P60", "D206", "ensemble", "fk_w03c32a_ut", ("fk6", "w03c32a_ut")),
    ("fk_w03c32a_uc", "w03c32a, conformal TabPFN at P60", "D206", "ensemble", "fk_w03c32a_uc", ("fk6", "w03c32a_uc")),
    ("fk_w03c32a_u", "w03c32a, three members at P60 (newsvendor)", "D206", "ensemble", "fk_w03c32a_u", ("fk6", "w03c32a_u")),
    ("fk_u65", "w03c32a, three members at P65 (newsvendor ladder)", "D236", "ensemble", "fk_u65", ("fk7", "u65")),
    ("fk_u70", "w03c32a, three members at P70 (newsvendor ladder)", "D236", "ensemble", "fk_u70", ("fk7", "u70")),
    ("fk_p2c32a_seg", "p2c32a + segment bias re-allocation", "D245", "ensemble", "fk_p2c32a_seg", ("seg", "p2c32a_seg")),
    ("fk_p2c32a_geo", "p2c32a + segment re-allocation and breakout model (geometric mean)", "D248", "ensemble", "fk_p2c32a_geo", ("geo", "p2c32a_geo")),
    ("q_ecns", "LightGBM + events + launch curves, no static descriptors", "D253", "member", "q_ecns", None),
    ("nx_ecns", "p2c32a with the events + launch-curves LightGBM", "D253", "ensemble", "nx_ecns", ("nx", "ec_ns|lgb")),
    ("nx_ecns_geo", "nx_ecns + segment and breakout corrections", "D257", "ensemble", "nx_ecns_geo", ("nx", "ec_ns|lgb|geo")),
    ("nx_ecnsx_geo", "nx_ecns + corrections with falling-item signals", "D259", "ensemble", "nx_ecnsx_geo", ("nx", "ec_ns_x|lgb|geo")),
    ("nx_u65ec", "Newsvendor ladder P65 with the new-feature LightGBM", "D260", "ensemble", "nx_u65ec", ("lnx", "u65ec")),
    ("nx_w03ecx_geo", "Volume-weighted new LightGBM + corrections", "D264", "ensemble", "nx_w03ecx_geo", None),
    ("nx_w03ecpl", "New LightGBM + look-level launch plan (needs the plan)", "D264", "ensemble", "nx_w03ecpl", None),
    ("nx_newitems", "Best system + separate new-items model (young items)", "D263", "ensemble", "nx_newitems", None),
    ("nx_u65ec_geo", "Ladder P65 + corrections", "D265", "ensemble", "nx_u65ec_geo", ("chase", "u65ec_geo")),
    ("nn_ec", "NN + events + launch curves", "D272", "member", "nn_ec", None),
    ("nx_ecnsnn", "New LightGBM + new NN", "D273", "ensemble", "nx_ecnsnn", ("nn", "ec_ns+nn|lgb")),
    ("nx_ecnsnn_geo", "New LightGBM + new NN + corrections", "D273", "ensemble", "nx_ecnsnn_geo", ("nn", "ec_ns+nn|lgb|geo")),
    ("q_ecsig", "LightGBM + 11 origin-known signals", "D307", "member", "q_ecsig", None),
    ("nx_ecsig", "11-signal LightGBM + new NN", "D293", "ensemble", "nx_ecsig", ("nnsig", "ecsig+nn|lgb")),
    ("nx_ecsig_geo", "11-signal LightGBM + new NN + corrections", "D293", "ensemble", "nx_ecsig_geo", ("nnsig", "ecsig+nn|lgb|geo")),
    ("nx_hub_geo", "Huber-loss LightGBM + new NN + corrections", "D353", "ensemble", "nx_hub_geo", None),
    ("nx_rcd_geo", "New LightGBM + new NN + recommender + corrections", "D354", "ensemble", "nx_rcd_geo", None),
    ("nx_hubrc_geo", "Huber LightGBM + new NN + recommender + corrections", "D355", "ensemble", "nx_hubrc_geo", None),
    ("nx_ecnsnnhubrq_geo", "Huber LightGBM + new NN + recommender at 1/4 weight + corrections", "D358", "ensemble", "nx_ecnsnnhubrq_geo", None),
    ("nx5r_geo", "new nx5 / nx6: routed new-item tree", "D389i", "ensemble", "nx5r_geo", None),
    ("nx8_geo", "nx8 = new nx5 + campaign uplift", "D421", "ensemble", "nx8_geo", None),
    ("nx9_geo", "nx9 / nx10 / nx11 = nx8 + look-alike correction (submitted as nx11)", "D422", "ensemble", "nx9_geo", None),
    ("nx_hpo_pf_geo", "nx4, TabPFN tuned overnight", "D326", "ensemble", "nx_hpo_pf_geo", None),
    ("nx_hpo_rules_geo", "nx4, rules tuned overnight", "D389b", "ensemble", "nx_hpo_rules_geo", None),
]
S0 = pd.read_csv("data/sim_systems_windows.csv"); S0 = S0[S0.policy == "production"]
NONCOM = {"q_tab35f", "tab_p35f", "tab_p2_35f", "fk_p2c_35fd"}   # TabPFN 3.5: non-commercial licence; never champion (D169)
SIMF = {"sys": S0.rename(columns={"system": "key"}), "best": pd.read_csv("data/sim_best_stack_windows.csv").rename(columns={"variant": "key"}),
        "new": pd.read_csv("data/sim_new_stack_windows.csv").rename(columns={"variant": "key"}),
        "wx": pd.read_csv("data/sim_wx_stack_windows.csv").rename(columns={"variant": "key"}),   # D161 (archive/eda/168)
        "dyn": pd.read_csv("data/sim_dyn_stack_windows.csv").rename(columns={"variant": "key"}),   # D165 (archive/eda/171)
        "tab": pd.read_csv("data/sim_tabpfn_stack_windows.csv").rename(columns={"variant": "key"}),   # D169 (archive/eda/175)
        "fkm5": pd.read_csv("data/sim_fkm5_stack_windows.csv").rename(columns={"variant": "key"}),   # D187 (archive/eda/175b)
        "fk2": pd.read_csv("data/sim_fk2_stack_windows.csv").rename(columns={"variant": "key"}),   # D188 (archive/eda/175c)
        "fk3": pd.read_csv("data/sim_fk3_stack_windows.csv").rename(columns={"variant": "key"}),   # D230 (archive/eda/175d)
        "fk4": pd.read_csv("data/sim_fk4_stack_windows.csv").rename(columns={"variant": "key"}),   # D231 (archive/eda/175e)
        "fk5": pd.read_csv("data/sim_fk5_stack_windows.csv").rename(columns={"variant": "key"}),   # D232 (archive/eda/175f)
        "fk6": pd.read_csv("data/sim_fk6_stack_windows.csv").rename(columns={"variant": "key"}),   # D233 (archive/eda/175g)
        "fk7": pd.read_csv("data/sim_fk7_stack_windows.csv").rename(columns={"variant": "key"}),   # D236 (archive/eda/175h)
        "seg": pd.read_csv("data/sim_seg_windows.csv").rename(columns={"variant": "key"}),   # D246 (archive/eda/175i)
        "geo": pd.read_csv("data/sim_geo_windows.csv").rename(columns={"variant": "key"}),   # D249 (archive/eda/175j)
        "nx": pd.read_csv("data/sim_nx_windows.csv").rename(columns={"variant": "key"}),   # D260 (archive/eda/175k)
        "lnx": pd.read_csv("data/sim_ladder_nx_windows.csv").rename(columns={"variant": "key"}),   # D260 (archive/eda/175l)
        "chase": pd.read_csv("data/sim_chase_windows.csv").rename(columns={"variant": "key"}),   # D265 (archive/eda/175m)
        "nn": pd.read_csv("data/sim_nn_windows.csv").rename(columns={"variant": "key"}),   # D273 (archive/eda/175n)
        "nnsig": pd.read_csv("data/sim_nnsig_windows.csv").rename(columns={"variant": "key"})}   # D293 (archive/eda/175p)
def simw(src): f, k = src; x = SIMF[f]; return x[x.key == k].set_index("start").loss_pct
r3loss = simw(("sys", "R3")).mean()
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
SUBS = [("recent", "Nine recent origins", REC), ("all", "All 13 origins", np.ones(len(T), bool)), ("targets", "The 100 targets (recent)", REC & T.target.values),
        ("bf", "Black Friday origins", BF),
        # D294 (error-study proposals #13 / #12c / #14b): the four early origins (the gate for new members) and origins by season, all 13 origins
        ("early", "Four early origins (Feb-May 2025)", T.origin.isin(["2025-02", "2025-03", "2025-04", "2025-05"]).values),
        ("s_spring", "Spring origins (Feb-Apr)", T.origin.str[-2:].isin(["02", "03", "04"]).values), ("s_summer", "Summer origins (May-Jul)", T.origin.str[-2:].isin(["05", "06", "07"]).values),
        ("s_autumn", "Autumn origins (Aug-Oct)", T.origin.str[-2:].isin(["08", "09", "10"]).values), ("s_winter", "Winter origins (Nov-Jan)", T.origin.str[-2:].isin(["11", "12", "01"]).values)]
# D294: the seasonal naive (R3's monthly level x the product type's like-for-like seasonal ratio per target month), as the error study's
_fe = pd.read_parquet("data/features.parquet"); _fe["st"] = _fe.m3 * np.exp(_fe.lfl_type.fillna(0))
T["sea_naive"] = _fe.groupby(["origin", "style_color"]).st.sum().reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color])).values
def blk(Z, c, ref="rules", B=2000, seed=0):   # D294: origin-block bootstrap of WAPE / R3 (system) - WAPE / R3 (ref): whole origins resampled
    g = Z.assign(a=(Z[c] - Z.y).abs(), b=(Z[ref] - Z.y).abs(), r=(Z.R3 - Z.y).abs()).groupby("origin")[["a", "b", "r"]].sum().values
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(g), (B, len(g))); G = g[ix].sum(1); d = (G[:, 0] - G[:, 1]) / G[:, 2]
    return round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)
cols = [s[4] for s in SYSTEMS]; rows = []
BOOT = {}                                    # 95% item-bootstrap interval of WAPE(system) - WAPE(reference) / WAPE(R3), within origins
for sid, sname, m in SUBS:
    if sid not in ("recent", "all"): continue
    Z = T[m].reset_index(drop=True); e3 = (Z.R3 - Z.y).abs().sum() / Z.y.sum()
    for s_, name, dec, role, c, src in SYSTEMS:
        if s_ in ("R3", "rules"): continue
        a = sl.boot(Z, c, "rules", B=300) / e3; b = sl.boot(Z, c, "r_q_nn", B=300) / e3 if c != "r_q_nn" else (None, None)
        BOOT[(sid, s_)] = (round(float(a[0]), 4), round(float(a[1]), 4), None if b[0] is None else round(float(b[0]), 4), None if b[1] is None else round(float(b[1]), 4))
    print("bootstrap done:", sid, flush=True)
def xm(Z, c):   # D293 (user via the fork's D289 / D305): per-item views next to WAPE / R3
    F, y = Z[c].values.astype(float), Z.y.values.astype(float); ok = np.isfinite(F) & np.isfinite(y)
    F, y = F[ok], y[ok]; ape = np.abs(F - y) / np.maximum(y, 1e-9); h = (np.abs(F - y) <= 0.2 * y) & (y > 0)
    r2u = 1 - ((F - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(); ly, lf = np.log1p(y), np.log1p(np.clip(F, 0, None)); r2l = 1 - ((lf - ly) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    S, R = Z.sea_naive.values.astype(float)[ok], Z.R3.values.astype(float)[ok]; Sok = np.isfinite(S)
    rel = (np.abs(F - y) + 1) / (np.abs(R - y) + 1)
    return {"within20": round(float(h.mean()), 4), "within20_units": round(float((h * y).sum() / y.sum()), 4), "median_ape": round(float(np.median(ape[y > 0])), 4),
            "r2_units": round(float(r2u), 4), "r2_log": round(float(r2l), 4),
            "abs_wape": round(float(np.abs(F - y).sum() / y.sum()), 4),                                                   # D294: absolute WAPE (error / units)
            "wape_sea": round(float(np.abs(F - y)[Sok].sum() / np.abs(S - y)[Sok].sum()), 4),                            # D294: ratio to the seasonal naive
            "mdrae": round(float(np.median(rel)), 4)}                                                                   # D294: median of (|e| + 1) / (|R3 e| + 1)
for sid, sname, m in SUBS:
    Z = T[m].reset_index(drop=True); tb = sl.table(Z, [c for c in cols if c != "rules"] + ["rules"], {k: v[m] for k, v in D.items()})
    tb = tb.rename(index={"R3 + rules (rule agents)": "rules"})
    for i, (s, name, dec, role, c, src) in enumerate(SYSTEMS):
        r = tb.loc[c]; loss = simw(src).mean() if src else np.nan
        bt = BOOT.get((sid, s), (None, None, None, None)); de, inp, st = INFO[s]
        bk = blk(Z, c) if (sid in ("recent", "all") and s not in ("R3", "rules")) else (None, None)
        rows.append({"system_id": s, "system": name, "decision": dec, "order": i, "role": role, "licence": ("non-commercial" if s in NONCOM else "ok"), "subset": sid, "subset_name": sname, "item_origins": int(m.sum()),
                     "description": de, "inputs": inp, "status": st, "boot_rules_lo": bt[0], "boot_rules_hi": bt[1], "boot_3w_lo": bt[2], "boot_3w_hi": bt[3], "blk_rules_lo": bk[0], "blk_rules_hi": bk[1],
                     "blk_rules": (f"{bk[0]:+.3f} to {bk[1]:+.3f}" if bk[0] is not None else None),
                     "wape_r3": round(float(r["WAPE / R3"]), 4), "relmae_r3": round(float(r["relMAE vs R3"]), 4), "win_r3": round(float(r["win vs R3"]), 4),
                     "win_rules": round(float(r["win vs R3+rules"]), 4), "mab_r3": round(float(r["(MAE+|bias|) / R3"]), 4), "pinball_r3": round(float(r["pinball / R3"]), 4),
                     "bias": round(float(r["bias"]), 4), **xm(Z, c), "sim_loss": (round(float(loss), 4) if src else None), "sim_loss_r3": (round(float(loss / r3loss), 4) if src else None)})
# D294 (#14b): gate for new systems. A system scored better than the submitted one on the nine recent origins passes only if it is also
# better on the four early origins (the error study found steps that reverse there). Last-year-window removal is checked separately (fork #10a).
SUBMITTED = "nx9_geo"   # D389i (was nx_ecnsnn_geo, candidate_nx4)
_w = {(r["subset"], r["system_id"]): r["wape_r3"] for r in rows}
for r in rows:
    rec, sub_rec = _w.get(("recent", r["system_id"])), _w.get(("recent", SUBMITTED)); ea, sub_ea = _w.get(("early", r["system_id"])), _w.get(("early", SUBMITTED))
    r["early_wape_r3"] = ea
    r["gate_early"] = (None if (r["role"] == "reference" or r["system_id"] == SUBMITTED or rec is None or rec >= sub_rec) else bool(ea <= sub_ea))
B = pd.DataFrame(rows); print(B[B.subset == "recent"][["system", "wape_r3", "relmae_r3", "pinball_r3", "bias", "sim_loss"]].to_string(index=False))
code = open(__file__, encoding="utf-8").read()
meta = [{"label": "159_scoreboard.py", "value": code, "type": "code", "lang": "python"},
        {"label": "Input", "value": "Backtest predictions of each system on 13 origins (data/fixes_items.parquet, q_backtest_*.parquet, bf_fix_ratios.parquet) and buying-simulation windows (data/sim_*_windows.csv)"},
        {"label": "Decision log", "value": "log.md in the project folder; each system's decision id (e.g. D153) points to its entry"}]
json.dump({"rows": rows, "meta": meta}, open("data/scoreboard.json", "w", encoding="utf-8"))
W = []
for s, name, dec, role, c, src in SYSTEMS:
    if src:
        for start, v in simw(src).items(): W.append({"system_id": s, "system": name, "window_start": start, "loss": round(float(v), 4)})
json.dump({"rows": W, "meta": meta[:1] + [{"label": "Input", "value": "data/sim_systems_windows.csv, sim_best_stack_windows.csv, sim_new_stack_windows.csv (production buying policy)"}]},
          open("data/scoreboard_sim.json", "w", encoding="utf-8"))
print("rows", len(rows), "| sim rows", len(W))
