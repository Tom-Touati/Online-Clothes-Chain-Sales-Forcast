"""Atlas data for the evaluation & error-analysis section (D132, extended to the submitted system at D275): adds J["eval"] to
data/project_map.json (run after eda/19). All numbers are computed here from the stored outputs: the standard R3-relative table
(eda/score_lib) for the systems up to the submitted one (stored backtests; each reproduces its logged score), the stage-by-stage value
from R3, the 2026 cascade (eda/19: data/final_2026_nx2_cascade_hierarchy.csv), bias by target month (the D132 members' monthly
forecasts), bias by origin x product group and the error segments of the submitted system, and the probe ledger (verdicts with their
decision ids). Buying losses not stored per system are taken from the log (decision ids in the code)."""
import json, numpy as np, pandas as pd, importlib.util, sys, os
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
J = json.load(open("data/project_map.json", encoding="utf-8")); E = {}
T = pd.read_parquet("data/fixes_items.parquet")
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type"]); F["st"] = F.m3 * np.exp(F.lfl_type.fillna(0))
T = T.merge(F.groupby(["origin", "style_color"]).st.sum().rename("sea_type").reset_index(), on=["origin", "style_color"], how="left")
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]; R = T[T.origin.isin(REC)]
S = pd.read_csv("data/sim_systems_windows.csv"); S = S[S.policy == "production"].groupby("system").loss_pct.mean(); SIM = {k: v / S["R3"] for k, v in S.items()}
NM = {"qmodel": "Quantile model", "lgb": "LightGBM (monthly)", "nn": "Embedding NN", "lgb_nn": "LightGBM + NN", "q_nn": "Quantile + NN", "r_q": "Rules + quantile", "r_q_nn": "Rules + quantile + NN"}
def tab(Z, cols, names, sim=None):
    t = sl.table(Z, cols, D, names, sim); return [dict(name=i, **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}) for i, r in t.iterrows()]
# ---- systems after D132: stored backtests, same items and origins ----
KEY = ["origin", "style_color"]
def col(path, c): return T[KEY].merge(pd.read_parquet(path)[KEY + [c]], on=KEY, how="left")[c].values
T["wx3_bf"] = col("data/fk_stack_preds.parquet", "wx3_bf"); T["dyn3_bf"] = col("data/fk_stack_preds.parquet", "dyn3_bf"); T["p2"] = col("data/fk_stack_m5_preds.parquet", "p2")
T["p2c32a"] = col("data/dec_base.parquet", "F"); T["ecns"] = col("data/dec_base_ec_ns.parquet", "F"); T["ecnsnn"] = col("data/dec_base_ecnsnn.parquet", "F")
T["final"] = col("data/geo_ecnsnn_x_preds.parquet", "geo")
for c_, k_ in [("m_tree", "qdyn"), ("m_nn", "nn"), ("m_pf2c", "pf2c"), ("m_pf2d", "pf2d32"), ("bf_window", "bf_window"), ("spread", "spread")]: T[c_] = col("data/dec_base_ecnsnn.parquet", k_)
NEWC = ["wx3_bf", "dyn3_bf", "p2", "p2c32a", "ecns", "ecnsnn", "final", "m_tree", "m_nn", "m_pf2c", "m_pf2d"]; assert T[NEWC].notna().all().all()
R = T[T.origin.isin(REC)]
NM = {"qmodel": "Quantile model alone (state at D132)", "r_q_nn": "Rules + quantile + NN, geometric (D132)", "wx3_bf": "+ temperature, events, Black Friday fix (D161)",
      "dyn3_bf": "+ dynamic co-purchase neighbours (D203)", "p2": "+ TabPFN v2 as a fourth member (D169)", "p2c32a": "Arithmetic mean of five, conformal TabPFN (D231)",
      "ecns": "New tree member: 31 event windows + launch curves (D253)", "ecnsnn": "New NN on the same inputs (D272)", "final": "Submitted: + segment and breakout corrections (D273)",
      "m_tree": "member alone: LightGBM, events + launch curves", "m_nn": "member alone: sales-curve NN (D272)", "m_pf2c": "member alone: TabPFN v2, conformal", "m_pf2d": "member alone: TabPFN v2, 32 estimators"}
# buying loss (production policy, 11 windows) as a ratio to R3's. Stored: the D132 systems (sim_systems_windows) and the submitted system (sim_nn_windows).
# From the log: wx3_bf 11.05% (D161), dyn3_bf 11.09% (D165), p2 10.78% (D169), p2c32a 10.74% (D231), ecns 10.71% (D260).
sn = pd.read_csv("data/sim_nn_windows.csv").groupby("variant").loss_pct.mean(); sld = pd.read_csv("data/sim_ladder_nn_windows.csv").groupby("variant").loss_pct.mean()
LOSS = {"wx3_bf": 0.1105, "dyn3_bf": 0.1109, "p2": 0.1078, "p2c32a": 0.1074, "ecns": 0.1071, "ecnsnn": float(sn["ec_ns+nn|lgb"]), "final": float(sn["ec_ns+nn|lgb|geo"])}
SIM.update({k: v / S["R3"] for k, v in LOSS.items()})
E["table_recent"] = tab(R, list(NM), NM, SIM); E["table_all"] = tab(T, list(NM), NM); E["table_targets"] = tab(R[R.target], list(NM), NM)
FV = {"R3": "R3: last 3 months x 4", "sea_type": "+ type seasonality", "r_q_nn": "+ rules, quantile trees and NN (D132)", "p2": "+ weather, events, co-purchase neighbours, TabPFN v2 (D169)",
      "p2c32a": "+ arithmetic stack with conformal TabPFN (D231)", "ecnsnn": "+ new tree member and new NN (D253, D272)", "final": "+ segment and breakout corrections = submitted (D273)"}
E["fva"] = tab(R, list(FV), FV)
_w = {r["name"]: r["WAPE / R3"] for r in E["fva"]}
E["fva_note"] = (f"Type seasonality alone removes {1 - _w[FV['sea_type']]:.0%} of recency's error; the three-way system of D132 reaches {1 - _w[FV['r_q_nn']]:.0%}; "
                 f"weather, events, co-purchase neighbours and TabPFN add {(_w[FV['r_q_nn']] - _w[FV['p2']]) * 100:.1f} points; the arithmetic stack with the conformal member "
                 f"{(_w[FV['p2']] - _w[FV['p2c32a']]) * 100:.1f}; the rebuilt tree and NN {(_w[FV['p2c32a']] - _w[FV['ecnsnn']]) * 100:.1f}; the corrections {(_w[FV['ecnsnn']] - _w[FV['final']]) * 100:.1f}. "
                 f"The submitted system removes {1 - _w[FV['final']]:.0%} of recency's error on the nine recent origins.")
E["waterfall"] = J["system"]["cascade"]
wr = lambda Z, c: float((Z[c] - Z.y).abs().sum() / (Z.R3 - Z.y).abs().sum())
J["system"].update(recent=round(wr(R, "final"), 4), all13=round(wr(T, "final"), 4), targets=round(wr(R[R.target], "final"), 4), bf=round(wr(T[T.bf_window == 1], "final"), 4),
                   bias_recent=round(float(R.final.sum() / R.y.sum() - 1), 4), bias_all=round(float(T.final.sum() / T.y.sum() - 1), 4), rules_recent=round(wr(R, "rules"), 4),
                   loss_final=round(LOSS["final"], 4), loss_stack=round(LOSS["ecnsnn"], 4), loss_ladder=round(float(sld["u65ec+nn"]), 4), loss_ladder_geo=round(float(sld["u65ec+nn_geo"]), 4), loss_r3=round(float(S["R3"]), 4),
                   n_recent=int(len(R)), n_all=int(len(T)))
b = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred", "y"]].rename(columns={"pred": "lgb"})
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50"]].rename(columns={"q50": "nn"})
X = b.merge(n, on=["origin", "style_color", "h"]); RR = []
for o in X.origin.unique():
    r = pd.read_csv(f"agent_eval/{o}/predictions_verified.csv", dtype={"month": str}); r["h"] = [(pd.Period(m, "M") - pd.Period(o, "M")).n for m in r.month]; RR.append(r.assign(origin=o))
X = X.merge(pd.concat(RR)[["origin", "style_color", "h", "pred"]].rename(columns={"pred": "rules"}), on=["origin", "style_color", "h"])
X["geo"] = np.exp((np.log(X.lgb.clip(lower=.01)) + np.log(X.nn.clip(lower=.01)) + np.log(X.rules.clip(lower=.01))) / 3)
X["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(X.origin, X.h)]; g = X.groupby("tm")[["y", "lgb", "nn", "rules", "geo"]].sum()
E["month_bias"] = {"months": list(g.index), **{c: [round(float(v), 3) for v in (g[c] / g.y - 1)] for c in ["lgb", "nn", "rules", "geo"]}}
hm = R.groupby(["origin", "macro"]).apply(lambda x: x.final.sum() / x.y.sum() - 1).unstack()
E["origin_macro"] = {"origins": list(hm.index), "groups": list(hm.columns), "bias": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in hm.values]}
def _prep(src):
    A_ = src.copy(); A_["e"] = (A_.final - A_.y).abs(); A_["volq"] = A_.groupby("origin")["level"].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    A_["spq"] = A_.groupby("origin")["spread"].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False)); return A_, A_.e.sum()
A, te = _prep(R)
def arch(m, nm):
    x = A[m]; return dict(name=nm, items=round(float(m.mean()), 3), units=round(float(x.y.sum() / A.y.sum()), 3), error=round(float(x.e.sum() / te), 3),
                          bias=round(float(x.final.sum() / x.y.sum() - 1), 3), rel=round(float((x.final - x.y).abs().sum() / (x.R3 - x.y).abs().sum()), 3))
E["archetypes"] = [arch(A.bf_window == 1, "Window contains Black Friday"), arch(A.bf_window == 0, "Window without Black Friday"),
                   arch(A.age_m < 3, "Launch under 3 months"), arch((A.age_m >= 3) & (A.age_m < 6), "Age 3-6 months"), arch((A.age_m >= 6) & (A.age_m < 12), "Age 6-12 months"), arch(A.age_m >= 12, "Age 12 months or more"),
                   arch(A.volq == 0, "Smallest volume fifth"), arch(A.volq == 1, "Second volume fifth"), arch(A.volq == 4, "Largest volume fifth"),
                   arch(A.macro == "dress", "Dresses"), arch(A.macro == "warm", "Warm items (hoodies, sweaters, jackets)"), arch(A.spq == 4, "Members disagree most (top fifth of spread)")]
# D348 (user comment on the atlas: "lets separate this error analysis table to recent months and older months"): the same segments on the
# four early origins (Feb–May 2025), next to the nine recent ones; the notes below stay on the recent origins
_names = [a_["name"] for a_ in E["archetypes"]]; _A0, _te0 = A, te; A, te = _prep(T[~T.origin.isin(REC)])
_seg = [A.bf_window == 1, A.bf_window == 0, A.age_m < 3, (A.age_m >= 3) & (A.age_m < 6), (A.age_m >= 6) & (A.age_m < 12), A.age_m >= 12, A.volq == 0, A.volq == 1, A.volq == 4, A.macro == "dress", A.macro == "warm", A.spq == 4]
E["archetypes_old"] = [arch(m_, nm_) if m_.any() else dict(name=nm_, items=0.0, units=0.0, error=0.0, bias=None, rel=None) for m_, nm_ in zip(_seg, _names)]; A, te = _A0, _te0
E["arch_periods"] = dict(recent=f"{REC[0]} to {REC[-1]} ({len(REC)} origins)", old=f"{sorted(set(T.origin) - set(REC))[0]} to {sorted(set(T.origin) - set(REC))[-1]} ({len(set(T.origin) - set(REC))} origins)")
es = A.e.sort_values(ascending=False); cs = es.cumsum() / te
E["concentration"] = dict(half=round(float((cs < 0.5).mean()), 3), eighty=round(float((cs < 0.8).mean()), 3), top5=round(float(es.iloc[:int(round(0.05 * len(es)))].sum() / te), 3))
c_ = E["concentration"]; g_ = lambda nm: next(a for a in E["archetypes"] if a["name"] == nm)
E["notes"] = [
    f"<b>Concentration:</b> {c_['half']:.0%} of item-origins carry half of all absolute error and {c_['eighty']:.0%} carry 80%; the worst 5% carry {c_['top5']:.0%}.",
    f"<b>When:</b> windows that contain Black Friday hold {g_('Window contains Black Friday')['units']:.0%} of the units and {g_('Window contains Black Friday')['error']:.0%} of the error. By target month the monthly split over-forecasts September and under-forecasts November and January (D244).",
    f"<b>Who:</b> launches under 3 months hold {g_('Launch under 3 months')['error']:.0%} of the error on {g_('Launch under 3 months')['units']:.0%} of the units and are under-forecast by {-g_('Launch under 3 months')['bias']:.0%}; the largest volume fifth is under-forecast by {-g_('Largest volume fifth')['bias']:.0%}.",
    "<b>Shared misses:</b> in the large misses every member is wrong in the same direction (all under: 31% of all error; all over: 13%; measured on the five-member stack, D244), so re-weighting the members cannot fix them. Misses do not persist: an item missed now is no more likely to be missed four months later (rank correlation 0.06).",
    "<b>Breakouts:</b> the largest single failure is a young launch that breaks out into the autumn and holiday peak, growing 3-10x while the members extrapolate a gentle ramp. It is only weakly visible at the origin (AUC 0.66), but half of the breakouts show within their first month (D247, D248).",
    "<b>Hierarchy:</b> looks (items bought together, with similar photos and text) hold about twice the shared error of product types at an equal number of groups, yet no group-level model beat the sum of the item forecasts (D240, D250).",
    "<b>Buying:</b> buying loss follows bias more than WAPE. The arithmetic stack and an upper quantile (P65-P70) of the quantile members buy best; the corrections improve accuracy only (D230, D236, D273).",
    "<b>Limits of the backtest:</b> all 1,500 catalogue items were still selling after July 2025, so no backtest window contains an item that was discontinued (D254); and items of a style the models have never seen score 0.505 against 0.411 for seen styles."]
E["probes"] = [
    ["Time-honest combination weights (global, by group, by age)", "D115", "worse", "History favoured the rules; which member is right shifts over time. Equal weights stay."],
    ["LightGBM stacking of the members", "D115", "worse", "+0.006 to +0.012 WAPE; halved it is level."],
    ["LightGBM as a 4th member, medians, trimmed means, a recency member", "D115", "worse or level", "The equal geometric three-way stays best."],
    ["Segment bias correction (volume x age)", "D115", "level", "WAPE level, bias worse."],
    ["Last-year calibration of each month's level", "D124", "trade-off", "Bias -8% to 0% and targets better, but WAPE +0.002 to +0.010."],
    ["Occasion timing features (days to the item's occasion)", "D112", "small gain", "Quantile model -0.35 WAPE points; buying simulation 12.8% to 12.6% loss."],
    ["Discount momentum features", "D112", "worse", "Discounts are deep and constant; momentum is noise."],
    ["Matryoshka hierarchical embeddings as tree features", "D113", "worse; level when out-of-fold", "Leakage fixed by out-of-fold styles; no gain."],
    ["Binned conformal ranges from member disagreement", "D117", "better ranges", "Pinball -2.3%, interval score -6.5%, calibrated; not for safety stock (peak)."],
    ["Rules + NN monthly split, no own-shape mix", "D120-D121", "better split", "Shape error 0.240 to 0.177; fixes the July under-allocation at its root."],
    ["Year-on-year rule (D052) on the three-way", "D127", "level / better typical item", "Validated on the quantile model; kept."],
    ["Bottom-up per location (median, expected value)", "D125", "worse", "Channel dynamics (TikTok, new partners, store turnover) do not carry forward."],
    ["Analog launch curves for young items", "D129", "worse", "Early sales do not follow peers' curves closely enough."],
    ["Finest grain: item x location x month, summed", "D126", "worse", "Poisson noise in sparse location-months; errors do not cancel."],
    ["Retail rollout features (store coverage, new stores)", "D122b", "worse", "Store counts trend over time; trees learn relations that do not hold later."],
    ["Tabular foundation model (TabDPT, CPU, 2,048-row context)", "D130", "worse", "0.530 vs 0.471 WAPE / R3; TabPFN needs a licence sign-in (user's call)."],
    ["Partly recursive monthly trees", "D131", "better", "Monthly WAPE 0.406 to 0.386; totals -0.008 to -0.019; in the combination level recent, better on all origins."],
    ["Lean quantile model (drop lifecycle, channel, discount, occasion flags)", "D134", "better in the combination", "Same accuracy alone with 30% fewer features; three-way -0.002."],
    ["Lean model + occasion timing", "D135", "best quantile model", "Alone -0.005, three-way -0.003, buying loss 12.8% to 12.1%."],
    ["Conformal quantiles for the 2026 targets", "D133", "calibrated", "Leave-one-origin-out coverage 49% / 79% for P25-75 / P10-90."],
    ["November level from last year (event-month calibration)", "D136", "better", "Recent 0.4397 to 0.4350-0.4367; November bias -24% to -7%/-16%."],
    ["Black Friday item-lift model + allocation", "D137", "better", "Item lift predictable (r 0.42); restores the typical item; BF origins 0.577 to 0.567."],
    ["Product-group bias correction", "D138", "worse", "Group factors flip over time."],
    ["Pace extrapolation for young items", "D140", "worse", "Members damp young momentum, but raw pace is too noisy."],
    ["Training window / recency weighting", "D139", "need >= 12 origins; no down-weighting", "Last 6 origins 0.745; last 12 level; half-life weights worse."],
    ["Both winning fixes stacked (lean model + Black Friday fix)", "D141", "best system", "Recent 0.4397 to 0.4333; bias -8.2% to -4.5%; targets 0.428 to 0.414."],
    ["Forecast stability across origins", "D142", "NN steadiest", "Median revision: NN 17%, combination 20%, rules 26%."],
    ["Review list for the 2026 targets", "D143", "18 items to check", "Members more than 2x apart on 18 targets (83k units)."],
    ["Launch plan: new colour of the same style in the window", "D144", "halo, +10%", "Existing colours under-forecast ~20%; stable time-honest uplift 1.08-1.14."],
    ["Style-level features (style level, colours, share, momentum)", "D145", "level", "The item's own history already carries what its style knows."],
    ["June 2025 origin (deliverable analogue) checked against the 2026 candidate", "D146", "no added risk", "Halloween 7.56x June vs 7.02x in 2025; warm 8.37x vs 8.85x."],
    ["TabDPT with a 4,096-row context", "D130b", "better than 2,048, still worse", "0.515 vs 0.471 alone; level in the three-way."],
    ["Residual-distribution member (sampled residue around R3)", "D148", "level", "0.4349 sharing the quantile slot; no gain over the quantile model."],
    ["Learned per-style embedding", "D149", "level", "The item's own history already carries what a style vector adds."],
    ["Cart co-purchase neighbours + LLM fashion scores as tree features", "D153", "better", "Three-way 0.4326; with the Black Friday fix 0.4299, buying loss 11.37%."],
    ["Statewide temperature and a US events calendar as features", "D159", "better", "Three-way 0.4234; replacing seasonality outright is worse (0.4665 alone). Kept with seasonality."],
    ["Multi-view item clusters (carts, co-movement, descriptors, weather)", "D158", "level", "Coherent profiles, but no gain as features."],
    ["Descriptor embeddings must predict carts before use (cart gate)", "D160-D168", "gate passed", "Content embedding v4: held-out styles, rank correlation 0.431 vs 0.180 for raw product features."],
    ["Time-series foundation models (Chronos-2, Bolt, TimesFM)", "D192-D197", "worse", "0.63-0.76 alone; most items have under a year of history, so a univariate model cannot see a season."],
    ["Dynamic co-purchase network (Poisson, re-fitted per origin)", "D193-D203", "better", "Next-month deviance explained 0.50 vs 0.27 static; as tree features 0.4202 recent, 0.4634 all 13."],
    ["Explanation layer: SHAP of the deviation from R3, LLM categories, three judges, surrogates", "D191-D195", "faithful", "11 categories recover 72-85% of the lift; judges agree with SHAP (r 0.94)."],
    ["Other tabular foundation models (TabDPT, Mitra, Nori, TabICL)", "D201", "worse or conditional", "TabDPT 0.455 alone; the rest worse than R3 at a 3k context."],
    ["TabPFN v2 as an ensemble member", "D204", "better", "Fourth member: 0.4100 recent, 0.4586 all 13; licence allows production use. TabPFN 3.5 is non-commercial and not used."],
    ["Cluster-history, attribute-trend, trend-cycle and nearest-neighbour features", "D205", "worse or level", "39 features: 0.4427 vs 0.4315 alone. Residual signal exists but the trees already carry it."],
    ["CatBoost, XGBoost, DART, Tweedie on units, Optuna tuning, seed and window bagging", "D206-D207", "worse or level", "Default LightGBM parameters stay best; no extra member helps."],
    ["Conformal calibration of the TabPFN member", "D206", "better", "Pinball 0.131 to 0.121; in the stack 0.4027 recent."],
    ["Arithmetic mean instead of geometric mean of the members", "D230", "better for buying", "Geometric stacks forecast low; buying loss 10.78% to 10.68%."],
    ["Newsvendor ladder: quantile members at P60-P75", "D233-D236", "better for buying", "Buying loss bottoms at P65-P70 (10.38%); used for the buying column only."],
    ["Graph neural network style embedding", "D208", "level", "Passes the cart gate (0.284 vs 0.159) but level with the content embedding."],
    ["Hierarchical reconciliation, MinT, pooling, learned shares", "D172-D186", "worse or level", "Group totals cannot be forecast better than the sum of the items."],
    ["Planned campaign calendar as features", "D173", "worse", "With two seasons a calendar feature cannot be separated from the month."],
    ["Stock-out-corrected level and targets", "D174", "small gain alone", "0.4320 vs 0.4355 for the quantile model alone; level in the system."],
    ["FFORMA-style learned combination weights", "D175", "worse", "Which member is right shifts over time; equal weights stay."],
    ["Multi-horizon neural nets on weekly series (NHITS, TFT, DeepAR)", "D177-D234", "worse", "0.76 or worse alone; as a member every version hurts."],
    ["Training-data augmentation, weekly and mid-month origins", "D178-D184", "level", "Mid-month rows at half weight give the quantile model a small gain alone; weekly pipelines are far weaker."],
    ["Google Trends features; product photos and text (CLIP)", "D181-D183", "no gain / gate passed", "Trends add nothing; photos improve the content embedding on the cart gate (0.281 vs 0.273)."],
    ["Level decomposition: site, look, type, style, colour; hierarchical PCA", "D238-D250", "explains, does not forecast", "Knowing the type x look cell totals would cut error by a third, but no cell model beats the bottom-up sum."],
    ["Segment re-allocation (volume x age x discount), totals kept", "D245", "better", "0.3977 to 0.3918 recent; buying level."],
    ["Breakout model on weekly momentum and look-peer growth", "D247-D248", "better", "AUC 0.66 at the origin; with the segment correction 0.3902."],
    ["Age, 15 more event windows, type trends, launch curves, peaks for the tree member", "D252-D253", "events and curves better", "Member alone 0.4326 to 0.4282; static descriptors dropped."],
    ["Leakage, memorisation and shuffled-target checks", "D254", "clean", "Features match a rebuild from truncated data; style held out +1%; survivorship limitation found."],
    ["Low-volume model, Black Friday models, separate new-items model", "D253-D263", "level or worse", "The Black Friday error is breakouts every model misses, not a missing November feature."],
    ["Launch plan at look level (new colours, look-alike launches in the window)", "D264", "better, but needs Edikted's plan", "0.3933 vs 0.3977; cannot enter the deliverable without the launch calendar."],
    ["Chasing breakouts in the peak months (buying)", "D265", "worse", "With a 21-day lead time the stock arrives as the spike fades."],
    ["TabPFN v2 rebuilt on the new features", "D270", "level in the system", "Much better alone (0.4395 vs 0.4626) but less upward bias, which the stack needed."],
    ["Sales-curve NN rebuilt on the new features", "D272-D273", "better", "0.3961 to 0.3939 in the stack; with re-fitted corrections 0.3851 recent, 0.4399 all 13; buying level."],
]
ABL = {"seasonality": "Seasonality (like-for-like indices, own season, month)", "momentum": "Recent momentum (lags, pace, trend, peak)", "product": "Product type and attributes",
       "size": "Size (level)", "channel": "Channel mix", "discount": "Discount", "occasion": "Occasion flags", "lifecycle": "Lifecycle (age, days live, colour number)"}
qb = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]); key = pd.MultiIndex.from_arrays([R.origin, R.style_color])
e3 = (R.R3 - R.y).abs().sum(); base = (qb.q50.reindex(key).values - R.y.values).__abs__().sum() / e3
E["ablation"] = [dict(group="(none removed: production quantile model)", rel=round(float(base), 4), delta=0.0)]
for k, nm in ABL.items():
    x = pd.read_parquet(f"data/q_backtest_abl_{k}.parquet").set_index(["origin", "style_color"]).q50.reindex(key).values
    v = np.abs(x - R.y.values).sum() / e3; E["ablation"].append(dict(group=nm, rel=round(float(v), 4), delta=round(float(v - base), 4)))
for k, nm in [("lean", "Lean set (lifecycle, channel, discount, occasion flags removed)"), ("leanocct", "Lean set + occasion timing")]:
    x = pd.read_parquet(f"data/q_backtest_abl_{k}.parquet").set_index(["origin", "style_color"]).q50.reindex(key).values
    v = np.abs(x - R.y.values).sum() / e3; E["ablation"].append(dict(group=nm, rel=round(float(v), 4), delta=round(float(v - base), 4)))
def po(Z, c):
    out = {}
    for o, g in Z.groupby("origin"):
        e = (g[c] - g.y).abs(); out[o] = dict(wape=round(float(e.sum() / g.y.sum()), 3), bias=round(float(g[c].sum() / g.y.sum() - 1), 3), w20=round(float((e <= 0.2 * g.y).mean()), 3), w50=round(float((e <= 0.5 * g.y).mean()), 3), n=int(len(g)))
    return out
J["per_origin"] = {"Submitted system (five-member stack + corrections)": po(T, "final"), **{k: v for k, v in J["per_origin"].items() if not k.startswith("Submitted")}}
J["eval"] = E; json.dump(J, open("data/project_map.json", "w", encoding="utf-8"), ensure_ascii=False)
print("eval keys:", list(E), "| table rows:", len(E["table_recent"]), "| probes:", len(E["probes"]))
