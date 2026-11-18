"""Waterfall from R3 to the final Jul-Oct 2026 number (D118), per target item and for the portfolio, for the rules + quantile + NN
candidate (data/cand_rnn_final_forecast_q4.csv). Steps (multiplicative factors, applied in this order):
  R3                      4 x mean monthly units Apr-Jun 2026
  recent-rate anchor      4 x level (last 91 days' daily rate x 30.44): the quantile model predicts relative to this
  quantile model drivers  TreeSHAP of the median models (both feature sets, refitted exactly as eda/22's June 2026 fit, averaged), in log
                          ratio to the anchor, grouped: base (average uplift), seasonality, recent momentum, lifecycle, product & attributes,
                          occasion, channel mix, discount, size; the exponentiated contributions are applied in that order
  model reconciliation    the refit's median vs the stored quantile P50 (rounding of the sorted quantiles; should be ~1)
  rules + NN pull         three-way centre / quantile P50
  year-on-year blend      after_yoy_blend / centre (items with a year of own history)
  segment rules           swimwear cap, Homecoming / Halloween floors
  final adjustments       campaign uplift, World Cup, out-of-stock item, monthly rounding
Writes data/waterfall_2026.csv and eda/waterfall_2026_output.txt (via stdout)."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings, importlib.util
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
spec = importlib.util.spec_from_file_location("q20", "eda/20_quantiles.py"); q20 = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(q20); sys.argv = _a
T1 = q20.T1; O = pd.Period("2026-06").ordinal
tr = T1[(T1.o_ord + 4 <= O) & T1.ytot.notna()]; te = T1[T1.origin == "2026-06"].set_index("style_color")
GROUPS = {"seasonality": lambda c: c in ("own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "tmonth"),
          "recent momentum": lambda c: c.startswith("lagr") or c in ("f_r7", "f_r28", "trend91", "peak_r"),
          "lifecycle": lambda c: c in ("age_m", "days_live", "color_num"),
          "product & attributes": lambda c: c in ("ptype",) or c.startswith("a_"),
          "occasion": lambda c: c.startswith("occ_"), "channel mix": lambda c: c in ("sh_ret", "sh_drop", "sh_tt"),
          "discount": lambda c: c in ("disc_item", "disc_rel"), "size": lambda c: c == "lvl"}
def grp(c):
    for g, f in GROUPS.items():
        if f(c): return g
    raise KeyError(c)
SH = pd.DataFrame(0.0, index=te.index, columns=["base"] + list(GROUPS)); zhat = np.zeros(len(te))
for cols in (q20.BASE, q20.FULL):
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **q20.P).fit(tr[cols], tr.z)
    C = m.predict(te[cols], pred_contrib=True); zhat += m.predict(te[cols]) / 2
    SH["base"] += C[:, -1] / 2
    for j, c in enumerate(cols): SH[grp(c)] += C[:, j] / 2
assert np.allclose(SH.sum(axis=1).values, zhat, atol=1e-6)
S = pd.read_csv("data/final_forecast_q4.csv", index_col=0); Cd = pd.read_csv("data/cand_rnn_final_forecast_q4.csv", index_col=0)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
idx = Cd.index; W = pd.DataFrame(index=idx)
W["R3"] = 4 * M.loc[idx, ["2026-04", "2026-05", "2026-06"]].mean(axis=1)
lv = 4 * te.level.reindex(idx) + 1
q_refit = lv * np.exp(pd.Series(zhat, index=te.index).reindex(idx)) - 1
steps = [("R3", W.R3)]; cur = (4 * te.level.reindex(idx)).fillna(W.R3); steps.append(("recent-rate anchor (4 x level)", cur.copy()))
for g in ["base"] + list(GROUPS):
    cur = (cur + 1) * np.exp(SH[g].reindex(idx).fillna(0)) - 1; steps.append((f"model: {'average uplift' if g == 'base' else g}", cur.copy()))
steps.append(("model reconciliation (stored quantile P50)", S.q50.reindex(idx)))
steps.append(("rules + NN pull (three-way centre)", Cd.q50))
steps.append(("year-on-year blend", Cd.after_yoy_blend)); steps.append(("segment rules (caps, floors)", Cd.after_segment_rule))
steps.append(("final adjustments (campaign, World Cup, out-of-stock, rounding)", Cd.final_total))
oos = ~te.index.reindex(idx)[1].astype(bool) if False else ~idx.isin(te.index)
for nm, s in steps: W[nm] = s.values
print(f"targets: {len(idx)} | not in the June 2026 model set (handled by rule): {list(idx[oos])}")
print("refit median vs stored quantile P50 (targets in the model set): median ratio", round(float(np.median((q_refit[~oos] + 1) / (S.q50.reindex(idx)[~oos] + 1))), 4))
print("\n== portfolio waterfall (100 targets, units)")
prev = None; rows = []
for nm, _ in steps:
    tot = W[nm].sum(); rows.append({"step": nm, "total after step": round(tot), "change": None if prev is None else round(tot - prev), "factor": None if prev is None else round(tot / prev, 3)}); prev = tot
P = pd.DataFrame(rows); print(P.to_string(index=False))
print("\n== where the model's change from the anchor comes from, per item (median factor and share of items moved up)")
for g in ["base"] + list(GROUPS):
    f = np.exp(SH[g].reindex(idx[~oos])); print(f"  {g:22s} median x{f.median():.3f} | IQR x{f.quantile(.25):.3f}-x{f.quantile(.75):.3f} | up for {(f > 1).mean():.0%} of targets")
print("\n== five example items (units after each step)")
ex = ["S17159_BLACK-WASHED", "ACC1070-1_MIX", "S18184_LIGHT-BLUE", "S24840_LIGHT-GRAY-MELANGE", "S11233_BLACK"]
print(W.loc[[e for e in ex if e in W.index]].T.round(0).to_string())
W.round(1).to_csv("data/waterfall_2026.csv"); SH.round(4).to_csv("data/waterfall_2026_shap_log.csv")
