"""D307 deliverable fit (fork; Main's request: "If one wins, I'll need its 2026-06 fit: the same eda/198t refit with the inputs at 2026-06,
training on o' + 4 <= 2026-06"). Builds the member table and the SIG / LM inputs exactly as eda/198t (its code up to the backtest loop is
executed unchanged), then fits Main's 'ec_ns' member at the June 2026 origin on every row with a known outcome (origins up to 2026-02):
  check  ec_ns without new inputs: must reproduce Main's stored data/q_backtest_nx_ec_ns_2026.parquet
  sig    + the 11 origin-known signals;  lm  + the 20 last-month inputs;  siglm  + both
Writes data/fk_member_refit_2026.parquet (origin, style_color, level, check, sig, lm, siglm: 4-month medians in units)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/198t_member_refit.py", encoding="utf-8").read(); cut = src.index("# 4 refit, time-honest"); g = {"__name__": "refit2026"}
exec(compile(src[:cut], "eda/198t_member_refit.py", "exec"), g)
T, sets, fit, SIG, LM = g["T"], g["sets"], g["fit"], g["SIG"], g["LM"]; g["ns"]["P"]["n_jobs"] = 4
o = "2026-06"; oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]
assert (tr.o_ord + 4 <= oo).all() and tr.origin.max() == "2026-02" and len(te) > 0, (tr.origin.max(), len(te))
print("training rows", len(tr), "origins", tr.origin.min(), "to", tr.origin.max(), "| June 2026 items", len(te), "| missing SIG", round(float(te[SIG].isna().mean().mean()), 3), "LM", round(float(te[LM].isna().mean().mean()), 3), flush=True)
B0, F0_ = sets("ec_ns"); out = te[["origin", "style_color", "level"]].copy()
for nm, add in [("check", []), ("sig", SIG), ("lm", LM), ("siglm", SIG + LM)]:
    z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out[nm] = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None); print(nm, "done", round(float(out[nm].sum())), flush=True)
st = pd.read_parquet("data/q_backtest_nx_ec_ns_2026.parquet").set_index("style_color").q50.reindex(out.style_color).values
print("check against Main's stored 2026 member: max relative difference", float(np.nanmax(np.abs(out.check.values - st) / np.maximum(st, 1))), "| items compared", int(np.isfinite(st).sum()))
W = pd.read_excel("Edikted_Forecasting_Exercise_filled.xlsx", sheet_name="Forecast").iloc[:, 1].dropna().astype(str).tolist(); tg = out.style_color.isin(W)
print("100 targets (99 modelled):", int(tg.sum()), "| sums:", {c: round(float(out.loc[tg, c].sum())) for c in ["check", "sig", "lm", "siglm"]})
out.to_parquet("data/fk_member_refit_2026.parquet"); print("saved")
