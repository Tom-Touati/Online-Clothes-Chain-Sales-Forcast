"""D248: (a) do the D245 segment re-allocation and the D247 breakout residual model stack? (b) how fast does a breakout show after the origin?
(a) Combined = F x (seg / F) x (res / F) for vol_age_disc_0.5n (D245) with res_all_0.5 (D247), renormalised to each origin's F total;
    also the geometric mean of the two corrected forecasts (renormalised). Standard table + bootstrap against F.
(b) Detection lag: for recent breakouts (e > 0.5, y >= 20) and for the rest, the ratio of actual to F's monthly forecast (F split by the
    LightGBM monthly shape, as D244) in target month 1, 2, 3; share of breakouts whose month-1 actual is already >= 1.5x its month-1
    forecast (false-alarm rate among non-breakouts); and the error share left if after month 1 the remaining 3 months were re-forecast
    by scaling with the month-1 ratio (capped at 3x, shrunk 0.5) -- an upper-bound view of a monthly re-forecast / chase-buy policy,
    not a 4-month forecast. Writes eda/stack_detect_output.txt (stdout)."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
B = pd.read_parquet("data/dec_base.parquet")
S = pd.read_parquet("data/bias_seg_preds.parquet")[["origin", "style_color", "vol_age_disc_0.5n"]]
K = pd.read_parquet("data/breakout_preds.parquet")[["origin", "style_color", "res_all_0.5", "res_all_0.5n", "brk"]]
B = B.merge(S, on=["origin", "style_color"]).merge(K, on=["origin", "style_color"])
tot = lambda c: B[c] * B.groupby("origin").F.transform("sum") / B.groupby("origin")[c].transform("sum")
B["comb"] = B.F * (B["vol_age_disc_0.5n"] / B.F.clip(lower=1e-6)) * (B["res_all_0.5"] / B.F.clip(lower=1e-6)); B["comb_n"] = tot("comb")
B["geo"] = np.sqrt(B["vol_age_disc_0.5n"].clip(lower=0) * B["res_all_0.5n"].clip(lower=0)); B["geo_n"] = tot("geo")
COLS = ["F", "vol_age_disc_0.5n", "res_all_0.5n", "comb_n", "geo_n"]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); Bi = B.set_index(["origin", "style_color"]).reindex(key)
for c in COLS: T[c] = Bi[c].values
REC_ = [o for o in ORIGINS if o >= "2025-06"]
print("== (a) stacking the segment re-allocation with the breakout residual model")
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("bootstrap vs F, recent:"); [print(f"  {c:18s}", np.round(sl.boot(F_, c, "F"), 4)) for c in COLS[1:]]
print("bootstrap comb_n vs vol_age_disc_0.5n:", np.round(sl.boot(F_, "comb_n", "vol_age_disc_0.5n"), 4))
# (b) detection lag
BT = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]]
BT["sh"] = BT.pred / BT.groupby(["origin", "style_color"]).pred.transform("sum").clip(lower=1e-9)
Mo = pd.read_parquet("data/monthly_item.parquet"); Mo.columns = [str(c) for c in Mo.columns]
mm = BT.merge(B[["origin", "style_color", "F", "y", "e", "brk", "age_m"]], on=["origin", "style_color"]); mm = mm[mm.origin >= "2025-06"]
mm["Fm"] = mm.F * mm.sh; mm["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(mm.origin, mm.h)]
mm["ym"] = [Mo.at[s, t] if t in Mo.columns and s in Mo.index else np.nan for s, t in zip(mm.style_color, mm.tm)]
W = mm.pivot_table(index=["origin", "style_color", "brk", "y", "F", "age_m"], columns="h", values=["Fm", "ym"]).dropna().reset_index()
r = lambda h: (W[("ym", h)] + 1) / (W[("Fm", h)] + 1)
print("\n== (b) detection: actual / forecast by target month (median), recent")
for b, g in W.groupby(W.brk):
    ix = g.index; print(f"  {'breakouts' if b else 'others':10s} n {len(g):5d} | month 1 {r(1)[ix].median():.2f} | month 2 {r(2)[ix].median():.2f} | month 3 {r(3)[ix].median():.2f} | month 4 {r(4)[ix].median():.2f}")
for thr in (1.3, 1.5, 2.0):
    flag = r(1) >= thr; br = W.brk == 1
    print(f"  month-1 ratio >= {thr}: catches {flag[br].mean():.1%} of breakouts ({W.loc[br & flag, 'y'].sum() / W.loc[br, 'y'].sum():.1%} of their units) | flags {flag[~br].mean():.1%} of the others | precision {(flag & br).sum() / max(flag.sum(), 1):.2f}")
young = W.age_m < 6
print(f"  young < 6 months: breakouts caught at month-1 ratio >= 1.5: {(r(1)[young & (W.brk == 1)] >= 1.5).mean():.1%}")
# upper-bound re-forecast after month 1
k = np.clip(r(1), 1 / 3, 3) ** 0.5; rest = sum(W[("Fm", h)] for h in (2, 3, 4)); yr = sum(W[("ym", h)] for h in (2, 3, 4))
e0 = (rest - yr).abs().sum(); e1 = (rest * k - yr).abs().sum()
print(f"  months 2-4: WAPE of F's split {e0 / yr.sum():.3f} -> scaled by the month-1 ratio^0.5 {e1 / yr.sum():.3f} (re-forecast one month in; not comparable to the 4-month backtest)")
