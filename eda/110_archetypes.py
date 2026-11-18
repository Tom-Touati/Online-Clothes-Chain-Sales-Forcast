"""Error archetypes seen in the galleries (D123), quantified over all item-origins (three-way combination, 4-month totals).
  spike  = transient spike at the origin: max weekly units / median weekly units over the 8 weeks before the origin (weeks the item was live)
  bfwin  = the forecast window contains Black Friday / Cyber week (origins Jul-Oct)
  bflift = realised Black Friday lift: units in the Thanksgiving-to-Cyber-Monday week / mean weekly units of the 4 weeks before it
           (for bfwin rows; outcome information, used to describe the error, not as a predictor)
  bflift_ly = the same item's lift a year earlier (known at the origin, if it existed); type_bf_ly = the median lift of its product type a year earlier
For each archetype: share of item-origins, share of units, share of absolute error, bias and WAPE / R3 of the three-way and members."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/error_map_items.parquet")
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
D = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(0).asfreq("D").fillna(0)          # dates x items
first = (D > 0).idxmax()
BF = {2024: pd.Timestamp("2024-11-28"), 2025: pd.Timestamp("2025-11-27")}                       # Thanksgiving; the week runs Thu..Mon (5 days)
def lift(s, y):
    t = BF[y]; wk = D.loc[t:t + pd.Timedelta(days=4), s].sum() / 5 * 7; base = D.loc[t - pd.Timedelta(days=28):t - pd.Timedelta(days=1), s].mean() * 7
    return wk / base if base > 0 else np.nan
rows = []
for o, s in zip(T.origin, T.style_color):
    E = pd.Period(o, "M").end_time.normalize(); w = D.loc[E - pd.Timedelta(days=55):E, s] if s in D else pd.Series(dtype=float)
    w = w[w.index >= first.get(s, E)]
    wk = w.groupby((w.index - w.index[0]).days // 7).sum() if len(w) else pd.Series(dtype=float)
    wk = wk[wk.index < len(w) // 7] if len(w) >= 7 else wk
    rows.append(wk.max() / max(wk.median(), 1) if len(wk) >= 2 else np.nan)
T["spike"] = rows
T["bfwin"] = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10", "2024-07", "2024-08", "2024-09", "2024-10"])
T["bflift"] = [lift(s, 2025) if (b and s in D) else np.nan for s, b in zip(T.style_color, T.bfwin)]
T["bflift_ly"] = [lift(s, 2024) if (b and s in D and first[s] < BF[2024] - pd.Timedelta(days=28)) else np.nan for s, b in zip(T.style_color, T.bfwin)]
tl = T[T.bfwin].groupby("ptype").bflift_ly.median(); T["type_bf_ly"] = T.ptype.map(tl)
T["e"] = T.r_q_nn - T.y; R = T[T.origin >= "2025-06"].copy(); tot_err = R.e.abs().sum()
def summ(mask, nm):
    x = R[mask]; e3 = (x.R3 - x.y).abs().sum()
    r = {"archetype": nm, "item-origins %": mask.mean() * 100, "units %": x.y.sum() / R.y.sum() * 100, "abs error %": x.e.abs().sum() / tot_err * 100,
         "bias 3-way": x.r_q_nn.sum() / x.y.sum() - 1}
    for c in ["rules", "qmodel", "nn", "r_q_nn"]: r[f"{c} WAPE/R3"] = (x[c] - x.y).abs().sum() / e3
    return r
R["young"] = R.age_m < 3
A = [summ(R.spike >= 3, "spike at origin (max/median week >= 3)"), summ(R.spike.between(2, 3), "moderate spike (2-3)"), summ(R.spike < 2, "no spike (< 2)"),
     summ(R.bfwin, "window has Black Friday"), summ(R.bfwin & (R.bflift >= 4), "  ... and the item spiked >= 4x on BF"), summ(R.bfwin & (R.bflift < 4), "  ... item spiked < 4x on BF"),
     summ(~R.bfwin, "window without Black Friday"), summ(R.young, "launch < 3 months"), summ(R.young & R.bfwin, "launch < 3 months, BF window"),
     summ((R.spike >= 3) & ~R.young, "spike at origin, item >= 3 months")]
print(pd.DataFrame(A).set_index("archetype").round(3).to_string())
print("\n== over-forecast by spike ratio (all recent item-origins): bias of the 3-way and median log error")
R["sb"] = pd.cut(R.spike, [0, 1.5, 2, 3, 5, 1e9], labels=["<1.5", "1.5-2", "2-3", "3-5", ">=5"])
print(R.groupby("sb").apply(lambda x: pd.Series({"n": len(x), "bias": x.r_q_nn.sum() / x.y.sum() - 1, "med log err": np.median(np.log((x.r_q_nn + 1) / (x.y + 1))),
                                                  "rules bias": x.rules.sum() / x.y.sum() - 1, "q bias": x.qmodel.sum() / x.y.sum() - 1, "nn bias": x.nn.sum() / x.y.sum() - 1})).round(3).to_string())
print("\n== Black Friday windows: realised item lift vs what was knowable")
B = R[R.bfwin & R.bflift.notna()]
print("realised BF lift: median", round(B.bflift.median(), 2), "| by macro:", B.groupby("macro").bflift.median().round(2).to_dict())
print("by age band:", B.groupby("age").bflift.median().round(2).to_dict())
b2 = B[B.bflift_ly.notna()]; print(f"items with a lift last year: {len(b2)} | corr(log lift, log lift last year) {np.corrcoef(np.log(b2.bflift.clip(lower=.1)), np.log(b2.bflift_ly.clip(lower=.1)))[0, 1]:.2f} | median ratio this/last {np.median(b2.bflift / b2.bflift_ly):.2f}")
b3 = B[B.type_bf_ly.notna()]; print(f"corr(log lift, log type lift last year) {np.corrcoef(np.log(b3.bflift.clip(lower=.1)), np.log(b3.type_bf_ly.clip(lower=.1)))[0, 1]:.2f}")
print("3-way bias in BF windows by realised lift band:", B.groupby(pd.cut(B.bflift, [0, 2, 4, 8, 1e9])).apply(lambda x: round(x.r_q_nn.sum() / x.y.sum() - 1, 3)).to_dict())
T.to_parquet("data/archetype_items.parquet", index=False)
