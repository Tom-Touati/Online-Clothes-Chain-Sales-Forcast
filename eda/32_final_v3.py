"""Final forecast v3 (D050-D055). Order of steps:
1 Model answer = quantile P50 (season-matched zero-bias quantile, D050).
2 Items with a full year of own history (effective start before April 2025): 50/50 geometric blend of the model with
  YN = own Jul-Oct 2025 x (Apr-Jun 2026 + 5)/(Apr-Jun 2025 + 5) x aging factor (median actual/YN over all backtest origins with known actuals) (D052).
  Monthly shape = average of the model's monthly profile and the item's own Jul-Oct 2025 shape.
3 Segment rules (D051, narrowed by D053): swimwear capped at 1.82x June for all items; Homecoming / Halloween floors only for items WITHOUT a year of own history;
  those young segment items take the segment's 2024/2025 monthly shape.
4 Campaign uplift (D031), World Cup rule (D033), out-of-stock item (D034).
5 Confidence (D054): calibrated model range re-centred on the answer; 1 / 2 / 3 by the +-20% / +-50% band rule; 3 where a segment rule moved the total
  by more than 10%, for the World Cup items and for S16079."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120)
O = pd.read_csv("data/final_forecast_q.csv", index_col=0); W = pd.read_csv("data/final_forecast.csv", index_col=0); T = list(O.index)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); SW = ["Swimwear Top", "Swimwear Bottom"]
mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
def seg(idx):
    s = pd.Series("other", index=idx)
    s[occ.reindex(idx).str.contains("Homecoming").values] = "Homecoming"; s[occ.reindex(idx).str.contains("Halloween").values] = "Halloween"
    s[occ.reindex(idx).str.contains("4th of July").values] = "4th of July"; s[it.product_type.reindex(idx).isin(SW).values] = "swimwear"
    return s
# segment history (D051)
hist = {}
for y in (2024, 2025):
    live = start[start < pd.Timestamp(f"{y}-05-15")].index.intersection(M.index); live = live[M.loc[live, f"{y}-06"] >= 50]
    X = M.loc[live, mo(y, [6, 7, 8, 9, 10])].values; sg = seg(live).values
    for s in np.unique(sg):
        g = X[sg == s]; t = g.sum(axis=0); hist[(s, y)] = dict(ratio=np.median(g[:, 1:].sum(axis=1) / g[:, 0]), shape=t[1:] / t[1:].sum())
# D063: floors = 30th percentile of the previous year's (2025) item-level Jul-Oct/June ratio; level chosen on the June and July 2025 origins (eda/43_floor_level_two_origins.py)
live25 = start[start < pd.Timestamp("2025-05-15")].index.intersection(M.index); live25 = live25[M.loc[live25, "2025-06"] >= 50]
r25 = M.loc[live25, mo(2025, [7, 8, 9, 10])].sum(axis=1) / M.loc[live25, "2025-06"]; sg25 = seg(live25).values
RULE = {"swimwear": ("cap", max(hist[("swimwear", 2024)]["ratio"], hist[("swimwear", 2025)]["ratio"])),
        "Homecoming": ("floor", float(r25[sg25 == "Homecoming"].quantile(0.30))),
        "Halloween": ("floor", float(r25[sg25 == "Halloween"].quantile(0.30)))}
SHAPE = {s: (hist[(s, 2024)]["shape"] + hist[(s, 2025)]["shape"]) / 2 for s in RULE}
# aging factor (D052): median actual / YN over every backtest origin with known actuals, items with a full year of own history
Q = pd.read_parquet("data/q_backtest.parquet"); lr = []
for o, g in Q.groupby("origin"):
    k = cols.index(o)
    if k < 14: continue
    ids = g.style_color.values; old = (start.reindex(ids) < pd.Period(cols[k - 14]).start_time).values
    ly = M.loc[ids, cols[k - 11:k - 7]].sum(axis=1).values; yn = ly * (M.loc[ids, cols[k - 2:k + 1]].sum(axis=1).values + 5) / (M.loc[ids, cols[k - 14:k - 11]].sum(axis=1).values + 5)
    keep = old & (ly >= 50); lr += list(np.log((g.ytot.values[keep] + 5) / (yn[keep] + 5)))
AGE = float(np.exp(np.median(lr))); print(f"aging factor: {AGE:.3f} from {len(lr)} item-origins")

out = O[["q10", "q25", "q50", "q75", "q90", "p10c", "p25c", "p75c", "p90c", "level", "ptype", "age_m", "macro", "web_share", "b0_total"]].copy()
out["seg"] = seg(out.index).values; out["jun"] = M.loc[T, "2026-06"].values
sc0 = "S16079_DARK-GRAY"; WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
out["model_answer"] = out.q50; out.loc[sc0, "model_answer"] = W.loc[sc0, "raw_total"]; out["raw_total"] = out.model_answer
out["mature"] = (start.reindex(T) < pd.Timestamp("2025-04-01")).values & (M.loc[T, mo(2025, [7, 8, 9, 10])].sum(axis=1).values >= 50) & (~out.index.isin([sc0]))
ly = M.loc[T, mo(2025, [7, 8, 9, 10])]; out["yoy_naive"] = ly.sum(axis=1) * (M.loc[T, mo(2026, [4, 5, 6])].sum(axis=1) + 5) / (M.loc[T, mo(2025, [4, 5, 6])].sum(axis=1) + 5) * AGE
out.loc[~out.mature, "yoy_naive"] = np.nan   # defined only for items with a full year of own history
tot = out.model_answer.copy()
mm = out.mature; tot[mm] = np.exp(0.5 * np.log(out.model_answer[mm] + 1) + 0.5 * np.log(out.yoy_naive[mm] + 1)) - 1
out["after_yoy_blend"] = tot.copy()
for s, (kind, r) in RULE.items():
    m = (out.seg == s) & ((~out.mature) if kind == "floor" else True)
    tot[m] = np.minimum(tot[m], out.jun[m] * r) if kind == "cap" else np.maximum(tot[m], out.jun[m] * r)
out["after_segment_rule"] = tot
out["seg_rule_mult"] = out.seg.map({k: v[1] for k, v in RULE.items()})
prof = W[["jul", "aug", "sep", "oct"]].div(W[["jul", "aug", "sep", "oct"]].sum(axis=1), axis=0).loc[T]; prof.columns = ["jul", "aug", "sep", "oct"]
own = ly.div(ly.sum(axis=1).replace(0, np.nan), axis=0); own.columns = ["jul", "aug", "sep", "oct"]
shape = prof.copy(); shape[mm] = 0.5 * prof[mm] + 0.5 * own[mm]
for s in RULE:
    m = (out.seg == s) & (~out.mature); shape.loc[m, :] = SHAPE[s]
M4 = shape.mul(tot, axis=0)
M4["aug"] *= 1 + 0.07 * W.web_share.loc[T]; M4["sep"] *= 1 + 0.04 * W.web_share.loc[T]                       # D031
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd = dd[~dd.cancelled]; dd["ny_date"] = pd.to_datetime(dd.ny_date)
r14 = dd[dd.style_color.isin(WC) & (dd.ny_date >= "2026-06-17")].groupby("style_color").qty.sum() / 14
for s in WC:                                                                                                 # D033
    M4.loc[s, "jul"] = min(M4.loc[s, "jul"], r14[s] * (19 + 0.75 * 12)); j = M4.loc[s, "jul"]
    for mth, cap in [("aug", 0.75), ("sep", 0.50), ("oct", 0.45)]: M4.loc[s, mth] = min(M4.loc[s, mth], cap * j)
out[["f_jul", "f_aug", "f_sep", "f_oct"]] = M4.round(0).values
out["final_total"] = out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(axis=1); out["adj_pct"] = out.final_total / out.raw_total - 1
out["yoy_pct"] = out.after_yoy_blend / out.model_answer - 1; out["seg_rule_pct"] = out.after_segment_rule / out.after_yoy_blend - 1
# confidence (D054): calibrated range re-centred on the answer
Fa = out.after_segment_rule; sh = (Fa + 1) / (out.model_answer + 1)
p25 = (out.p25c + 1) * sh - 1; p75 = (out.p75c + 1) * sh - 1
band = np.where((p25 >= Fa / 1.2) & (p75 <= Fa / 0.8), 1, np.where((p25 >= Fa / 1.5) & (p75 <= Fa / 0.5), 2, 3))
out["p25_final"] = p25; out["p75_final"] = p75; out["conf"] = band
out["conf_reason"] = np.select([band == 1, band == 2], ["P25-P75 inside +/-20%", "P25-P75 inside +/-50%"], "P25-P75 wider than +/-50%")
dis = np.abs(np.log((out.model_answer + 1) / (out.yoy_naive + 1))); out["model_vs_trend"] = np.where(out.mature, dis, np.nan)   # D055
big = out.mature & (dis > np.log(2)); mid = out.mature & (dis > np.log(1.5)) & ~big
out.loc[mid & (out.conf == 1), "conf_reason"] = "capped at 2: model and own trend differ by more than 50%"; out.loc[mid, "conf"] = np.maximum(out.loc[mid, "conf"], 2)
out.loc[big, "conf"] = 3; out.loc[big, "conf_reason"] = "model and own trend differ by more than 2x"
judg = (out.seg_rule_pct.abs() > 0.10) | out.index.isin(WC + [sc0])
out.loc[judg, "conf"] = 3; out.loc[judg, "conf_reason"] = "number set by judgement"
out["iqr_ratio"] = out.p75c / out.p25c; out["answer"] = out.after_segment_rule; out["prev_final"] = O.final_total; out["prev_conf"] = O.conf
out.to_csv("data/final_forecast_q3.csv")
print("totals: model P50", int(out.raw_total.sum()), "| after YoY blend", int(out.after_yoy_blend.sum()), "| after segment rules", int(out.after_segment_rule.sum()), "| final", int(out.final_total.sum()), "| previous (q, P61)", int(O.final_total.sum()))
print("by month:", out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum().astype(int).to_dict(), "| mature items blended:", int(mm.sum()))
print("confidence:", out.conf.value_counts().sort_index().to_dict())
c = ["seg", "ptype", "age_m", "jun", "model_answer", "yoy_naive", "after_yoy_blend", "after_segment_rule", "final_total", "prev_final", "f_jul", "f_oct", "conf"]
print(out[out.mature].sort_values("yoy_pct")[c + ["yoy_pct"]].round(2).to_string())
print(out[(~out.mature) & (out.seg != "other")][c + ["seg_rule_pct"]].round(2).to_string())
