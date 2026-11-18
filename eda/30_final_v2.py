"""Final forecast v2 after the common-sense audit (D050-D053).
- Answer = P50: the zero-bias quantile on the season-matched origins (Jun and Jul 2025); 0.61 came from Black Friday windows.
- Segment rules from two consistent years (2024, 2025) of Jul-Oct / June ratios: swimwear capped at the higher year's item median,
  Homecoming and Halloween floored at the lower year's item median. These three segments take their average 2024/2025 monthly shape.
- Then the unchanged judgement steps: campaign uplift (D031), World Cup rule (D033), out-of-stock item (D034).
- Confidence: IQR band rule (D046) on the model answer; 3 where judgement moved the total by more than 10%, World Cup items and S16079."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 240); pd.set_option("display.max_rows", 120)
O = pd.read_csv("data/final_forecast_q.csv", index_col=0); W = pd.read_csv("data/final_forecast.csv", index_col=0); T = list(O.index)
M = pd.read_parquet("data/monthly_item.parquet"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); SW = ["Swimwear Top", "Swimwear Bottom"]
def seg(idx):
    s = pd.Series("other", index=idx)
    s[occ.reindex(idx).str.contains("Homecoming").values] = "Homecoming"; s[occ.reindex(idx).str.contains("Halloween").values] = "Halloween"
    s[occ.reindex(idx).str.contains("4th of July").values] = "4th of July"; s[it.product_type.reindex(idx).isin(SW).values] = "swimwear"
    return s
hist = {}
for y in (2024, 2025):
    live = start[start < pd.Timestamp(f"{y}-05-15")].index.intersection(M.index); live = live[M.loc[live, f"{y}-06"] >= 50]
    X = M.loc[live, [f"{y}-{m:02d}" for m in (6, 7, 8, 9, 10)]].values; sg = seg(live).values
    for s in np.unique(sg):
        g = X[sg == s]; t = g.sum(axis=0)
        hist[(s, y)] = dict(ratio=np.median(g[:, 1:].sum(axis=1) / g[:, 0]), shape=t[1:] / t[1:].sum(), n=len(g))
RULE = {"swimwear": ("cap", max(hist[("swimwear", 2024)]["ratio"], hist[("swimwear", 2025)]["ratio"])),
        "Homecoming": ("floor", min(hist[("Homecoming", 2024)]["ratio"], hist[("Homecoming", 2025)]["ratio"])),
        "Halloween": ("floor", min(hist[("Halloween", 2024)]["ratio"], hist[("Halloween", 2025)]["ratio"]))}
SHAPE = {s: (hist[(s, 2024)]["shape"] + hist[(s, 2025)]["shape"]) / 2 for s in RULE}
print("rules:", {s: (k, round(v, 2)) for s, (k, v) in RULE.items()}); print("segment shapes Jul-Oct:", {s: np.round(v, 3).tolist() for s, v in SHAPE.items()})

out = O[["q10", "q25", "q50", "q75", "q90", "p10c", "p25c", "p75c", "p90c", "level", "ptype", "age_m", "macro", "web_share", "b0_total"]].copy()
out["seg"] = seg(out.index).values; out["jun"] = M.loc[T, "2026-06"].values
out["model_answer"] = out.q50; sc0 = "S16079_DARK-GRAY"; out.loc[sc0, "model_answer"] = W.loc[sc0, "raw_total"]
out["raw_total"] = out.model_answer                                                # column N: model answer before any judgement
tot = out.model_answer.copy()
for s, (kind, r) in RULE.items():
    m = out.seg == s
    tot[m] = np.minimum(tot[m], out.jun[m] * r) if kind == "cap" else np.maximum(tot[m], out.jun[m] * r)
out["after_segment_rule"] = tot
prof = W[["jul", "aug", "sep", "oct"]].div(W[["jul", "aug", "sep", "oct"]].sum(axis=1), axis=0)
M4 = prof.mul(tot, axis=0); M4.columns = ["jul", "aug", "sep", "oct"]
for s in RULE:
    m = out.seg == s; M4.loc[m, :] = np.outer(tot[m], SHAPE[s])
M4["aug"] *= 1 + 0.07 * W.web_share; M4["sep"] *= 1 + 0.04 * W.web_share                # D031
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd = dd[~dd.cancelled]; dd["ny_date"] = pd.to_datetime(dd.ny_date)
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
r14 = dd[dd.style_color.isin(WC) & (dd.ny_date >= "2026-06-17")].groupby("style_color").qty.sum() / 14
for s in WC:                                                                         # D033
    M4.loc[s, "jul"] = min(M4.loc[s, "jul"], r14[s] * (19 + 0.75 * 12)); j = M4.loc[s, "jul"]
    for mth, cap in [("aug", 0.75), ("sep", 0.50), ("oct", 0.45)]: M4.loc[s, mth] = min(M4.loc[s, mth], cap * j)
out[["f_jul", "f_aug", "f_sep", "f_oct"]] = M4.round(0).values
out["final_total"] = out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(axis=1); out["adj_pct"] = out.final_total / out.raw_total - 1
out["seg_rule_pct"] = out.after_segment_rule / out.model_answer - 1
F = out.model_answer
band = np.where((out.p25c >= F / 1.2) & (out.p75c <= F / 0.8), 1, np.where((out.p25c >= F / 1.5) & (out.p75c <= F / 0.5), 2, 3))
out["conf_iqr"] = band; out["conf"] = band
out["conf_reason"] = np.select([band == 1, band == 2], ["P25-P75 inside +/-20%", "P25-P75 inside +/-50%"], "P25-P75 wider than +/-50%")
judg = (out.seg_rule_pct.abs() > 0.10) | out.index.isin(WC + [sc0])
out.loc[judg, "conf"] = 3; out.loc[judg, "conf_reason"] = "number set by judgement, not the model range"
out["iqr_ratio"] = out.p75c / out.p25c; out["prev_final"] = O.final_total; out["prev_conf"] = O.conf; out["answer"] = out.model_answer
out.to_csv("data/final_forecast_q2.csv")
print("\ntotals: model answer (P50)", int(out.raw_total.sum()), "| after segment rules", int(out.after_segment_rule.sum()), "| final", int(out.final_total.sum()), "| previous", int(O.final_total.sum()))
print("by month:", out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum().astype(int).to_dict())
print("confidence:", out.conf.value_counts().sort_index().to_dict(), "| IQR band only:", pd.Series(band).value_counts().sort_index().to_dict())
cols = ["seg", "ptype", "age_m", "jun", "model_answer", "after_segment_rule", "seg_rule_pct", "final_total", "prev_final", "f_jul", "f_aug", "f_sep", "f_oct", "conf", "prev_conf"]
print(out[out.seg != "other"].sort_values(["seg", "final_total"])[cols].round(2).to_string())
ma = out.age_m >= 12; print("mature items: final / (4 x level) =", round(out[ma & out.level.notna()].final_total.sum() / (4 * out[ma & out.level.notna()].level.sum()), 3))
print("other segment Jul-Oct / June:", round(out[out.seg == "other"].final_total.sum() / out[out.seg == "other"].jun.sum(), 2), "| item median", round((out[out.seg == "other"].final_total / out[out.seg == "other"].jun).median(), 2))
