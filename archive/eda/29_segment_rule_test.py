"""Out-of-sample test of segment rules at the June 2025 origin, using only 2024 history (the information available then).
Rule: swimwear Jul-Oct/June capped at the 2024 item-median ratio; Homecoming and Halloween floored at their 2024 item-median ratio. Answer = P50.
Also: IQR coverage (k = 1.3) at the summer origins."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest.parquet"); M = pd.read_parquet("data/monthly_item.parquet"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); SW = ["Swimwear Top", "Swimwear Bottom"]
def seg(idx):
    s = pd.Series("other", index=idx)
    s[occ.reindex(idx).str.contains("Homecoming").values] = "Homecoming"; s[occ.reindex(idx).str.contains("Halloween").values] = "Halloween"
    s[occ.reindex(idx).str.contains("4th of July").values] = "4th of July"; s[it.product_type.reindex(idx).isin(SW).values] = "swimwear"
    return s
def ratios(y):
    live = start[start < pd.Timestamp(f"{y}-05-15")].index.intersection(M.index); live = live[M.loc[live, f"{y}-06"] >= 50]
    r = M.loc[live, [f"{y}-{m:02d}" for m in (7, 8, 9, 10)]].sum(axis=1) / M.loc[live, f"{y}-06"]
    return r.groupby(seg(live).values).median()
r24 = ratios(2024); print("2024 item-median Jul-Oct/June by segment:", r24.round(2).to_dict())
g = Q[Q.origin == "2025-06"].copy(); g["seg"] = seg(g.style_color.values).values; g["jun"] = M.loc[g.style_color, "2025-06"].values
g["p50"] = g.q50; g["rule"] = g.p50
cap = g.seg == "swimwear"; g.loc[cap, "rule"] = np.minimum(g.loc[cap, "p50"], g.loc[cap, "jun"] * r24["swimwear"])
for s in ["Homecoming", "Halloween"]:
    fl = g.seg == s; g.loc[fl, "rule"] = np.maximum(g.loc[fl, "p50"], g.loc[fl, "jun"] * r24[s])
def met(d, col):
    ape = (d[col] - d.ytot).abs() / d.ytot.clip(lower=1)
    return pd.Series(dict(n=len(d), bias=d[col].sum() / d.ytot.sum() - 1, med_ratio=(d[col] / d.ytot.clip(lower=1)).median(), w20=(ape <= .2).mean(), w50=(ape <= .5).mean(), wape=(d[col] - d.ytot).abs().sum() / d.ytot.sum()))
out = []
for s, d in list(g.groupby("seg")) + [("ALL", g)]:
    out.append(pd.concat([met(d, "p50").add_prefix("P50_"), met(d, "rule").add_prefix("rule_")]).rename(s))
print("\nJune 2025 origin, P50 vs P50 + 2024 segment rule:"); print(pd.DataFrame(out).round(3).drop(columns=["rule_n"]).to_string())
for o in ["2025-06", "2025-07"]:
    d = Q[Q.origin == o]; L = np.log1p(d[QC].values); med = L[:, 3:4]; Ls = med + 1.3 * (L - med)
    la = np.log1p(d.ytot.values); print(o, "coverage at k=1.3: P25-P75", round(((la >= Ls[:, 1]) & (la <= Ls[:, 5])).mean(), 3), "| P10-P90", round(((la >= Ls[:, 0]) & (la <= Ls[:, 6])).mean(), 3))
