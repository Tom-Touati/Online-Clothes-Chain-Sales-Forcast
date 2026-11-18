"""Closes the three open common-sense questions with direct tests.
1 Items 3-12 months old at a June origin: does a peer-ratio forecast (June x the prior year's peer Jul-Oct/June ratio for items of the same age
  and peak position) beat the model? Tested at the June 2025 origin with 2024 peers only. Blends included.
2 Mature items: like-for-like reconciliation of forecast / last year against spring trend x ageing, same items, same statistic;
  and the bias of trend x 0.843 at the matching origins.
3 Floors: where each floored number sits inside the model's own calibrated range; young-item backtest of the floor at the June 2025 origin."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
O = pd.read_csv("data/final_forecast_q3.csv", index_col=0); Q = pd.read_parquet("data/q_backtest.parquet")
def met(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float); ape = np.abs(f - a) / np.clip(a, 1, None)
    return dict(n=len(a), bias=round(f.sum() / a.sum() - 1, 3), wape=round(np.abs(f - a).sum() / a.sum(), 3), w20=round((ape <= .2).mean(), 3), w50=round((ape <= .5).mean(), 3))
def pos_bucket(ids, y):
    k = cols.index(f"{y}-06"); pk = M.loc[ids, cols[:k + 1]].max(axis=1); r = M.loc[ids, f"{y}-06"] / pk
    return pd.cut(r, [0, .35, .7, 10], labels=["<35%", "35-70%", ">=70%"])

print("== 1  items 3-12 months old at the June origin: peer-ratio vs model")
def young(y):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(M.index)).dt.days / 30.44
    return M.index[((age >= 3) & (age < 12)).values & (M[f"{y}-06"] >= 50).values]
p24 = young(2024); b24 = pos_bucket(p24, 2024)
peer = (M.loc[p24, mo(2024, [7, 8, 9, 10])].sum(axis=1) / M.loc[p24, "2024-06"]).groupby(b24.values).median()
print("2024 peer item-median Jul-Oct/June by peak position:", peer.round(2).to_dict(), "| n:", b24.value_counts().to_dict())
g = Q[Q.origin == "2025-06"].set_index("style_color"); ids = [s for s in young(2025) if s in g.index]
b25 = pos_bucket(pd.Index(ids), 2025)
pr = M.loc[ids, "2025-06"] * peer.reindex(b25.values).values
G = pd.DataFrame({"model": g.loc[ids, "q50"].values, "peer": pr.values, "a": g.loc[ids, "ytot"].values, "pos": b25.values}, index=ids)
G["blend"] = np.exp(.5 * np.log(G.model + 1) + .5 * np.log(G.peer + 1)) - 1
rows = [dict(method=m, seg="all", **met(G[m], G.a)) for m in ["model", "peer", "blend"]]
for p, d in G.groupby("pos", observed=True): rows += [dict(method=m, seg=str(p), **met(d[m], d.a)) for m in ["model", "peer", "blend"]]
print(pd.DataFrame(rows).set_index(["seg", "method"]).to_string())

print("\n== 2  mature items, like-for-like")
m = O[O.mature].index
jo25 = M.loc[m, mo(2025, [7, 8, 9, 10])].sum(axis=1)
yoy_aj = (M.loc[m, mo(2026, [4, 5, 6])].sum(axis=1) + 5) / (M.loc[m, mo(2025, [4, 5, 6])].sum(axis=1) + 5)
yoy_mj = M.loc[m, mo(2026, [3, 4, 5, 6])].sum(axis=1) / M.loc[m, mo(2025, [3, 4, 5, 6])].sum(axis=1)
fc = O.loc[m, "final_total"]; exp_ = jo25 * yoy_aj * 0.843
print(f"aggregate (sum over the {len(m)} items): forecast / Jul-Oct 2025 = {fc.sum() / jo25.sum():.3f}; expected from trend = sum(Jul-Oct 2025 x Apr-Jun yoy x 0.843) / sum(Jul-Oct 2025) = {exp_.sum() / jo25.sum():.3f}; Mar-Jun yoy aggregate {M.loc[m, mo(2026,[3,4,5,6])].sum().sum() / M.loc[m, mo(2025,[3,4,5,6])].sum().sum():.3f} x 0.843 = {0.843 * M.loc[m, mo(2026,[3,4,5,6])].sum().sum() / M.loc[m, mo(2025,[3,4,5,6])].sum().sum():.3f}")
print(f"item medians: forecast / Jul-Oct 2025 = {(fc / jo25).median():.3f}; Apr-Jun yoy x 0.843 = {(yoy_aj * 0.843).median():.3f}; Mar-Jun yoy x 0.843 = {(yoy_mj * 0.843).median():.3f}")
print(f"spread of forecast vs expected (item level): p10 {(fc / exp_).quantile(.1):.2f}, median {(fc / exp_).median():.2f}, p90 {(fc / exp_).quantile(.9):.2f}; items outside 0.67-1.5x: {int(((fc / exp_) < .67).sum() + ((fc / exp_) > 1.5).sum())}")
# bias of trend x 0.843 at June/July 2025 origins (from 31_yoy_test.py logic)
src = open("eda/31_yoy_test.py", encoding="utf-8").read(); src = src[:src.index("def met(")]; ns = {}; exec(src, ns); D = ns["D"]
J = D[D.origin.isin(["2025-06", "2025-07"])]; print("trend x 0.843 at the June/July 2025 origins (items with a year of history):", met(J.yn * 0.843, J.a), "| the blend actually used (50/50 with model):", met(np.exp(.5 * np.log(J.p50 + 1) + .5 * np.log(J.yn * 0.843 + 1)) - 1, J.a))

print("\n== 3  floors")
f = O[(O.seg.isin(["Homecoming", "Halloween"])) & (~O.mature) & (O.seg_rule_pct > 0.005)]
pctl = lambda r: np.interp(np.log1p(r.after_segment_rule), np.log1p([r.p10c, r.p25c, r.model_answer, r.p75c, r.p90c]), [10, 25, 50, 75, 90], left=0, right=100)
print(pd.DataFrame({"seg": f.seg, "model": f.model_answer.round(0), "P75": f.p75c.round(0), "P90": f.p90c.round(0), "floored": f.after_segment_rule.round(0), "lift": f.seg_rule_pct.round(2),
                    "percentile_in_model_range": [round(float(pctl(r)), 0) for _, r in f.iterrows()]}).to_string())
src = open("eda/29_segment_rule_test.py", encoding="utf-8").read(); src = src[:src.index("def met(")]; ns = {}; exec(src, ns); g29 = ns["g"]
age = (pd.Timestamp("2025-06-30") - start.reindex(g29.style_color)).dt.days.values / 30.44
yg = g29[(age < 15) & g29.seg.isin(["Homecoming", "Halloween"]).values]
for s, d in list(yg.groupby("seg")) + [("both", yg)]:
    print(f"June 2025 origin, young {s} items: model {met(d.p50, d.ytot)} | with floor {met(d.rule, d.ytot)} | floor binding on {int((d.rule > d.p50 + 1e-6).sum())}")
