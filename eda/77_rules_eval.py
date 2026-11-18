"""Rule-agent evaluation (D103): score the agents' monthly-updated rule forecasts against our system on the same items and months.
Systems: rules (agent_eval/<o>/predictions_verified.csv), production quantile model median 4-month total (q_backtest q50),
monthly LightGBM ensemble (best_backtest_preds), embedding NN 6-seed median (nn_backtest_preds_S13x6), LightGBM+NN average,
run-rate baseline (last 3 months x 4), and rules+model average (a diagnostic, not a candidate).
Writes data/rules_eval_by_origin.csv, data/rules_eval_segments.csv and prints the summary."""
import os, numpy as np, pandas as pd
ROOT = "agent_eval"; rng = np.random.default_rng(0)
ORIGINS = [o for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")] if os.path.exists(f"{ROOT}/{o}/predictions_verified.csv")]
b = pd.read_parquet("data/best_backtest_preds.parquet"); b = b[b.origin.isin(ORIGINS)]
n = pd.read_parquet("data/nn_backtest_preds_S13x6.parquet")[["origin", "style_color", "h", "q50", "level"]]
b = b.merge(n, on=["origin", "style_color", "h"])
b["month"] = b.apply(lambda r: str(pd.Period(r.origin, "M") + r.h), axis=1)
R = pd.concat([pd.read_csv(f"{ROOT}/{o}/predictions_verified.csv", dtype={"month": str}).assign(origin=o) for o in ORIGINS]).rename(columns={"pred": "rules"})
m = b.merge(R, on=["origin", "style_color", "month"], how="left")
assert m.rules.notna().all(), "rules missing for some backtest item-months"
act = pd.read_parquet(f"{ROOT}/actuals.parquet"); chk = m.merge(act, on=["style_color", "month"], how="left"); assert np.allclose(chk.y_x, chk.y_y.fillna(0)), "actuals mismatch"
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
T = m.groupby(["origin", "style_color"]).agg(y=("y", "sum"), lgb=("pred", "sum"), nn=("q50", "sum"), rules=("rules", "sum"), level=("level", "first")).reset_index()
q = pd.read_parquet("data/q_backtest.parquet")[["origin", "style_color", "q50", "ytot", "age_m"]].rename(columns={"q50": "qmodel"})
T = T.merge(q, on=["origin", "style_color"], how="left"); assert np.allclose(T.y, T.ytot), "q_backtest totals differ"
T["runrate"] = [4 * M.loc[s, [str(pd.Period(o, "M") - k) for k in range(3)]].mean() for o, s in zip(T.origin, T.style_color)]
T["lgb_nn"] = (T.lgb + T.nn) / 2
T["rules_q"] = np.sqrt(T.rules.clip(lower=0.5) * T.qmodel.clip(lower=0.5))          # diagnostic only
SYS = {"rules": "Rule agents", "qmodel": "Production quantile median", "lgb": "LightGBM ensemble", "nn": "Embedding NN", "lgb_nn": "LightGBM + NN average", "runrate": "Run-rate (last 3 months x 4)", "rules_q": "Rules x quantile median (geometric)"}
it = pd.read_parquet("data/items.parquet").set_index("style_color")
MACRO = {"Top": "top", "Tank Top": "top", "T-Shirt": "top", "Crop Top": "top", "Corset": "top", "Strapless": "top", "Bodysuit": "top", "Shirt": "top", "Knitted Top": "top",
         "Jeans": "bottom", "Shorts": "bottom", "Skirt": "bottom", "Skort": "bottom", "Pants": "bottom", "Short Jeans": "bottom", "Sweatpants": "warm",
         "Hoodie": "warm", "Sweater": "warm", "Jacket": "warm", "Cardigan": "warm", "Dress": "dress", "Swimwear Top": "swim", "Swimwear Bottom": "swim"}
T["macro"] = T.style_color.map(it.product_type).map(MACRO).fillna("other")
T["age"] = pd.cut(T.age_m, [-1, 3, 12, 1e9], labels=["under 3 months", "3-12 months", "12 months+"]).astype(str)
T["target"] = T.style_color.isin(pd.read_csv("data/targets.csv").style_color)
T["occ"] = T.style_color.map(it.occasion).fillna("none").map(lambda s: "tagged" if s != "none" else "untagged")

def met(x, c):
    e = x[c] - x.y; r = x[c] / x.y.replace(0, np.nan)
    return pd.Series({"items": len(x), "wape": e.abs().sum() / x.y.sum(), "bias": x[c].sum() / x.y.sum() - 1,
                      "w20": ((r - 1).abs() <= 0.2).mean(), "w50": ((r - 1).abs() <= 0.5).mean()})
rows = []
for o, x in T.groupby("origin"):
    for c in SYS: rows.append(met(x, c).rename(None).to_frame().T.assign(origin=o, system=c))
BO = pd.concat(rows); BO.to_csv("data/rules_eval_by_origin.csv", index=False)
piv = BO.pivot(index="origin", columns="system", values="wape")[list(SYS)]
print(f"== WAPE of 4-month item totals by origin ({len(ORIGINS)} origins)"); print(piv.round(3).to_string())
summ = pd.DataFrame({c: {"WAPE mean": piv[c].mean(), "WAPE pooled": met(T, c).wape, "bias pooled": met(T, c).bias, "within 20%": met(T, c).w20, "within 50%": met(T, c).w50,
                         "rules better at": (piv.rules < piv[c]).sum() if c != "rules" else np.nan} for c in SYS}).T
print("\n== summary over all item-origins"); print(summ.round(3).to_string())
REC = [o for o in ORIGINS if o >= "2025-06"]          # the nine recent origins the production model was chosen on (D042)
TR = T[T.origin.isin(REC)]
summ2 = pd.DataFrame({c: {"WAPE mean": piv.loc[REC, c].mean(), "WAPE pooled": met(TR, c).wape, "bias pooled": met(TR, c).bias, "within 20%": met(TR, c).w20, "within 50%": met(TR, c).w50,
                          "rules better at": (piv.loc[REC, "rules"] < piv.loc[REC, c]).sum() if c != "rules" else np.nan} for c in SYS}).T
print(f"\n== summary over the {len(REC)} recent origins (Jun 2025 onwards)"); print(summ2.round(3).to_string())
ab = BO.pivot(index="origin", columns="system", values="bias")[["rules", "qmodel", "lgb", "nn", "lgb_nn"]]
print("\n== aggregate bias of 4-month totals by origin"); print(ab.round(3).to_string())

# monthly accuracy (item-month WAPE, and site aggregate by target month)
mm = m.assign(lgb_nn=(m.pred + m.q50) / 2)
print("\n== item-month WAPE: rules %.3f | LightGBM %.3f | NN %.3f | average %.3f" % tuple((mm[c] - mm.y).abs().sum() / mm.y.sum() for c in ["rules", "pred", "q50", "lgb_nn"]))
ag = mm.groupby(["origin", "h"])[["y", "rules", "pred", "q50"]].sum()
print("aggregate |log error| by horizon (rules / LightGBM / NN):")
print(pd.DataFrame({c: np.log(ag[c] / ag.y).abs().groupby(level="h").mean() for c in ["rules", "pred", "q50"]}).round(3).to_string())

# paired item bootstrap of pooled WAPE difference, resampling items within each origin
def boot(a, c, B=2000, F=None):
    F = T if F is None else F
    g = [x for _, x in F.groupby("origin")]; d = []
    for _ in range(B):
        s = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g]); d.append(((s[a] - s.y).abs().sum() - (s[c] - s.y).abs().sum()) / s.y.sum())
    return np.percentile(d, [2.5, 50, 97.5])
print("\n== pooled WAPE difference, rules minus system (95% item bootstrap; negative = rules better)")
for c in ["qmodel", "lgb", "nn", "lgb_nn", "runrate"]: print(f"  {SYS[c]:32s}", np.round(boot("rules", c, 1000), 3))
print("  recent origins only:")
for c in ["qmodel", "lgb", "nn", "lgb_nn"]: print(f"  {SYS[c]:32s}", np.round(boot("rules", c, 1000, TR), 3))

seg = []
for col in ["age", "macro", "target", "occ"]:
    for v, x in T.groupby(col):
        for c in ["rules", "qmodel", "lgb_nn"]: seg.append(met(x, c).rename(None).to_frame().T.assign(segment=col, value=str(v), system=c))
S = pd.concat(seg); S.to_csv("data/rules_eval_segments.csv", index=False)
print("\n== by segment (pooled): WAPE / bias"); P = S.pivot_table(index=["segment", "value"], columns="system", values=["wape", "bias"]); print(P.round(3).to_string())
