"""Evaluation against a recency baseline (D109). Same 13 origins and items as eda/77/79 (4-month item totals).
Baselines: R3 = mean of the last 3 months x 4 (primary), R1 = last month x 4.
Metrics per system:
  rWAPE  = WAPE(system) / WAPE(R3) on the same items (< 1 = better than recency)
  win    = share of items where the system's absolute error is smaller than R3's (ties count half)
  mdlog  = median |log((pred + 1) / (actual + 1))|, every item weighs the same (WAPE is dominated by big sellers)
  bias   = total predicted / total actual - 1
Splits:
  ex post : by R3's own error on the item, |R3 / actual - 1| <= 20% (recency good), 20-50%, > 50% or actual 0 (recency bad).
            Uses the outcome, so it describes where models add value; it cannot be used to pick a model in advance.
  ex ante : by R3's track record on the same item at origin - 4 (its last complete 4-month window, known at the origin):
            same three bands, plus 'no record' for items too young to have one."""
import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/79_combo_eval.py", encoding="utf-8").read().split('C = ["rules"')[0])
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
P = lambda o, k: str(pd.Period(o, "M") + k)
T["R3"] = T.runrate
T["R1"] = [4 * M.loc[s, o] for o, s in zip(T.origin, T.style_color)]
# R3's track record: forecast made at origin-4 for months origin-3..origin, against what happened (needs 3 months of history before origin-4)
def past(o, s):
    if P(o, -6) not in M.columns: return np.nan
    h = M.loc[s, [P(o, -k) for k in (4, 5, 6)]]; a = M.loc[s, [P(o, -k) for k in (0, 1, 2, 3)]].sum()
    if (h == 0).all() or a == 0: return np.nan                      # not on sale then / nothing to score
    return abs(4 * h.mean() / a - 1)
T["r3_past"] = [past(o, s) for o, s in zip(T.origin, T.style_color)]
T["r3_now"] = np.where(T.y > 0, (T.R3 / T.y.replace(0, np.nan) - 1).abs(), np.inf)
band = lambda e: np.select([e <= 0.2, e <= 0.5], ["1 recency good (<=20%)", "2 recency middling (20-50%)"], "3 recency bad (>50%)")
T["post"] = band(T.r3_now)
T["ante"] = np.where(T.r3_past.isna(), "4 no record (young)", band(T.r3_past.fillna(9)))
SY = {"R1": "Last month x4", "rules": "Rule agents", "qmodel": "Quantile median", "lgb": "LightGBM", "nn": "Embedding NN", "lgb_nn": "LightGBM+NN",
      "q_nn": "Quantile x NN", "r_q": "Rules x quantile", "r_q_nn": "Rules x quantile x NN"}
def metrics(F):
    base = (F.R3 - F.y).abs()
    out = {}
    for c in SY:
        e = (F[c] - F.y).abs()
        out[SY[c]] = {"rWAPE": e.sum() / base.sum(), "win": ((e < base) + 0.5 * (e == base)).mean(), "mdlog": np.median(np.abs(np.log((F[c] + 1) / (F.y + 1)))),
                      "bias": F[c].sum() / F.y.sum() - 1}
    out["Recency R3 (reference)"] = {"rWAPE": 1.0, "win": 0.5, "mdlog": np.median(np.abs(np.log((F.R3 + 1) / (F.y + 1)))), "bias": F.R3.sum() / F.y.sum() - 1}
    return pd.DataFrame(out).T.round(3)
REC_ = [o for o in ORIGINS if o >= "2025-06"]; TR_ = T[T.origin.isin(REC_)]
for nm, F in [("ALL 13 ORIGINS", T), ("NINE RECENT ORIGINS", TR_)]:
    print(f"\n######## {nm}: R3 WAPE {(F.R3 - F.y).abs().sum() / F.y.sum():.3f} on {len(F)} item-origins")
    print(metrics(F).to_string())
    for col, title in [("post", "EX POST split (by recency's own error)"), ("ante", "EX ANTE split (by recency's track record at origin-4)")]:
        print(f"\n== {title}")
        for b, x in F.groupby(col):
            print(f"-- {b}: {len(x)} item-origins ({len(x) / len(F):.0%}), {x.y.sum() / F.y.sum():.0%} of units | R3 WAPE {(x.R3 - x.y).abs().sum() / x.y.sum():.3f}")
            print(metrics(x)[["rWAPE", "win", "mdlog", "bias"]].to_string())
# rWAPE by origin, and bootstrap of rWAPE for the main systems on recent origins
print("\n== rWAPE by origin (all items)")
print(pd.DataFrame({SY[c]: T.groupby("origin").apply(lambda x: (x[c] - x.y).abs().sum() / (x.R3 - x.y).abs().sum()) for c in ["R1", "rules", "qmodel", "lgb_nn", "q_nn", "r_q_nn"]}).round(3).to_string())
rng = np.random.default_rng(0)
def boot_r(F, c, B=1000):
    g = [x for _, x in F.groupby("origin")]; r = []
    for _ in range(B):
        s = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g]); r.append((s[c] - s.y).abs().sum() / (s.R3 - s.y).abs().sum())
    return np.round(np.percentile(r, [2.5, 97.5]), 3)
print("\n== 95% bootstrap interval of rWAPE, recent origins (all items | ex-ante 'recency good' | ex-ante 'recency bad' | no record)")
for c in ["rules", "qmodel", "lgb_nn", "q_nn", "r_q_nn"]:
    print(f"  {SY[c]:22s}", boot_r(TR_, c), boot_r(TR_[TR_.ante.str.startswith("1")], c), boot_r(TR_[TR_.ante.str.startswith("3")], c), boot_r(TR_[TR_.ante.str.startswith("4")], c))
T[["origin", "style_color", "y", "R3", "R1", "post", "ante"] + [c for c in SY if c != "R1"]].to_parquet("data/relative_eval.parquet", index=False)
