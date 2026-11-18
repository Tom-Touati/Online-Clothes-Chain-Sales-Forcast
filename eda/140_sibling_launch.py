"""Value of the launch plan (D144): do existing items lose sales the models do not foresee when a new colour of the same style launches
during the forecast window? The flag uses the outcome (first sale of a sibling in the 4 target months), as a stand-in for the merchandising
calendar the business knows in advance. Also: a new item of the same product type launching in the window (type-level assortment growth).
Bias and WAPE / R3 of the three-way and members by flag, recent origins."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/error_map_items.parquet"); R = T[T.origin >= "2025-06"].copy()
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1)); it = pd.read_parquet("data/items.parquet").set_index("style_color")
style = pd.Series(M.index.str.split("_").str[0], index=M.index); ptype = it.product_type.reindex(M.index)
def new_in_window(o, s, by):
    P = pd.Period(o, "M"); lo, hi = str(P + 1), str(P + 4)
    key = style if by == "style" else ptype
    sib = key.index[(key == key.get(s)) & (key.index != s)]
    f = first.reindex(sib); return int(((f >= lo) & (f <= hi)).sum())
R["new_sib"] = [new_in_window(o, s, "style") for o, s in zip(R.origin, R.style_color)]
R["new_type"] = [new_in_window(o, s, "type") for o, s in zip(R.origin, R.style_color)]
def seg(x):
    e3 = (x.R3 - x.y).abs().sum()
    return pd.Series({"n": len(x), "bias 3-way": x.r_q_nn.sum() / x.y.sum() - 1, "rules": x.rules.sum() / x.y.sum() - 1, "q": x.qmodel.sum() / x.y.sum() - 1,
                      "nn": x.nn.sum() / x.y.sum() - 1, "WAPE/R3": (x.r_q_nn - x.y).abs().sum() / e3})
print("== a new colour of the same style launches in the window:"); print(R.groupby(R.new_sib.clip(upper=2)).apply(seg).round(3).to_string())
R["tb"] = pd.qcut(R.new_type.rank(method="first"), 4, labels=["fewest new same-type launches", "q2", "q3", "most"])
print("\n== new items of the same product type launching in the window (quartiles):"); print(R.groupby("tb").apply(seg).round(3).to_string())
print("\nmedian new same-type launches by quartile:", R.groupby("tb").new_type.median().to_dict())

# time-honest uplift for items whose style gets a new colour in the window (the launch plan treated as known in advance)
import importlib.util
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
A = T.copy(); A["new_sib"] = [new_in_window(o, s, "style") for o, s in zip(A.origin, A.style_color)]
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([A.origin, A.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
ORD = {o: pd.Period(o, "M").ordinal for o in A.origin.unique()}; OO = A.origin.map(ORD).values; C0 = A.r_q_nn.values; Y = A.y.values; F = (A.new_sib > 0).values
A["sib_half"] = C0; A["sib_full"] = C0; LOG = []
for o in sorted(A.origin.unique()):
    te = np.flatnonzero((A.origin.values == o) & F); h = np.flatnonzero((OO + 4 <= ORD[o]) & F)
    if len(h) < 100 or len(te) == 0: continue
    f = min(np.arange(0.9, 1.61, 0.02), key=lambda f: np.abs(f * C0[h] - Y[h]).sum()); LOG.append((o, len(h), round(float(f), 2)))
    A.loc[te, "sib_half"] = C0[te] * (0.5 + 0.5 * f); A.loc[te, "sib_full"] = C0[te] * f
print("\ntime-honest factors (origin, history rows, f):", LOG)
REC = (A.origin >= "2025-06").values
for nm, m in [("nine recent origins, all items", REC), ("recent, items whose style gets a new colour", REC & F)]:
    Z = A[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, ["r_q_nn", "sib_half", "sib_full"], {k: v[m] for k, v in D.items()}).round(4).to_string())
Z = A[REC].reset_index(drop=True); print("\nbootstrap vs three-way, recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in ["sib_half", "sib_full"]})
