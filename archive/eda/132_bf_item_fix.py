"""Black Friday fix with item-level lift (D137b), time-honest, origins 2025-07..2025-10 (windows containing November 2025).
Start: the three-way's monthly split (eda/112 'base'). Step 1 (level): November scaled to last year's like-for-like ratio (D136, full).
Step 2 (allocation): within November, items re-weighted by their predicted Black Friday lift L_i (eda/131 model trained on the 2024 lifts
with features at origin o - 12, predicting from features at origin o): November_i x (25 + 5 L_i) / (25 + 5 L_mean), renormalised so the
November total stays at step 1's level (the Thanksgiving-to-Cyber-Monday span is 5 of November's 30 days). Items without a prediction keep 1.
Scored on 4-month totals in the standard table for the four Black Friday origins and the nine recent origins."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, importlib.util, warnings, io, contextlib
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(open("eda/130_nov_calibration.py", encoding="utf-8").read().split("tot = X.groupby")[0].replace("CC = [", "_CC = [").replace("sys.stdout.reconfigure(encoding=\"utf-8\"); ", ""))
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
DD = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(0).asfreq("D").fillna(0)
def lifts(t):
    wk = DD.loc[t:t + pd.Timedelta(days=4)].sum() / 5 * 7; base = DD.loc[t - pd.Timedelta(days=28):t - pd.Timedelta(days=1)].mean() * 7
    return np.log((wk + 1) / (base + 1))[base >= 3]
L24 = lifts(pd.Timestamp("2024-11-28"))
Fe = pd.read_parquet("data/features.parquet"); Fe = Fe[Fe.h == 1]
for c in ["ptype", "macro"] + [c for c in Fe.columns if c.startswith("a_")]: Fe[c] = Fe[c].astype("category")
XF = [c for c in ["sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "age_m", "lvl", "f_r7", "f_r28", "trend91", "peak_r", "ptype", "macro", "color_num", "own_sea", "lfl_type"] + [c for c in Fe.columns if c.startswith("occ_")]]
X["bf"] = X["calnov1.0"]
for o in ["2025-07", "2025-08", "2025-09", "2025-10"]:
    o12 = str(pd.Period(o, "M") - 12); tr = Fe[Fe.origin == o12].set_index("style_color").join(L24.rename("lift"), how="inner"); te = Fe[Fe.origin == o].set_index("style_color")
    m = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(tr[XF], tr.lift)
    Lh = pd.Series(np.exp(m.predict(te[XF])), index=te.index)
    h = (pd.Period("2025-11", "M") - pd.Period(o, "M")).n; sel = (X.origin == o) & (X.h == h)
    w = ((25 + 5 * Lh) / (25 + 5 * Lh.mean())).reindex(X.loc[sel, "style_color"]).fillna(1.0).values
    nov = X.loc[sel, "calnov1.0"].values; new = nov * w; X.loc[sel, "bf"] = new * nov.sum() / new.sum()
    print(o, f"train items {len(tr)} | predicted lift mean {Lh.mean():.2f}, p10 {Lh.quantile(.1):.2f}, p90 {Lh.quantile(.9):.2f} | November weights p10 {np.quantile(w, .1):.3f} p90 {np.quantile(w, .9):.3f}")
tot = X.groupby(["origin", "style_color"])[["calnov1.0", "bf"]].sum().reset_index()
T = T.merge(tot, on=["origin", "style_color"], how="left"); T["calnov1.0"] = T["calnov1.0"].fillna(T.r_q_nn); T["bf"] = T["bf"].fillna(T.r_q_nn)
COLS = ["r_q_nn", "calnov1.0", "bf"]; NM = {"r_q_nn": "three-way", "calnov1.0": "+ November level (last year)", "bf": "+ November level + item lift allocation"}
B = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values
for nm, msk in [("Black Friday origins (Jul-Oct 2025)", B), ("nine recent origins", (T.origin >= "2025-06").values), ("recent, 100 targets", (T.origin >= "2025-06").values & T.target.values)]:
    Z = T[msk].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[msk] for k, v in D.items()}, NM).round(4).to_string())
Z = T[B].reset_index(drop=True); print("\nbootstrap, BF origins:", {f"{a}-{b}": np.round(sl.boot(Z, a, b), 4).tolist() for a, b in [("bf", "calnov1.0"), ("bf", "r_q_nn")]})
XN = X[(X.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"])) & np.array([str(pd.Period(o, "M") + h) == "2025-11" for o, h in zip(X.origin, X.h)])]
print("November 2025 item-month WAPE:", {c: round((XN[c] - XN.y).abs().sum() / XN.y.sum(), 4) for c in ["base", "calnov1.0", "bf"]})
# save the per-item November adjustment ratios (fixed / three-way monthly) for the buying simulation (eda/136)
A = X[["origin", "style_color", "h", "base", "calnov1.0", "bf"]].copy(); A["r_level"] = A["calnov1.0"] / A.base.clip(lower=1e-9); A["r_bf"] = A.bf / A.base.clip(lower=1e-9)
A[["origin", "style_color", "h", "r_level", "r_bf"]].to_parquet("data/bf_fix_ratios.parquet", index=False)
