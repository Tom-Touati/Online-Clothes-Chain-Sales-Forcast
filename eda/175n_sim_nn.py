"""D273: as eda/175k_sim_nx.py, with the NN member replaced by the fork's D272 NN on the D252 event + launch-curve inputs (data/nn_backtest_preds_S13ecx6.parquet)
and the corrections re-fitted on that system (data/geo_ecnsnn_x_preds.parquet)."""
"""D256: buying simulation (production policy, 11 windows; machinery of eda/175j / D249) for the D253 members and the D255 shape.
Systems = arithmetic mean of rules, a LightGBM member (stored D198 'dyn' or a data/q_backtest_nx_<tag>.parquet median), NN,
conformal TabPFN and TabPFN dyn 32 (p2c32a construction). The 4-month members are split over months by
  lgb      the LightGBM monthly profile (as before)
  ts       D255: geometric mean of the LightGBM profile and last year's like-for-like product-type monthly pattern, renormalised
Optional per-item multiplier 'geo': D248 for the stored member, the D257 re-fit (data/geo_<tag>_preds.parquet) for a new one.
Usage: python eda/175k_sim_nx.py <tag> [...]  (tag = an nx variant; 'dyn' = the stored member). Writes data/sim_nx_windows.csv."""
import sys, numpy as np, pandas as pd
TAGS = sys.argv[1:] or ["dyn"]
sys.stdout.reconfigure(encoding="utf-8")
exec(open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
v2.NN = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet")   # D272 NN
FILES = {"dyn": "data/q_backtest_fkdyn_dyn.parquet", "conf": "data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet", "v2d32": "data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet"}
FILES |= {t: f"data/q_backtest_nx_{t}.parquet" for t in TAGS if t != "dyn"}
Q = {k: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50 for k, f in FILES.items()}
_bs = pd.read_parquet("data/bias_seg_preds.parquet").merge(pd.read_parquet("data/breakout_preds.parquet")[["origin", "style_color", "res_all_0.5n"]], on=["origin", "style_color"])
_bs["geo"] = np.sqrt(_bs["vol_age_disc_0.5n"].clip(lower=0) * _bs["res_all_0.5n"].clip(lower=0)); _bs["geo_n"] = _bs.geo * _bs.groupby("origin").F.transform("sum") / _bs.groupby("origin").geo.transform("sum")
MULT = (_bs.set_index(["origin", "style_color"]).geo_n / _bs.set_index(["origin", "style_color"]).F.clip(lower=1e-9)).clip(0.2, 5)
MULTS = {"dyn": MULT}   # D257: the geo correction re-fitted on each new system where available (data/geo_<tag>_preds.parquet)
for t in TAGS:
    if t != "dyn" and __import__("os").path.exists("data/geo_ecnsnn_x_preds.parquet"):
        _g = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").set_index(["origin", "style_color"]); MULTS[t] = (_g.geo / _g.F.clip(lower=1e-9)).clip(0.2, 5)
PT = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(Mo.index).fillna("na"); FIRST = Mo.gt(0).idxmax(axis=1)
def type_ly(o, items):
    P = pd.Period(o, "M"); ly = [str(P + h - 12) for h in range(1, 5)]; coh = FIRST < str(P - 14)
    MT = Mo[coh].groupby(PT[coh]).sum().reindex(columns=ly).fillna(0); ts = (MT + 1).div((MT + 1).sum(1), axis=0)
    return ts.reindex(PT.reindex(items).fillna("na").values).fillna(0.25).values
CUR = {}
def mm(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    sh = b.values / b.sum(axis=1).values[:, None]
    if CUR["shape"] == "ts":
        g = np.exp(0.5 * np.log(np.clip(sh, 1e-6, None)) + 0.5 * np.log(np.clip(type_ly(o, items), 1e-6, None))); sh = np.where(np.isfinite(sh), g / g.sum(1, keepdims=True), sh)
    mem = lambda k: np.nan_to_num(sh * Q[k].loc[o].reindex(items).values[:, None])
    lg = mem(CUR["base"])
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    ex = [np.where(np.isnan(m := mem(k)), lg, m) for k in ["conf", "v2d32"]]
    out = np.maximum(0, np.mean([np.maximum(x, 0) for x in [ru, lg, nn, *ex]], axis=0))
    if CUR["mult"]: out = out * MULTS.get(CUR["base"], MULT).loc[o].reindex(items).fillna(1.0).values[:, None]
    return out
v2.month_matrix = mm
VARS = [(t, s, g) for t in TAGS for s in ("lgb", "ts") for g in (False, True)]
VARS = [v for v in VARS if v[1] == "lgb"]   # the stored member with the LightGBM shape is already simulated (p2c32a 10.74%, p2c32a_geo 10.78%)
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for t, s, g in VARS:
        CUR.update(base=t, shape=s, mult=g); w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=f"{t}+nn|{s}{'|geo' if g else ''}", **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); old = pd.read_csv("data/sim_nn_windows.csv") if __import__("os").path.exists("data/sim_nn_windows.csv") else pd.DataFrame()
R = pd.concat([old[~old.variant.isin(R.variant)] if len(old) else old, R]); R.to_csv("data/sim_nn_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct"); print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).sort_values().to_dict())
