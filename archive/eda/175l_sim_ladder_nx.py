"""D260: newsvendor ladder with the D253 / D260 volume-weighted new-feature LightGBM (data/q_backtest_nx_w03ec.parquet, q65) in place of the
fork's w03 at q65 (u65ec; the conformal and TabPFN members at q65 as in u65). Built from eda/175h (D236). Writes data/sim_ladder_nx_windows.csv."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
exec(open("eda/87_sim_systems.py", encoding="utf-8").read().split("SYS = [")[0])
Q = {k: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q50 for k, f in
     {"dyn": "data/q_backtest_fkdyn_dyn.parquet", "sel": "data/q_backtest_fkgb_sel.parquet", "v2": "data/q_backtest_fktab_tabpfn_v2.parquet", "v2d": "data/q_backtest_fktab_tabpfn_v2_dyn.parquet",
      "conf": "data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet", "v35d": "data/q_backtest_fktab_tabpfn_v35f_dyn.parquet", "v2d32": "data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet", "w03": "data/q_backtest_fkgb_wvol03.parquet"}.items()}
for _k, _f in {"w03": "data/q_backtest_fkgb_wvol03_x.parquet", "conf": "data/q_backtest_fkconf_pfn2dyn_seg_tails_x.parquet", "v2d32": "data/q_backtest_fktab_tabpfn_v2_dyn_e32_x.parquet"}.items():
    for _q in ("q65", "q70", "q75"): Q[_k + "_" + _q[1:]] = pd.read_parquet(_f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"])[_q]
Q["w03ec_65"] = pd.read_parquet("data/q_backtest_nx_w03ec.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).q65
CUR = {"extra": []}
def mm(o, items, src):
    b = sim.BT[sim.BT.origin == o].pivot(index="style_color", columns="h", values="pred").reindex(items)
    sh = b.values / b.sum(axis=1).values[:, None]
    mem = lambda k: np.nan_to_num(sh * Q[k].loc[o].reindex(items).values[:, None])
    lg = mem(CUR.get("base", "dyn"))
    nn = v2.NN[v2.NN.origin == o].pivot(index="style_color", columns="h", values="q50").reindex(items).values; nn = np.where(np.isnan(nn), lg, nn)
    ru = RUL[o].reindex(items).values; ru = np.where(np.isnan(ru), lg, ru)
    ex = [np.where(np.isnan(m := mem(k)), lg, m) for k in CUR["extra"]]
    return np.maximum(0, np.mean([np.maximum(x, 0) for x in [ru, lg, nn, *ex]], axis=0))   # arithmetic mean (D190)
v2.month_matrix = mm
rows = []
for start in pd.date_range("2025-03-01", "2026-01-01", freq="MS"):
    ideal = sim.simulate(sim.Window(start), "ideal")
    for nm, base, ex in [("u65ec", "w03ec_65", ["conf_65", "v2d32_65"])]:
        CUR["extra"] = ex; CUR["base"] = base; w = v2.build(start, src="r_q_nn", shape="cal")
        rows.append(dict(start=str(start.date()), variant=nm, **sim.summarize(w, v2.simulate2(w, now="level", peak_off=True), ideal)))
    print(start.date(), flush=True)
R = pd.DataFrame(rows); R.to_csv("data/sim_ladder_nx_windows.csv", index=False)
piv = R.pivot(index="start", columns="variant", values="loss_pct").join(pd.read_csv("data/sim_fk7_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct")).join(pd.read_csv("data/sim_tabpfn_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct")).join(pd.read_csv("data/sim_fkm5_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct")).join(pd.read_csv("data/sim_dyn_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct")).join(pd.read_csv("data/sim_wx_stack_windows.csv").pivot(index="start", columns="variant", values="loss_pct"))
print(piv.round(4).to_string()); print("mean loss:", piv.mean().round(4).to_dict())
