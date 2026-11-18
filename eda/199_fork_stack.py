"""D203 (fork): the fork's additions stacked on the D161 champion (rules x D159 quantile x NN, with the Black Friday fix):
  dyn3        rules x D198 quantile (D159 + dynamic co-occurrence neighbour features) x NN
  dyn3_bf     dyn3 x the Black Friday adjustment (eda/137/157 adj)
  dyn4        rules x D198 quantile x NN x TabDPT (D201)
  dyn4_bf     dyn4 x the Black Friday adjustment
against wx3 (D159 three-way) and wx3_bf (D161 champion). Standard table; paired bootstrap against wx3_bf.
Writes data/fk_stack_preds.parquet (origin, style_color, each system's 4-month total) for the scoreboard owner."""
import sys, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/157_stack_new.py", encoding="utf-8").read().split("BF = T.origin.isin")[0].replace('sys.stdout.reconfigure(encoding="utf-8"); ', ""))
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
rd = lambda p: np.nan_to_num(pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key).q50.values, nan=0.0)
T["q_wx"] = rd("data/q_backtest_wx_cpllm_we.parquet"); T["q_dyn"] = rd("data/q_backtest_fkdyn_dyn.parquet"); T["tabdpt"] = rd("data/q_backtest_fktab_tabdpt.parquet")
T["wx3"] = G(T.rules.values, T.q_wx.values, T.nn.values); T["wx3_bf"] = T.wx3 * T.adj
T["dyn3"] = G(T.rules.values, T.q_dyn.values, T.nn.values); T["dyn3_bf"] = T.dyn3 * T.adj
T["dyn4"] = G(T.rules.values, T.q_dyn.values, T.nn.values, T.tabdpt.values); T["dyn4_bf"] = T.dyn4 * T.adj
COLS = ["wx3", "wx3_bf", "dyn3", "dyn3_bf", "dyn4", "dyn4_bf"]
NM = {"wx3": "rules + D159 quantile + NN", "wx3_bf": "... + Black Friday fix (D161 champion)", "dyn3": "rules + D198 quantile (dynamic co-occurrence) + NN", "dyn3_bf": "D198 three-way + Black Friday fix",
      "dyn4": "rules + D198 quantile + NN + TabDPT", "dyn4_bf": "D198 three-way + TabDPT + Black Friday fix"}
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
for nm, m in [("nine recent origins", REC), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", REC & T.target.values), ("Black Friday origins", BF)]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z = T[REC].reset_index(drop=True)
print("\nbootstrap vs the D161 champion (wx3_bf), recent:", {c: np.round(sl.boot(Z, c, "wx3_bf"), 4).tolist() for c in COLS if c != "wx3_bf"})
Za = T.reset_index(drop=True); print("bootstrap vs the D161 champion, all 13:", {c: np.round(sl.boot(Za, c, "wx3_bf"), 4).tolist() for c in ["dyn3_bf", "dyn4_bf"]})
T[["origin", "style_color", "y"] + COLS].to_parquet("data/fk_stack_preds.parquet", index=False)
