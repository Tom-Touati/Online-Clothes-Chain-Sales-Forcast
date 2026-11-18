"""D204 (fork): TabPFN members stacked on the accuracy champion (D203 dyn3_bf = rules x D198 quantile x NN x Black Friday fix).
  dyn3_bf             the champion (reference)
  p35f / p35f_bf      rules x D198 quantile x NN x TabPFN-3.5-fast (+ Black Friday fix)
  p35 / p35_bf        ... x TabPFN-3.5
  p2 / p2_bf          ... x TabPFN v2 (Apache-style licence; usable in production)
  p2_35f_bf           ... x TabPFN v2 x TabPFN-3.5-fast, Black Friday fix
  p2d / p2c / p2c_35fd  D206: TabPFN v2 with dyn feats; the same with conformalised tails (eda/199c, seg_tails); + 3.5-fast dyn
  p35fd / p2_35fd     D205: TabPFN-3.5-fast fed the dynamic co-occurrence neighbour features too (alone, and with v2)
Geometric mean of the members (as eda/199). Standard table; paired bootstrap against dyn3_bf.
Writes data/fk_stack_tabpfn_preds.parquet (origin, style_color, y, each system's 4-month total) for the scoreboard owner."""
import sys, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/157_stack_new.py", encoding="utf-8").read().split("BF = T.origin.isin")[0].replace('sys.stdout.reconfigure(encoding="utf-8"); ', ""))
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
rd = lambda p: np.nan_to_num(pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key).q50.values, nan=0.0)
T["q_dyn"] = rd("data/q_backtest_fkdyn_dyn.parquet")
for k, p in {"pf2": "tabpfn_v2", "pf35": "tabpfn_v35", "pf35f": "tabpfn_v35f", "pf35fd": "tabpfn_v35f_dyn", "pf2d": "tabpfn_v2_dyn"}.items(): T[k] = rd(f"data/q_backtest_fktab_{p}.parquet")
T["pf2c"] = rd("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet")   # D206 F6: TabPFN v2 + dyn feats, conformalised tails (eda/199c)
b = [T.rules.values, T.q_dyn.values, T.nn.values]
T["dyn3_bf"] = G(*b) * T.adj
for nm, xs in {"p35f": [T.pf35f.values], "p35": [T.pf35.values], "p2": [T.pf2.values], "p2_35f": [T.pf2.values, T.pf35f.values], "p35fd": [T.pf35fd.values], "p2_35fd": [T.pf2.values, T.pf35fd.values], "p2d": [T.pf2d.values], "p2c": [T.pf2c.values], "p2c_35fd": [T.pf2c.values, T.pf35fd.values]}.items():
    T[nm] = G(*b, *xs); T[nm + "_bf"] = T[nm] * T.adj
COLS = ["dyn3_bf", "p35f", "p35f_bf", "p35", "p35_bf", "p2", "p2_bf", "p2_35f", "p2_35f_bf", "p35fd", "p2_35fd", "p2d", "p2c", "p2c_35fd"]
NM = {"dyn3_bf": "D203 champion: rules + D198 quantile + NN + BF fix", "p35f": "rules + D198 q + NN + TabPFN-3.5-fast", "p35f_bf": "rules + D198 q + NN + TabPFN-3.5-fast + BF fix",
      "p35": "rules + D198 q + NN + TabPFN-3.5", "p35_bf": "rules + D198 q + NN + TabPFN-3.5 + BF fix", "p2": "rules + D198 q + NN + TabPFN v2", "p2_bf": "rules + D198 q + NN + TabPFN v2 + BF fix",
      "p2_35f": "rules + D198 q + NN + TabPFN v2 + 3.5-fast", "p2_35f_bf": "rules + D198 q + NN + TabPFN v2 + 3.5-fast + BF fix", "p35fd": "rules + D198 q + NN + TabPFN-3.5-fast with dyn feats", "p2_35fd": "rules + D198 q + NN + TabPFN v2 + 3.5-fast with dyn feats", "p2d": "rules + D198 q + NN + TabPFN v2 with dyn feats", "p2c": "... + TabPFN v2 dyn feats, conformal tails", "p2c_35fd": "... + TabPFN v2 dyn conformal + 3.5-fast dyn"}
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
for nm, m in [("nine recent origins", REC), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", REC & T.target.values), ("Black Friday origins", BF)]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z = T[REC].reset_index(drop=True); Za = T.reset_index(drop=True)
print("bootstrap vs p2 (production champion), recent:", {c: np.round(sl.boot(Z, c, "p2"), 4).tolist() for c in ["p2d", "p2c", "p2c_35fd"]}, "| all 13:", {c: np.round(sl.boot(Za, c, "p2"), 4).tolist() for c in ["p2d", "p2c", "p2c_35fd"]})
print("\nbootstrap vs dyn3_bf, recent:", {c: np.round(sl.boot(Z, c, "dyn3_bf"), 4).tolist() for c in COLS[1:]})
print("bootstrap vs dyn3_bf, all 13:", {c: np.round(sl.boot(Za, c, "dyn3_bf"), 4).tolist() for c in ["p35f", "p2", "p2_35f", "p35fd", "p2_35fd", "p2d", "p2c", "p2c_35fd"]})
T[["origin", "style_color", "y"] + COLS].to_parquet("data/fk_stack_tabpfn_preds.parquet", index=False)
