"""D206 (fork): general stack scorer for the M5 runs. Every system is an equal geometric mean of members (as eda/199, 199b; Main's D169
buying-sim reconstruction uses the same rule). Members: rules, nn (from eda/157), adj (the Black Friday multiplier) and any
q_backtest file given as name=path. Systems are written as 'label:member*member*...' (adj multiplies after the mean).
Reference: p2 = rules*qdyn*nn*pf2 (the production champion, D169). Standard R3-relative table on four subsets; paired bootstrap
against p2 on the recent and all-13 sets. Optional out=<file> writes the systems' 4-month totals (origin, style_color, y, ...).
Usage: python eda/199d_stack_any.py m:cat=data/q_backtest_fkgb_cat.parquet ... s:'p2cat:rules*qdyn*nn*pf2*cat' ... [out=...]"""
import sys, io, contextlib, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("eda/157_stack_new.py", encoding="utf-8").read().split("BF = T.origin.isin")[0].replace('sys.stdout.reconfigure(encoding="utf-8"); ', ""))
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
def rd(p):   # 'path@q60' reads another quantile column (buying favours an upper quantile: the newsvendor critical ratio)
    p, q = (p.split("@") + ["q50"])[:2]
    return np.nan_to_num(pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key)[q].values, nan=0.0)
MEM = {"rules": T.rules.values, "nn": T.nn.values, "qdyn": rd("data/q_backtest_fkdyn_dyn.parquet"), "pf2": rd("data/q_backtest_fktab_tabpfn_v2.parquet")}
OUT = None; REF = "p2"; SYS = {"p2": "rules*qdyn*nn*pf2"}
for a in ARGS:
    if a.startswith("m:"): k, p = a[2:].split("=", 1); MEM[k] = rd(p)
    elif a.startswith("s:"): k, v = a[2:].split(":", 1); SYS[k] = v
    elif a.startswith("out="): OUT = a[4:]
    elif a.startswith("ref="): REF = a[4:]
A = lambda xs: np.mean(xs, axis=0)   # 'label:a:m1*m2' = arithmetic mean (less negative bias than the geometric mean)
for k, v in SYS.items():
    am = v.startswith("a:"); v = v[2:] if am else v
    ms = v.split("*"); bf = "adj" in ms; T[k] = (A if am else G)([MEM[m] for m in ms if m != "adj"]) * (T.adj.values if bf else 1)
COLS = list(SYS); NM = {k: f"{k} = {v}" for k, v in SYS.items()}
BF = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).values; REC = (T.origin >= "2025-06").values
for nm, m in [("nine recent origins", REC), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", REC & T.target.values), ("Black Friday origins", BF)]:
    Z = T[m].reset_index(drop=True); print(f"\n######## {nm}"); print(sl.table(Z, COLS, {k: v[m] for k, v in D.items()}, NM).round(4).to_string())
Z, Za = T[REC].reset_index(drop=True), T.reset_index(drop=True)
print(f"\nbootstrap WAPE(system) - WAPE({REF}): recent | all 13")
for c in [c for c in COLS if c != REF]: print(f"  {c:14s}", np.round(sl.boot(Z, c, REF), 4), "|", np.round(sl.boot(Za, c, REF), 4))
if OUT: T[["origin", "style_color", "y"] + COLS].to_parquet(OUT, index=False); print("saved", OUT)
