"""D201 (fork): a candidate quantile-model replacement in every ensemble role, against the D159 three-way (rules x D159 quantile x NN).
For each candidate file (q_backtest layout; argv: name=path): alone; replacing the D159 quantile model; added as a fourth member;
sharing the quantile slot (geometric mean of the D159 quantile and the candidate). Standard R3-relative table, paired bootstrap
against the D159 three-way. Usage: python eda/197_score_members.py tabdpt=data/q_backtest_fktab_tabdpt.parquet [...]"""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
ARGV = dict(a.split("=", 1) for a in sys.argv[1:]); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
rd = lambda p: pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key).q50.values
T["q_wx"] = rd("data/q_backtest_wx_cpllm_we.parquet"); T["wx3"] = G(T.rules.values, T.q_wx.values, T.nn.values)
cols, N = ["q_wx", "wx3"], {"q_wx": "D159 quantile alone", "wx3": "rules + D159 quantile + NN (reference)"}
for nm, p in ARGV.items():
    T[nm] = np.nan_to_num(rd(p), nan=0.0)
    T[nm + "_rep"] = G(T.rules.values, T[nm].values, T.nn.values); T[nm + "_add"] = G(T.rules.values, T.q_wx.values, T.nn.values, T[nm].values)
    T[nm + "_sh"] = G(T.rules.values, G(T.q_wx.values, T[nm].values), T.nn.values)
    cols += [nm, nm + "_rep", nm + "_add", nm + "_sh"]
    N |= {nm: f"{nm} alone", nm + "_rep": f"rules + {nm} + NN (replaces D159 quantile)", nm + "_add": f"rules + D159 quantile + NN + {nm}", nm + "_sh": f"rules + (D159 quantile x {nm}) + NN"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for s, m in {"nine recent origins": T.origin.isin(REC_), "recent, 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins": np.ones(len(T), bool)}.items():
    print(f"\n######## {s}"); print(sl.table(T[m], cols, D, N).round(4).to_string())
F = T[T.origin.isin(REC_)]
print("\n== paired bootstrap, recent: WAPE(system) - WAPE(rules + D159 quantile + NN)")
for c in cols[2:]: print(f"   {N[c]:56s}", np.round(sl.boot(F, c, "wx3"), 4))
