"""D196 (fork): score the zero-shot time-series foundation models (eda/184 on the remote GPU; data/fk_tsfm_<model>.parquet) on the
eda/86 items and origins with the standard R3-relative table (eda/score_lib.py), alone and in the ensemble:
  alone                 the model's 4-month P50 (sum of the monthly medians)
  + to the best         rules x D159 quantile (temperature + events) x NN x foundation model (equal geometric weights)
  replaces NN           rules x D159 quantile x foundation model
Paired bootstrap against the D159 three-way (rules x D159 quantile x NN). Usage: python eda/185_score_tsfm.py [models...]"""
import sys, glob, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); G = lambda *xs: np.exp(np.mean([np.log(np.maximum(x, 0) + 0.5) for x in xs], axis=0)) - 0.5
T["q_wx"] = pd.read_parquet("data/q_backtest_wx_cpllm_we.parquet").set_index(["origin", "style_color"]).reindex(key).q50.values
T["wx3"] = G(T.rules.values, T.q_wx.values, T.nn.values)
MODELS = sys.argv[1:] or [p.split("fk_tsfm_")[1][:-8] for p in sorted(glob.glob("data/fk_tsfm_*.parquet")) if "emb" not in p]
cols, names = ["q_wx", "wx3"], {"q_wx": "D159 quantile model alone", "wx3": "rules + D159 quantile + NN (reference)"}
for m in MODELS:
    P = pd.read_parquet(f"data/fk_tsfm_{m}.parquet").groupby(["origin", "style_color"]).q50.sum().reindex(key)
    print(m, "items without a forecast:", int(P.isna().sum()))
    T[m] = P.fillna(0).values; T[m + "_add"] = G(T.rules.values, T.q_wx.values, T.nn.values, T[m].values); T[m + "_rnn"] = G(T.rules.values, T.q_wx.values, T[m].values)
    cols += [m, m + "_add", m + "_rnn"]; names |= {m: f"{m} alone (zero-shot)", m + "_add": f"rules + D159 quantile + NN + {m}", m + "_rnn": f"rules + D159 quantile + {m} (replaces NN)"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for s, msk in {"nine recent origins": T.origin.isin(REC_), "recent, 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins": np.ones(len(T), bool)}.items():
    print(f"\n######## {s}"); print(sl.table(T[msk], cols, D, names).round(4).to_string())
F = T[T.origin.isin(REC_)]
print("\n== paired bootstrap, recent: WAPE(system) - WAPE(rules + D159 quantile + NN)")
for c in cols[2:]: print(f"   {names[c]:52s}", np.round(sl.boot(F, c, "wx3"), 4))
