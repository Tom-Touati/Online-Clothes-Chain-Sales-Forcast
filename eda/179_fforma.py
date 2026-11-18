"""D175 (overnight M5 goal, direction 4): FFORMA-style learned combination weights (M4 runner-up; meta-learning over series features).
Members of p2 (D169): rules, the D198 quantile model, the NN, TabPFN v2 (4-month totals). Per item-origin the weights come from a
meta-model on item features: each member's log ratio to the members' geometric mean (disagreement profile) and to R3, log level,
age, product type, first target month, the Black Friday window flag, LLM scores, rules availability.
Time-honest: at origin o the meta-model trains on origins o' with o' + 4 <= o (first usable o = 2025-06, 4 origins); earlier
origins keep equal weights.
  cls     LightGBM multiclass on the member with the smallest |log error|; weights = predicted probabilities
  direct  small network (softmax weights) minimising the absolute error of the weighted geometric combination, in units, each
          item-origin weighted by 1 / its origin's actual total (a WAPE-like loss)
  shrink  weights = 0.5 x equal + 0.5 x direct
Writes data/fforma_preds.parquet; standard table and bootstrap against equal weights (p2), recent origins, all 13, targets."""
import sys, contextlib, io, glob, importlib.util, numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); torch.set_num_threads(4)
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); qv = lambda f: pd.read_parquet(f).set_index(["origin", "style_color"]).reindex(key).q50.values
MEM = ["rules", "qdyn", "nn", "tab2"]
T["qdyn"] = qv("data/q_backtest_fkdyn_dyn.parquet"); T["tab2"] = qv("data/q_backtest_fktab_tabpfn_v2.parquet")
LM = np.log(np.maximum(T[MEM].values, 0) + 0.5); GM = LM.mean(1, keepdims=True)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color")
X = pd.DataFrame(LM - GM, columns=[f"dev_{m}" for m in MEM], index=T.index)
X["g_r3"] = GM[:, 0] - np.log(T.R3.clip(lower=0) + 0.5); X["lvl"] = np.log1p(T.level); X["age"] = T.age_m if "age_m" in T else np.nan
X["ptype"] = it.product_type.reindex(T.style_color).astype("category").cat.codes.values; X["tm"] = [(pd.Period(o, "M") + 1).month for o in T.origin]
X["bf"] = T.origin.isin(["2025-07", "2025-08", "2025-09", "2025-10"]).astype(int); X["has_rules"] = T.rules.notna().astype(int)
for c in ["warmth", "party", "staple", "statement", "halloween"]: X[c] = L[c].reindex(T.style_color).values
X["disp"] = (LM - GM).std(1)
y = T.y.values; err = np.abs(LM - np.log(y + 0.5)[:, None]); best = err.argmin(1)
ORD = sorted(T.origin.unique()); oo = np.array([pd.Period(o).ordinal for o in T.origin]); tot = T.groupby("origin").y.transform("sum").values
W = {v: np.full((len(T), 4), 0.25) for v in ["cls", "direct", "shrink"]}
class Net(nn.Module):
    def __init__(s, d): super().__init__(); s.f = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 4))
    def forward(s, x): return torch.softmax(s.f(x), 1)
for o in ORD:
    po = pd.Period(o).ordinal; tr = oo + 4 <= po; te = (T.origin == o).values
    if np.unique(T.origin[tr]).size < 4: continue
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=8).fit(X[tr], best[tr])
    pr = np.zeros((te.sum(), 4)); pr[:, clf.classes_] = clf.predict_proba(X[te]); W["cls"][te] = pr
    Xn = X.fillna(X[tr].median()); mu, sd = Xn[tr].mean(), Xn[tr].std().replace(0, 1)
    xt = torch.tensor(((Xn[tr] - mu) / sd).values, dtype=torch.float32); lt = torch.tensor(LM[tr], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32); wt = torch.tensor(1 / tot[tr], dtype=torch.float32)
    torch.manual_seed(0); net = Net(xt.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(400):
        w = net(xt); f = torch.exp((w * lt).sum(1)) - 0.5; loss = ((f - yt).abs() * wt).sum() / wt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): W["direct"][te] = net(torch.tensor(((Xn[te] - mu) / sd).values, dtype=torch.float32)).numpy()
    W["shrink"][te] = 0.5 * 0.25 + 0.5 * W["direct"][te]
    print(o, "mean weights direct", W["direct"][te].mean(0).round(3), "| cls", W["cls"][te].mean(0).round(3), flush=True)
T["p2"] = np.exp(LM.mean(1)) - 0.5
for v, w in W.items(): T[f"ff_{v}"] = np.exp((w * LM).sum(1)) - 0.5
REC_ = [o for o in ORIGINS if o >= "2025-06"]; COLS = ["p2", "ff_cls", "ff_direct", "ff_shrink"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool)), ("recent, 100 targets", T.origin.isin(REC_) & T.target)]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs equal weights (p2), recent"); [print(f"  {c:10s}", np.round(sl.boot(F_, c, "p2"), 4)) for c in COLS[1:]]
T[["origin", "style_color"] + COLS].to_parquet("data/fforma_preds.parquet", index=False)
