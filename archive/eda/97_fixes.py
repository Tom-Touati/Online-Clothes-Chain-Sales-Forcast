"""Fixes from the error map (D115), all time-honest: the fix applied at origin o learns only from origins o' with o' + 4 <= o
(their 4-month outcomes are complete by o). Origins with no such history keep the plain three-way combination.
  w_glob   : geometric weights over (rules, quantile, NN) on a 0.1 grid, chosen by pooled WAPE on the history
  w_macro  : the same per product group (falls back to w_glob below 300 history rows), shrunk halfway to w_glob
  w_age    : the same per age band
  bias_seg : multiplicative factor per volume band x age band (grid 0.8-1.4) minimising WAPE on the history, shrunk halfway to 1
  stack    : LightGBM (L1, 150 trees, 7 leaves, >= 100 rows per leaf) predicts log((y+1)/(c+1)) for the three-way c from member
             log-differences, disagreement, level, age, volume, retail share, trend, product group, occasion tag, origin month;
             stack_half applies half the predicted correction.
Scored with eda/score_lib.py on the nine recent origins and all 13."""
import numpy as np, pandas as pd, contextlib, io, sys, itertools, lightgbm as lgb, warnings, importlib.util
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
_a = sys.argv; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
sys.argv = _a
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
F1 = pd.read_parquet("data/features.parquet"); F1 = F1[F1.h == 1][["origin", "style_color", "level", "sh_ret", "trend91", "f_r28"]]
T = T.merge(F1, on=["origin", "style_color"], how="left", suffixes=("", "_f"))
T["vol"] = pd.cut(T.groupby("origin")["level"].rank(pct=True), [0, .2, .4, .6, .8, 1.01], labels=False)
LG = {c: np.log(T[c].clip(lower=0.5)).values for c in ["rules", "qmodel", "nn", "lgb"]}
GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=2) if w[0] + w[1] <= 1.0001]
Y = T.y.values; C0 = T.r_q_nn.values


def comb(idx, w): return np.exp(w[0] * LG["rules"][idx] + w[1] * LG["qmodel"][idx] + (1 - w[0] - w[1]) * LG["nn"][idx])


def best_w(idx): return min(GRID, key=lambda w: np.abs(comb(idx, w) - Y[idx]).sum())


ORD = {o: pd.Period(o, "M").ordinal for o in ORIGINS}; OO = T.origin.map(ORD).values
NEW = {c: C0.copy() for c in ["w_glob", "w_macro", "w_age", "bias_seg", "stack", "stack_half"]}
X = pd.DataFrame({"dr": LG["rules"] - LG["qmodel"], "dn": LG["nn"] - LG["qmodel"], "dl": LG["lgb"] - LG["qmodel"],
                  "dis": np.std(np.vstack(list(LG.values())), axis=0), "lc": np.log1p(C0), "lvl": np.log1p(T.level.values), "age": T.age_m.values,
                  "vol": T.vol.values, "ret": T.sh_ret.values, "trend": T.trend91.values, "pace": T.f_r28.values,
                  "macro": T.macro.astype("category").values, "occ": (T.occ == "tagged").astype(int).values, "om": T.origin.str[-2:].astype(int).values})
TGT = np.log((Y + 1) / (C0 + 1)); WLOG = []
for o in ORIGINS:
    te = np.flatnonzero(T.origin.values == o); hist = np.flatnonzero(OO + 4 <= ORD[o])
    if len(hist) == 0: continue
    wg = best_w(hist); NEW["w_glob"][te] = comb(te, wg); WLOG.append((o, len(hist), (wg[0], wg[1], round(1 - wg[0] - wg[1], 1))))
    for col, cname in [("macro", "w_macro"), ("age", "w_age")]:
        cv = T[col].values
        for v in np.unique(cv[te]):
            h = hist[cv[hist] == v]; t = te[cv[te] == v]
            w = best_w(h) if len(h) >= 300 else wg; w = tuple(0.5 * a + 0.5 * b for a, b in zip(w, wg))
            NEW[cname][t] = comb(t, w)
    vb, ab = T.vol.values, T.age.values
    for v in np.unique(vb[te]):
        for a in np.unique(ab[te]):
            h = hist[(vb[hist] == v) & (ab[hist] == a)]; t = te[(vb[te] == v) & (ab[te] == a)]
            if len(h) < 100 or len(t) == 0: continue
            f = min(np.arange(0.8, 1.41, 0.05), key=lambda f: np.abs(f * C0[h] - Y[h]).sum()); NEW["bias_seg"][t] = C0[t] * (0.5 + 0.5 * f)
    if len(hist) >= 1500:
        m = lgb.LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, random_state=0).fit(X.iloc[hist], TGT[hist])
        p = m.predict(X.iloc[te]); NEW["stack"][te] = (C0[te] + 1) * np.exp(p) - 1; NEW["stack_half"][te] = (C0[te] + 1) * np.exp(p / 2) - 1
for c, v in NEW.items(): T[c] = v
print("time-honest global weights (origin, history rows, (rules, quantile, NN)):"); [print("  ", w) for w in WLOG]
COLS = ["R3", "qmodel", "q_nn", "r_q_nn", "w_glob", "w_macro", "w_age", "bias_seg", "stack", "stack_half"]
NM = {"R3": "R3", "qmodel": "quantile model", "q_nn": "quantile + NN", "r_q_nn": "rules + quantile + NN (equal)"}
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, F in [("nine recent origins", T[T.origin.isin(REC_)]), ("all 13 origins", T), ("recent, 100 targets", T[T.origin.isin(REC_) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(F, COLS, D, NM).round(4).to_string())
F = T[T.origin.isin(REC_)]
print("\n== 95% bootstrap of WAPE(fix) - WAPE(equal three-way), nine recent origins")
for c in ["w_glob", "w_macro", "w_age", "bias_seg", "stack", "stack_half"]: print(f"  {c:10s}", np.round(sl.boot(F, c, "r_q_nn"), 4))
T.to_parquet("data/fixes_items.parquet", index=False)
