"""D347 (fork 2; user: "lets add some more temporal features for the trees like moments, entropy, derivatives etc"). A block of time-series
shape features from each item's daily units (all channels, cancelled lines dropped) in the 182 days before each origin, all scale-free and
known at the origin, added to the tree member (Main's 'ec_ns', eda/180q_models.py; the same fit and the BASE/FULL average) and refitted
time-honestly (training origins o' + 4 <= o). Pattern as eda/198t (fork D307).
  MOM  moments: coefficient of variation, skew and kurtosis of daily units (91 days) and of weekly units (26 weeks); max-day / mean-day;
       share of zero days (91)
  DER  derivatives: slope of log weekly units over the last 4, 8, 13 and 26 weeks (OLS), curvature (quadratic term) over 13 weeks,
       change of slope (last 8 weeks against the 8 before), last 4 weeks against the 4 before, weeks since the weekly peak (26)
  ENT  entropy: normalised Shannon entropy of daily units (91 days) and weekly units (26 weeks); spectral entropy of the weekly series
  RHY  rhythm: autocorrelation of daily units at lags 1 and 7 (91 days) and of weekly units at lags 1 and 4 (26 weeks); weekday
       concentration (share of the best weekday)
Variants: ec_ns refit (check), + each block, + all; and the all-block version with last year's window removed (o - 12 +- 1, the gate).
Scored alone and inside the five-member mean (swapping the tree member), before corrections and with the submitted correction multiplier.
Writes data/rc_temporal_feats.parquet, data/rc_tree_temporal_preds.parquet; stdout -> eda/tree_temporal_output.txt.
Run with NJOBS=3 on the laptop (user: the laptop must not lag)."""
import sys, os, time, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]; print("member table", T.shape, flush=True)
# ---------- the temporal block
dd = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color", "ny_date", "cancelled", "qty"]); dd = dd[~dd.cancelled]; dd["d"] = pd.to_datetime(dd.ny_date)
dd = dd.groupby(["style_color", "d"], as_index=False).qty.sum(); items = sorted(T.style_color.unique()); iix = {s: k for k, s in enumerate(items)}
days = pd.date_range(dd.d.min(), dd.d.max()); dix = {d: k for k, d in enumerate(days)}; dd = dd[dd.style_color.isin(iix)]
PAD = 182; U = np.zeros((len(items), len(days) + PAD), np.float32); U[dd.style_color.map(iix).values, dd.d.map(dix).values + PAD] = dd.qty.values   # zero days before the data start, so early origins have a full window
def nent(a):
    s = a.sum(1, keepdims=True); p = np.divide(a, s, out=np.zeros_like(a), where=s > 0); h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1); return np.where(s[:, 0] > 0, h / np.log(a.shape[1]), np.nan)
def moments(a):
    m = a.mean(1); sd = a.std(1); z = (a - m[:, None]) / np.where(sd > 0, sd, np.nan)[:, None]; return sd / np.where(m > 0, m, np.nan), np.nanmean(z ** 3, 1), np.nanmean(z ** 4, 1) - 3
def slope(L, n):
    x = np.arange(n) - (n - 1) / 2; return (L[:, -n:] * x).sum(1) / (x ** 2).sum()
def acf(a, k):
    a = a - a.mean(1, keepdims=True); v = (a ** 2).sum(1); return np.where(v > 0, (a[:, k:] * a[:, :-k]).sum(1) / np.where(v > 0, v, 1), np.nan)
parts = []
for o in sorted(T.origin.unique()):
    idx = np.where((T.origin == o).values)[0]; ii = T.style_color.map(iix).values[idx]; end = dix.get(pd.Period(o, "M").end_time.normalize())
    if end is None: end = len(days) - 1
    end = end + PAD
    D91 = U[ii, end - 90:end + 1]; D182 = U[ii, end - 181:end + 1]; W = D182.reshape(len(ii), 26, 7).sum(2); L = np.log1p(W); f = pd.DataFrame(index=idx)
    f["tm_cv_d"], f["tm_skew_d"], f["tm_kurt_d"] = moments(D91); f["tm_cv_w"], f["tm_skew_w"], f["tm_kurt_w"] = moments(W)
    f["tm_maxmean_d"] = D91.max(1) / np.where(D91.mean(1) > 0, D91.mean(1), np.nan); f["tm_zero_d"] = (D91 == 0).mean(1)
    for n in (4, 8, 13, 26): f[f"td_slope{n}"] = slope(L, n)
    x = np.arange(13) - 6.0; X2 = np.stack([np.ones(13), x, x ** 2 - (x ** 2).mean()], 1); f["td_curv13"] = (L[:, -13:] @ np.linalg.pinv(X2).T)[:, 2]
    f["td_dslope8"] = slope(L, 8) - slope(L[:, :-8], 8); f["td_l4p4"] = np.log((W[:, -4:].sum(1) + 1) / (W[:, -8:-4].sum(1) + 1)); f["td_wk_since_peak"] = 25 - W.argmax(1)
    f["te_ent_d"] = nent(D91); f["te_ent_w"] = nent(W); P = np.abs(np.fft.rfft(W - W.mean(1, keepdims=True), axis=1))[:, 1:] ** 2; f["te_spec"] = nent(P)
    f["tr_acf1_d"] = acf(D91, 1); f["tr_acf7_d"] = acf(D91, 7); f["tr_acf1_w"] = acf(W, 1); f["tr_acf4_w"] = acf(W, 4)
    wd = np.array([d.dayofweek for d in pd.date_range(days[0] + pd.Timedelta(days=end - PAD - 90), periods=91)]); S = np.stack([D91[:, wd == k].sum(1) for k in range(7)], 1); f["tr_wkday_max"] = S.max(1) / np.where(S.sum(1) > 0, S.sum(1), np.nan)
    parts.append(f)
TF = pd.concat(parts).reindex(T.index); T = pd.concat([T, TF], axis=1); BLK = {"MOM": [c for c in TF if c.startswith("tm_")], "DER": [c for c in TF if c.startswith("td_")], "ENT": [c for c in TF if c.startswith("te_")], "RHY": [c for c in TF if c.startswith("tr_")]}
T[["origin", "style_color"] + list(TF.columns)].to_parquet("data/rc_temporal_feats.parquet", index=False); print({k: len(v) for k, v in BLK.items()}, "| coverage", round(float(TF.notna().mean().mean()), 3), flush=True)
# ---------- refit
B0, F0_ = sets("ec_ns"); ALL = sum(BLK.values(), [])
VAR = [("ec_ns (refit, check)", [], 0)] + [(f"+ {k}", v, 0) for k, v in BLK.items()] + [("+ all four blocks", ALL, 0), ("ec_ns, last year removed", [], 1), ("+ all four, last year removed", ALL, 1)]; PRED = {}
if "gate2" in sys.argv:   # D347b: the last-year gate for the two blocks that carry the gain, and the two together
    VAR = [("ec_ns (refit, check)", [], 0), ("+ MOM + RHY", BLK["MOM"] + BLK["RHY"], 0), ("ec_ns, last year removed", [], 1), ("+ MOM, last year removed", BLK["MOM"], 1), ("+ RHY, last year removed", BLK["RHY"], 1), ("+ MOM + RHY, last year removed", BLK["MOM"] + BLK["RHY"], 1)]
for lab, add, noly in VAR:
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna() & ~((T.o_ord - (oo - 12)).abs() <= 1 if noly else False)]; te = T[T.origin == o]
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", len(F0_ + add), "features |", round(time.time() - t0), "s", flush=True)
# ---------- score
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); rec = (D.origin >= "2025-06").values; early = ~rec; allm = np.ones(len(D), bool); y = D.y.values; ar = (D.R3 - D.y).abs().values
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; rng = np.random.default_rng(0)
wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean())
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar})[rec].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B, n))].sum(1); d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]
    return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))]
m0 = PRED[VAR[0][0]].reindex(K).values; s0 = (X.sum(1) - D.qdyn.values + m0) / 5; print("refit check: max rel diff to the stored member", round(float(np.nanmax(np.abs(m0 - D.qdyn.values) / np.maximum(D.qdyn.values, 1))), 4))
rows = []; save = D[["origin", "style_color"]].copy()
for lab, add, noly in VAR:
    m = PRED[lab].reindex(K).values; st = (X.sum(1) - D.qdyn.values + m) / 5; save[lab] = m
    rows.append(dict(variant=lab, n_add=len(add), alone=r(wape(m)), alone_all=r(wape(m, allm)), stack=r(wape(st)), stack_all=r(wape(st, allm)), stack_early=r(wape(st, early)), stack_hit=r(hit(st), 3),
                     corrected=r(wape(st * corr)), corrected_all=r(wape(st * corr, allm)), ci_stack=boot(s0, st) if (add and not noly) else None))
R = pd.DataFrame(rows); print(R.to_string(index=False)); save.to_parquet("data/rc_tree_temporal_preds" + ("_gate2" if "gate2" in sys.argv else "") + ".parquet", index=False)
json.dump(rows, open("data/rc_tree_temporal" + ("_gate2" if "gate2" in sys.argv else "") + ".json", "w"), indent=0)
