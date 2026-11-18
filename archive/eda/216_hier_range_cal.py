"""Fork 2 (user: "calibration should also happen for the ranges, based on item and group history, with a focus on these months" -- the
months being last year's test window Jul-Oct 2025 and the latest Mar-Jun 2026).
Ranges as quantiles of the system's own past log errors r = log((y + 1) / (F + 1)), F = submitted system (five-member mean x corrections):
  levels   item, embedding cluster, product type, all items; at each level the weighted 10 / 25 / 75 / 90% quantiles of r over past rows
  shrink   all -> product type -> cluster -> item, each step blending the quantiles in with n / (n + kappa)
  focus    rows of the focus windows weigh omega times more. In the backtest the 'focus' is the latest ended window (the same season a year
           earlier does not exist before 2026); for 2026 it is Jul-Oct 2025 (origin 2025-06) and Mar-Jun 2026 (origin 2026-02).
  time     at origin o only rows with windows ended (origins <= o - 4); settings chosen on the tune origins 2025-06 .. 2025-09, judged on
           the held-out 2025-10 .. 2026-02, against the stored P10 / P90.
Scores: share of actuals below P10 / above P90 (target 10% / 10%), below P25 / above P75 (25% / 25%), pinball loss in units over the
four quantiles (and over P10 / P90 alone against the stored ranges). Writes data/range_cal_2026.csv for the 100 targets (candidate only),
data/range_cal.json; stdout -> eda/range_cal_output.txt"""
import sys, json, itertools, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
K = pd.MultiIndex.from_arrays([D.origin, D.style_color])
q50 = lambda f: pd.read_parquet(f).drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K).q50.values
nn_ec = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet").groupby(["origin", "style_color"]).q50.sum().reindex(K).values
M = np.column_stack([D.rules.values, q50("data/q_backtest_nx_ec_ns.parquet"), np.where(np.isfinite(nn_ec), nn_ec, D.nn.values),
                     q50("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet"), q50("data/q_backtest_fktab_tabpfn_v2_dyn_e32.parquet")])
g = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet").drop_duplicates(["origin", "style_color"]).set_index(["origin", "style_color"]).reindex(K)
F = M.mean(1) * (g.geo / g.F.clip(lower=1e-9)).fillna(1).values
y, R3 = D.y.values, D.R3.values
cl = pd.read_parquet("data/clusters_2026-06.parquet").set_index("style_color").cluster
LEV = {"item": D.style_color.values, "cluster": D.style_color.map(cl).fillna(-1).astype(int).astype(str).values, "ptype": D.ptype.astype(str).values, "all": np.array(["all"] * len(D))}
r = np.log((y + 1) / (F + 1)); oi = pd.PeriodIndex(D.origin, freq="M").asi8; O = D.origin.values
QS = [0.10, 0.25, 0.75, 0.90]
def wq(df):
    """weighted quantiles of r per group: df has k, r, w"""
    out = {}
    for k, s in df.sort_values("r").groupby("k", sort=False):
        c = np.cumsum(s.w.values); c = (c - 0.5 * s.w.values) / c[-1]
        out[k] = [float(np.interp(q, c, s.r.values)) for q in QS] + [len(s)]
    return pd.DataFrame.from_dict(out, orient="index", columns=QS + ["n"])
def quantiles(rows, Hm, wts, k_item, k_grp):
    Q = np.zeros((len(rows), 4))
    for lev, kap in [("all", 0), ("ptype", k_grp), ("cluster", k_grp), ("item", k_item)]:
        T = wq(pd.DataFrame(dict(k=LEV[lev][Hm], r=r[Hm], w=wts[Hm])))
        T = T.reindex(LEV[lev][rows]); n = T.n.fillna(0).values; v = T[QS].values
        if lev == "all": Q = np.nan_to_num(v); continue
        lam = np.where(n >= 2, n / (n + kap), 0.0)[:, None]; Q = np.where(np.isfinite(v), (1 - lam) * Q + lam * v, Q)
    return Q
def ranges(k_item, k_grp, omega):
    P = np.full((len(D), 4), np.nan)
    for o in np.unique(oi):
        rows = np.where(oi == o)[0]; Hm = oi <= o - 4
        if Hm.sum() < 200: continue
        last = oi[Hm].max(); wts = np.where(oi == last, omega, 1.0)
        P[rows] = (F[rows, None] + 1) * np.exp(quantiles(rows, Hm, wts, k_item, k_grp)) - 1
    return P
def pin(yv, qv, q): d = yv - qv; return np.maximum(q * d, (q - 1) * d)
def score(P, m, stored=False):
    m = m & np.isfinite(P[:, [0, 3]]).all(1)
    s = dict(n=int(m.sum()), below10=round(float((y[m] < P[m, 0]).mean()), 3), above90=round(float((y[m] > P[m, 3]).mean()), 3))
    if not stored: s.update(below25=round(float((y[m] < P[m, 1]).mean()), 3), above75=round(float((y[m] > P[m, 2]).mean()), 3))
    s["pin1090"] = round(float((pin(y[m], P[m, 0], .1) + pin(y[m], P[m, 3], .9)).sum() / np.abs(R3 - y)[m].sum()), 4)
    s["logwidth"] = round(float(np.median(np.log((P[m, 3] + 1) / (P[m, 0] + 1)))), 3); return s, m
ST = np.column_stack([D.p10.values, np.full(len(D), np.nan), np.full(len(D), np.nan), D.p90.values])
TUNE = (O >= "2025-06") & (O <= "2025-09"); HOLD = O >= "2025-10"; REC = O >= "2025-06"
grid = []
for ki, kg, om in itertools.product([2, 5, 15], [20, 100, 400], [1, 2, 4]):
    P = ranges(ki, kg, om); s, _ = score(P, TUNE); h, _ = score(P, HOLD)
    err = abs(s["below10"] - .1) + abs(s["above90"] - .1) + abs(s["below25"] - .25) + abs(s["above75"] - .25)
    grid.append(dict(k_item=ki, k_grp=kg, omega=om, tune_cov_err=round(err, 3), tune_pin=s["pin1090"], hold_pin=h["pin1090"], hold_below10=h["below10"], hold_above90=h["above90"]))
G = pd.DataFrame(grid).sort_values(["tune_pin"]); print(G.to_string(index=False))
b = G.iloc[0]; Pb = ranges(int(b.k_item), int(b.k_grp), float(b.omega)); Pg = ranges(10 ** 9, 10 ** 9, 1.0)
res = {}
for nm, m in [("tune", TUNE), ("held out", HOLD), ("recent 9", REC)]:
    _, mm = score(Pb, m)
    res[nm] = dict(stored=score(ST, mm, stored=True)[0], global_only=score(Pg, mm)[0], hierarchical=score(Pb, mm)[0])
print("\nchosen", dict(b)); print(json.dumps(res, indent=1))
# held out by size: where the hierarchy changes coverage
_, mm = score(Pb, HOLD); dec = pd.qcut(R3[mm], 4, labels=["small", "mid-small", "mid-large", "large"])
for d in dec.categories:
    z = np.zeros(len(D), bool); z[np.where(mm)[0][dec == d]] = True
    print(d, "stored", score(ST, z, True)[0], "| hierarchical", score(Pb, z)[0])
# 2026: history = every origin (windows end by Jun 2026), focus = Jul-Oct 2025 and Mar-Jun 2026
Z = pd.read_csv("data/final_2026_nx4_cascade.csv", index_col=0)
Hm = np.ones(len(D), bool); wts = np.where(np.isin(O, ["2025-06", "2026-02"]), float(b.omega) if b.omega > 1 else 2.0, 1.0)
items = Z.index.values; LEV26 = {"item": items, "cluster": pd.Series(items).map(cl).fillna(-1).astype(int).astype(str).values, "ptype": Z.product_type.astype(str).values, "all": np.array(["all"] * len(items))}
Q = np.zeros((len(items), 4))
for lev, kap in [("all", 0), ("ptype", int(b.k_grp)), ("cluster", int(b.k_grp)), ("item", int(b.k_item))]:
    T = wq(pd.DataFrame(dict(k=LEV[lev][Hm], r=r[Hm], w=wts[Hm]))).reindex(LEV26[lev]); n = T.n.fillna(0).values; v = T[QS].values
    if lev == "all": Q = np.nan_to_num(v); continue
    lam = np.where(n >= 2, n / (n + kap), 0.0)[:, None]; Q = np.where(np.isfinite(v), (1 - lam) * Q + lam * v, Q)
tot = Z.total.values
C = pd.DataFrame({"total": tot, "P10_now": Z.P10.round().values, "P90_now": Z.P90.round().values,
                  **{f"P{int(q * 100)}_cal": np.round((tot + 1) * np.exp(Q[:, j]) - 1) for j, q in enumerate(QS)}}, index=items)
C.to_csv("data/range_cal_2026.csv")
print("\n2026, 100 targets: median P10/total now", round(float(np.nanmedian(C.P10_now / C.total)), 3), "cal", round(float(np.median(C.P10_cal / C.total)), 3),
      "| P90/total now", round(float(np.nanmedian(C.P90_now / C.total)), 3), "cal", round(float(np.median(C.P90_cal / C.total)), 3))
print("items without stored P10/P90", int(C.P10_now.isna().sum()), "| sum of P10 / P90 now", int(C.P10_now.sum()), int(C.P90_now.sum()), "| cal", int(C.P10_cal.sum()), int(C.P90_cal.sum()))
json.dump(dict(grid=grid, chosen={k: float(v) for k, v in b.items()}, scores=res), open("data/range_cal.json", "w"), indent=1)
