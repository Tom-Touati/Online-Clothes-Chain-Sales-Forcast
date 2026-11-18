"""Fork 2 (user: "make the new system nx5 with the new calibration": ranges "based on item and group history, with a focus on these months",
i.e. Jul-Oct 2025 and Mar-Jun 2026). The D329 per-segment (Mondrian) structure, refitted on the new nx5's own backtest errors with the two
focus windows weighted up:
  item     widths from the conformal TabPFN member's quantiles: w_X = log((q50 + 1) / (qX + 1)) below, log((qX + 1) / (q50 + 1)) above, floor 0.02
  group    segment = age band at the origin (< 3, 3-6, 6-12, 12+ months) x volume third (recency level ranked within the origin); per segment
           and side a factor a so that the weighted share of actuals beyond (P50 + 1) exp(-/+ a w) - 1 is 10% (P10 / P90) and 25% (P25 / P75)
  focus    rows of origins 2025-06 (Jul-Oct 2025) and 2026-02 (Mar-Jun 2026) weigh omega; omega = 1 is the D329 method on the new system
Read: leave one origin out (as D328): coverage overall, by segment, on the focus windows, pinball loss of the four quantiles in units / R3's error.
2026: factors from all 13 origins -> data/ranges_nx5r_2026.csv (ratios r10 / r25 / r75 / r90; P = (final + 1) x r - 1).
stdout -> eda/ranges_nx5r_output.txt"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")
Q = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").drop_duplicates(["origin", "style_color"])
G = G.merge(Q[["origin", "style_color", "level", "q10", "q25", "q50", "q75", "q90"]], on=["origin", "style_color"], how="left")
AGE = lambda a: pd.cut(a.fillna(99), [-1, 3, 6, 12, 1e9], labels=["under 3 months", "3-6 months", "6-12 months", "over 12 months"]).astype(str)
VOL = lambda Z: Z.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])).astype(str)
def widths(Z):
    out = {}
    for q in ["q10", "q25"]: out[q] = np.maximum(np.log((Z.q50 + 1) / (Z[q] + 1)), 0.02)
    for q in ["q75", "q90"]: out[q] = np.maximum(np.log((Z[q] + 1) / (Z.q50 + 1)), 0.02)
    return pd.DataFrame(out, index=Z.index).fillna(pd.Series({"q10": 0.6, "q25": 0.3, "q75": 0.3, "q90": 0.6}))
G["seg"] = AGE(G.age_m) + " | " + VOL(G); W = widths(G); e = np.log((G.y + 1) / (G.geo + 1)).values
TGT = {"q10": 0.10, "q25": 0.25, "q75": 0.75, "q90": 0.90}; FOCUS = G.origin.isin(["2025-06", "2026-02"]).values
def wquant(v, wt, p):
    o = np.argsort(v); v, wt = v[o], wt[o]; c = (np.cumsum(wt) - 0.5 * wt) / wt.sum(); return float(np.interp(p, c, v))
def factors(m, omega):
    wt = np.where(FOCUS, omega, 1.0); F = {}
    for s in np.unique(G.seg[m]):
        k = m & (G.seg.values == s)
        if k.sum() < 30: k = m   # tiny segment: pooled
        F[s] = {q: (-wquant(e[k] / W[q].values[k], wt[k], p) if q in ("q10", "q25") else wquant(e[k] / W[q].values[k], wt[k], p)) for q, p in TGT.items()}
    return F
def apply(F, rows):
    P = np.zeros((len(rows), 4))
    for j, q in enumerate(["q10", "q25", "q75", "q90"]):
        a = np.array([F.get(s, F[next(iter(F))])[q] for s in G.seg.values[rows]]); sgn = -1 if q in ("q10", "q25") else 1
        P[:, j] = (G.geo.values[rows] + 1) * np.exp(sgn * a * W[q].values[rows]) - 1
    return np.maximum(P, 0)
def loo(omega):
    P = np.zeros((len(G), 4))
    for o in G.origin.unique():
        te = (G.origin == o).values; P[te] = apply(factors(~te, omega), np.where(te)[0])
    return P
y = G.y.values; R3 = G.R3.values
def read(P, m):
    pin = sum(np.maximum(p * (y - P[:, j]), (p - 1) * (y - P[:, j]))[m].sum() for j, p in enumerate([.1, .25, .75, .9])) / np.abs(R3 - y)[m].sum()
    return dict(below10=round(float((y < P[:, 0])[m].mean()), 3), below25=round(float((y < P[:, 1])[m].mean()), 3), above75=round(float((y > P[:, 2])[m].mean()), 3),
                above90=round(float((y > P[:, 3])[m].mean()), 3), in1090=round(float(((y >= P[:, 0]) & (y <= P[:, 3]))[m].mean()), 3),
                in2575=round(float(((y >= P[:, 1]) & (y <= P[:, 2]))[m].mean()), 3), pinball=round(float(pin), 4))
O = G.origin.values; SETS = {"all 13": np.ones(len(G), bool), "recent 9": O >= "2025-06", "Jul-Oct 25": O == "2025-06", "Mar-Jun 26": O == "2026-02", "held out": O >= "2025-10"}
res = {}
for om in [1, 4]:
    P = loo(om); res[om] = P
    print(f"omega {om}:"); [print(f"   {k:10s}", read(P, m)) for k, m in SETS.items()]
# by segment, chosen vs omega 1
best = 4; print("\nchosen omega 4: within 0.0002 of omega 8 on the focus windows and held out (first run, omegas 1 / 2 / 4 / 8), better on all 13 and on segment balance (omega 8 leaves young items about 40% inside P25-P75)")
for s in sorted(G.seg.unique()):
    m = G.seg.values == s; a, b = read(res[1], m), read(res[best], m)
    print(f"{s:28s} n {m.sum():5d} | omega 1 in1090 {a['in1090']} in2575 {a['in2575']} | omega {best} in1090 {b['in1090']} in2575 {b['in2575']}")
# 2026
Fall = factors(np.ones(len(G), bool), best)
Z = pd.read_parquet("data/fk_2026_pf2c.parquet").drop_duplicates("style_color").reset_index(drop=True)
Z["seg"] = AGE(Z.age_m) + " | " + VOL(Z); Wz = widths(Z)
out = pd.DataFrame({"style_color": Z.style_color, "segment": Z.seg})
for q, nm in [("q10", "r10"), ("q25", "r25"), ("q75", "r75"), ("q90", "r90")]:
    a = np.array([Fall.get(s, Fall[next(iter(Fall))])[q] for s in Z.seg]); out[nm] = np.exp((-1 if q in ("q10", "q25") else 1) * a * Wz[q].values).round(4)
out.to_csv("data/ranges_nx5r_2026.csv", index=False)
print("\n2026 ratios, median over items:", out[["r10", "r25", "r75", "r90"]].median().round(3).to_dict(), "| omega", best)
old = pd.read_csv("data/fk_ranges_2026.csv", comment="#").set_index("style_color"); j = out.set_index("style_color").join(old[["r10", "r25", "r75", "r90"]], rsuffix="_d329")
print("D329 ratios, median:", j[["r10_d329", "r25_d329", "r75_d329", "r90_d329"]].median().round(3).to_dict())
# saved for the item explorer's past forecasts: the leave-one-out ranges with the chosen omega (added after the first runs)
_P = res[best]; pd.DataFrame({"origin": G.origin, "style_color": G.style_color, "p10": _P[:, 0], "p25": _P[:, 1], "p75": _P[:, 2], "p90": _P[:, 3]}).to_parquet("data/ranges_nx5r_loo.parquet", index=False)
