"""Fork "fashion" (D404): scores the next-month LSTM (eda/230b_lstm_gpu.py -> data/lstm_preds.parquet) at every backtest month-end on the
scored items, against
  R1         last month's units                       R3m   the mean of the last three months (the recency baseline, per month)
  NN m1      the neural-net member's month-1 median (data/nn_backtest_preds_S13ecx6.parquet, h = 1)
  rules m1   the rule agents' first month (agent_eval/<origin>/predictions_verified.csv)
  system m1  the submitted system's 4-month total (data/geo_nx5r_x_preds.parquet) x the month-1 share of the monthly split (geometric mean
             of the rules' and the network's monthly profiles, as the workbook), i.e. what the deliverable says for its first month
  blend      geometric mean of the LSTM and system m1
Metrics on next-month units: WAPE, ratio to R3m's WAPE, bias, share within +/-20%; recent 9, held out (2025-10 .. 2026-02), all 13, and
July 2025 (origin 2025-06); style bootstrap of each against system m1. stdout -> eda/lstm_score_output.txt"""
import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
import glob; P = pd.concat([pd.read_parquet(f).assign(version=lambda d, f=f: d.version + ("_lyd" if "_lyd" in f else "")) for f in sorted(glob.glob("data/lstm_preds*.parquet")) if "_h4" not in f]);   # D408: gate runs tagged _lyd   # D405: the 60- and 365-day versions together
Z = np.load("data/lstm_inputs.npz", allow_pickle=True)
E = pd.DataFrame({"origin": Z["ev_origin"], "style_color": Z["ev_item"], "y1": Z["ev_y1"].astype(float)}); E = E[E.y1.notna()]
for v in P.version.unique(): E = E.merge(P[P.version == v][["origin", "style_color", "y1_hat"]].rename(columns={"y1_hat": f"lstm_{v}"}), on=["origin", "style_color"], how="left")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
mo = lambda o, k: str(pd.Period(o, "M") + k)
E["R1"] = [float(M.loc[s, o]) for o, s in zip(E.origin, E.style_color)]; E["R3m"] = [float(M.loc[s, [mo(o, 0), mo(o, -1), mo(o, -2)]].mean()) for o, s in zip(E.origin, E.style_color)]
NN = pd.read_parquet("data/nn_backtest_preds_S13ecx6.parquet"); NNp = NN.pivot_table(index=["origin", "style_color"], columns="h", values="q50")
E["nn_m1"] = NNp[1].reindex(pd.MultiIndex.from_frame(E[["origin", "style_color"]])).values
ru, prof = [], []
for o in sorted(E.origin.unique()):
    f = f"agent_eval/{o}/predictions_verified.csv"
    if not os.path.exists(f): continue
    r = pd.read_csv(f, dtype={"month": str}).pivot(index="style_color", columns="month", values="pred").reindex(columns=[mo(o, h) for h in (1, 2, 3, 4)])
    n = NNp.loc[o].reindex(r.index) if o in NNp.index.get_level_values(0) else None
    rp = r.div(r.sum(axis=1), axis=0).clip(lower=1e-4); npf = n.div(n.sum(axis=1), axis=0).clip(lower=1e-4) if n is not None else rp
    gp = np.exp((np.log(rp.values) + np.log(npf.reindex(rp.index).fillna(rp).values)) / 2); gp = gp / gp.sum(1, keepdims=True)
    ru.append(pd.DataFrame({"origin": o, "style_color": r.index, "rules_m1": r.iloc[:, 0].values, "share1": gp[:, 0]}))
E = E.merge(pd.concat(ru), on=["origin", "style_color"], how="left")
G = pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo"]]; E = E.merge(G, on=["origin", "style_color"], how="left")
E["sys_m1"] = E.geo * E.share1.fillna(0.25)
for v in P.version.unique(): E[f"blend_{v}"] = np.exp((np.log1p(E[f"lstm_{v}"]) + np.log1p(E.sys_m1)) / 2) - 1
E["style"] = E.style_color.str.split("_").str[0]; y = E.y1.values; rng = np.random.default_rng(0)
COLS = ["R1", "R3m", "nn_m1", "rules_m1", "sys_m1"] + [c for c in E.columns if c.startswith(("lstm_", "blend_"))]
def ci(a, b, m):
    d = pd.DataFrame(dict(s=E["style"].values[m], a=np.abs(E[a].values - y)[m], b=np.abs(E[b].values - y)[m])).groupby("s").sum()
    i = rng.integers(0, len(d), size=(400, len(d))); v = d.a.values[i].sum(1) / d.b.values[i].sum(1); return f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
O = E.origin.values
for lab, m in [("recent 9", O >= "2025-06"), ("held out", O >= "2025-10"), ("all 13", O <= "2026-02"), ("July 2025 (origin 2025-06)", O == "2025-06")]:
    m = m & E[COLS].notna().all(axis=1).values; base = np.abs(E.R3m.values - y)[m].sum()
    print(f"\n== {lab}: {m.sum()} item-months, actual {y[m].sum():,.0f} units")
    for c in COLS:
        f = E[c].values; print(f"   {c:12s} WAPE {np.abs(f - y)[m].sum() / y[m].sum():.3f} | / R3m {np.abs(f - y)[m].sum() / base:.3f} | bias {f[m].sum() / y[m].sum() - 1:+.3f} | within 20% {np.mean(np.abs(f - y)[m] <= 0.2 * y[m]):.2f}"
              + (f" | error / system m1's {np.abs(f - y)[m].sum() / np.abs(E.sys_m1.values - y)[m].sum():.3f} {ci(c, 'sys_m1', m)}" if c.startswith(("lstm", "blend")) else ""))
E.to_parquet("data/lstm_scored.parquet", index=False)

# D407: the 4-month versions (lstm_preds*_h4.parquet) against the system's own 4-month forecast, WAPE / R3 (the project's measure)
H = [pd.read_parquet(f) for f in sorted(glob.glob("data/lstm_preds*_h4.parquet"))]
if H:
    H = pd.concat(H); B = pd.read_parquet("data/dec_base_nx5r26.parquet"); B = B[B.y.notna()][["origin", "style_color", "y", "R3", "rules", "nn"]]
    B = B.merge(G, on=["origin", "style_color"], how="left")
    for v in H.version.unique(): B = B.merge(H[H.version == v][["origin", "style_color", "y1_hat"]].rename(columns={"y1_hat": f"h4_{v}"}), on=["origin", "style_color"], how="left")
    B["style"] = B.style_color.str.split("_").str[0]; y4 = B.y.values; O4 = B.origin.values
    C4 = ["R3", "rules", "nn", "geo"] + [c for c in B.columns if c.startswith("h4_")]
    for v in [c for c in B.columns if c.startswith("h4_")]: B["blend_" + v] = np.exp((np.log1p(B[v]) + np.log1p(B.geo)) / 2) - 1
    C4 += [c for c in B.columns if c.startswith("blend_h4_")]
    def ci4(a, m):
        d = pd.DataFrame(dict(s=B["style"].values[m], a=np.abs(B[a].values - y4)[m], b=np.abs(B.geo.values - y4)[m])).groupby("s").sum()
        i = rng.integers(0, len(d), size=(400, len(d))); v = d.a.values[i].sum(1) / d.b.values[i].sum(1); return f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    for lab, m in [("recent 9", O4 >= "2025-06"), ("held out", O4 >= "2025-10"), ("all 13", O4 <= "2026-02"), ("Jul-Oct 2025 (origin 2025-06)", O4 == "2025-06")]:
        m = m & B[C4].notna().all(axis=1).values; base = np.abs(B.R3.values - y4)[m].sum()
        print(f"\n== 4 MONTHS, {lab}: {m.sum()} items, actual {y4[m].sum():,.0f} units (geo = the submitted system)")
        for c in C4:
            f = B[c].values; print(f"   {c:20s} WAPE / R3 {np.abs(f - y4)[m].sum() / base:.4f} | WAPE {np.abs(f - y4)[m].sum() / y4[m].sum():.3f} | bias {f[m].sum() / y4[m].sum() - 1:+.3f} | within 20% {np.mean(np.abs(f - y4)[m] <= 0.2 * y4[m]):.2f}"
                  + (f" | error / system's {np.abs(f - y4)[m].sum() / np.abs(B.geo.values - y4)[m].sum():.3f} {ci4(c, m)}" if "h4_" in c else ""))
