"""Persistent product-group bias (D138): dresses and bottoms were under-forecast at nearly every origin (eda/116 panel 2). Time-honest
multiplicative correction per product group: at origin o, the factor f_g that minimises WAPE of the three-way on group g's history rows
(origins with o' + 4 <= o; grid 0.8-1.4), shrunk halfway to 1 ('half') or used fully ('full'); groups with < 300 history rows keep 1.
Also the same with the factor as the ratio of summed actuals to summed forecasts (bias-zeroing, 'ratio_half'). Standard table."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
ORD = {o: pd.Period(o, "M").ordinal for o in T.origin.unique()}; OO = T.origin.map(ORD).values; C0 = T.r_q_nn.values; Y = T.y.values; G = T.macro.values
for c in ["half", "full", "ratio_half"]: T[c] = C0.copy()
LOG = []
for o in sorted(T.origin.unique()):
    te = np.flatnonzero(T.origin.values == o); h = np.flatnonzero(OO + 4 <= ORD[o])
    for g in np.unique(G[te]):
        hg = h[G[h] == g]; tg = te[G[te] == g]
        if len(hg) < 300: continue
        f = min(np.arange(0.8, 1.41, 0.02), key=lambda f: np.abs(f * C0[hg] - Y[hg]).sum()); r = Y[hg].sum() / C0[hg].sum()
        T.loc[tg, "half"] = C0[tg] * (0.5 + 0.5 * f); T.loc[tg, "full"] = C0[tg] * f; T.loc[tg, "ratio_half"] = C0[tg] * (0.5 + 0.5 * r)
        if o >= "2025-10": LOG.append((o, g, round(float(f), 2), round(float(r), 2)))
print("factors at the later origins (origin, group, WAPE-optimal f, actual/forecast ratio):"); [print("  ", x) for x in LOG]
COLS = ["r_q_nn", "half", "full", "ratio_half"]; NM = {"r_q_nn": "three-way", "half": "group factor, half", "full": "group factor, full", "ratio_half": "group bias ratio, half"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D, NM).round(4).to_string())
Z = T[T.origin.isin(REC)]; print("\nbootstrap vs three-way:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in COLS[1:]})
print("bias by group, recent, three-way vs group factor half:", {g: (round(x.r_q_nn.sum() / x.y.sum() - 1, 3), round(x.half.sum() / x.y.sum() - 1, 3)) for g, x in Z.groupby("macro")})
