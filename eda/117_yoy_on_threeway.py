"""Does the production year-on-year rule (D052) help the three-way combination at every origin? (D127)
For items with a year of own history (>= 50 units in the same 4 months a year earlier, live in all three months a year before the origin):
  yoy = units in the same 4 months last year x (last 3 months + 5) / (same 3 months last year + 5) x ageing
  ageing = median of actual / (yoy without ageing) over the history origins (outcomes complete by the origin), else 0.843 (D052's value)
  blend = geometric 50/50 of the three-way and yoy (D052), and a 'seasonal groups only' version (warm, swim, other).
Standard table on recent and all origins, plus bias by origin x macro before/after."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; P = lambda o, k: str(pd.Period(o, "M") + k)
raw = []
for o, s in zip(T.origin, T.style_color):
    ly = [P(o, h - 12) for h in range(1, 5)]; b3 = [P(o, -k - 12) for k in range(3)]; n3 = [P(o, -k) for k in range(3)]
    if b3[-1] not in M.columns or s not in M.index or (M.loc[s, b3] <= 0).any() or M.loc[s, ly].sum() < 50: raw.append(np.nan); continue
    raw.append(M.loc[s, ly].sum() * (M.loc[s, n3].sum() + 5) / (M.loc[s, b3].sum() + 5))
T["yoy_raw"] = raw
ORD = {o: pd.Period(o, "M").ordinal for o in T.origin.unique()}; OO = T.origin.map(ORD).values; AG = {}
for o in sorted(T.origin.unique()):
    h = (OO + 4 <= ORD[o]) & T.yoy_raw.notna().values & (T.yoy_raw.values > 0)
    AG[o] = float(np.median(T.y.values[h] / T.yoy_raw.values[h])) if h.sum() >= 200 else 0.843
T["yoy"] = T.yoy_raw * T.origin.map(AG)
print("time-honest ageing factor by origin:", {k: round(v, 3) for k, v in AG.items()})
has = T.yoy.notna(); print(f"item-origins with a year of own history: {has.mean():.0%}")
T["blend"] = np.where(has, np.exp(0.5 * np.log(T.r_q_nn + 1) + 0.5 * np.log(T.yoy.fillna(0) + 1)) - 1, T.r_q_nn)
SEAS = T.macro.isin(["warm", "swim", "other"])
T["blend_seas"] = np.where(has & SEAS, T.blend, T.r_q_nn)
T["yoy_q"] = np.where(has, np.exp(0.5 * np.log(T.qmodel + 1) + 0.5 * np.log(T.yoy.fillna(0) + 1)) - 1, T.qmodel)
COLS = ["qmodel", "yoy_q", "r_q_nn", "blend", "blend_seas"]
NM = {"qmodel": "quantile model", "yoy_q": "quantile + yoy rule (production D052)", "r_q_nn": "three-way", "blend": "three-way + yoy rule", "blend_seas": "three-way + yoy, seasonal groups only"}
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("nine recent origins", T[T.origin.isin(REC)]), ("all 13 origins", T), ("recent, items with own history", T[T.origin.isin(REC) & has]),
              ("recent, 100 targets", T[T.origin.isin(REC) & T.target])]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, D, NM).round(4).to_string())
Z = T[T.origin.isin(REC)]
print("\nbootstrap, recent:", {f"{a}-{b}": np.round(sl.boot(Z, a, b), 4).tolist() for a, b in [("blend", "r_q_nn"), ("blend_seas", "r_q_nn"), ("yoy_q", "qmodel")]})
Zh = Z[has[Z.index]]
for c in ["r_q_nn", "blend"]:
    print(f"\nbias by origin x macro, items with own history, {c}:"); print(Zh.groupby(["origin", "macro"]).apply(lambda x: x[c].sum() / x.y.sum() - 1).unstack().round(2).to_string())
