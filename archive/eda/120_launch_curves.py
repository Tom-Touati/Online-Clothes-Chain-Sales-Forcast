"""Analog launch curves for young items (D129), time-honest. For an item a days old at the origin (a <= 120) with S units so far:
peers = items launched at least a + 122 days before the origin (so their next-122-day window after age a is complete), within the last
540 days, same product group (fallback: all) and, for the seasonal variant, launched within +-45 days of the item's launch day-of-year a
year earlier or in the same 2025/2024 season window (fallback: same group).
ratio_p = (units in days a+1..a+122 of the peer's life + 1) / (units in days 1..a + 1); analog = (S + 1) x exp(median log ratio) - 1.
Compared on young item-origins (age < 3 months, the archetype carrying 21% of the error) and blended geometrically with the three-way."""
import numpy as np, pandas as pd, sys, importlib.util
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
spec = importlib.util.spec_from_file_location("sl", "eda/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([T.origin, T.style_color]))
LQ = np.log1p(q[["q10", "q25", "q50", "q75", "q90"]].values); D = {c: LQ[:, j] - LQ[:, 2] for j, c in enumerate(["q10", "q25", "q50", "q75", "q90"])}
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
DD = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(0).asfreq("D").fillna(0); C = DD.cumsum()
first = (DD > 0).idxmax(); first = first[first > pd.Timestamp("2024-01-15")]          # launches observed from day 1
macro = T.drop_duplicates("style_color").set_index("style_color").macro
def cum(s, t): return C.at[min(t, C.index[-1]), s] if t >= C.index[0] else 0.0
an, ans = [], []
for o, s, age in zip(T.origin, T.style_color, T.age_m):
    E = pd.Period(o, "M").end_time.normalize()
    if s not in first.index or age >= 3 or first[s] > E: an.append(np.nan); ans.append(np.nan); continue
    a = (E - first[s]).days + 1; S = cum(s, E)
    P = first[(first <= E - pd.Timedelta(days=a + 122)) & (first >= E - pd.Timedelta(days=540))]
    def ratios(peers):
        r = []
        for p, f0 in peers.items():
            s0 = cum(p, f0 + pd.Timedelta(days=a - 1)); s1 = cum(p, f0 + pd.Timedelta(days=a + 121)) - s0
            r.append(np.log((s1 + 1) / (s0 + 1)))
        return r
    pg = P[P.index.map(macro).values == macro.get(s)] if len(P) else P
    base = pg if len(pg) >= 10 else P
    r = ratios(base); an.append((S + 1) * np.exp(np.median(r)) - 1 if len(r) >= 10 else np.nan)
    doy = first[s].dayofyear; ps = base[np.abs(((base.dt.dayofyear - doy + 182) % 365) - 182) <= 45]
    r2 = ratios(ps) if len(ps) >= 10 else r; ans.append((S + 1) * np.exp(np.median(r2)) - 1 if len(r2) >= 10 else np.nan)
T["analog"] = an; T["analog_sea"] = ans
Y = T[T.analog.notna()].copy(); print("young item-origins with an analog:", len(Y), "of", int((T.age_m < 3).sum()))
for c in ["analog", "analog_sea"]:
    Y[f"3way+{c}"] = np.exp((np.log(Y.r_q_nn + 1) + np.log(Y[c] + 1)) / 2) - 1
    Y[f"4way {c}"] = np.exp((np.log(Y.rules.clip(lower=.5)) + np.log(Y.qmodel.clip(lower=.5)) + np.log(Y.nn.clip(lower=.5)) + np.log(Y[c].clip(lower=.5))) / 4)
Y = Y.reset_index(drop=True); Dy = {k: v[T.analog.notna().values] for k, v in D.items()}
COLS = ["qmodel", "r_q_nn", "analog", "analog_sea", "3way+analog", "3way+analog_sea", "4way analog", "4way analog_sea"]
REC = [o for o in sorted(T.origin.unique()) if o >= "2025-06"]
for nm, Z in [("young items, nine recent origins", Y[Y.origin.isin(REC)]), ("young items, all origins", Y)]:
    print(f"\n######## {nm}"); print(sl.table(Z, COLS, Dy).round(4).to_string())
Z = Y[Y.origin.isin(REC)]; print("\nbootstrap vs three-way, young recent:", {c: np.round(sl.boot(Z, c, "r_q_nn"), 4).tolist() for c in ["3way+analog", "3way+analog_sea", "4way analog", "4way analog_sea"]})
T.to_parquet("data/launch_analog_items.parquet", index=False)
