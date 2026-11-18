"""D197 (fork): inputs for Chronos-2 with group context and known future covariates (eda/184 model "chronos2mv", remote).
Daily arrays 2024-01-01 .. 2026-12-31:
  type_sales / macro_sales   units of each item's product type / macro group (all items of that group), per day
  ev_bfcm, ev_other          Black Friday-Cyber Monday days, and the sum of the other event windows' daily intensities (events_month
                             counts spread evenly over their month), known in advance
  t_norm                     sales-weighted national 2014-2023 normal temperature of the day's calendar month (known in advance)
Writes data/fk_fm_cov.npz (items, group indices, the group series and the covariates)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
Z = np.load("data/fk_daily.npz"); items = Z["items"]; Y = Z["Y"]
F0 = pd.read_parquet("data/features.parquet", columns=["style_color", "ptype", "macro"]).drop_duplicates("style_color").set_index("style_color").reindex(items)
pt = F0.ptype.astype(str).fillna("na").values; mc = F0.macro.astype(str).fillna("na").values
ptu, pti = np.unique(pt, return_inverse=True); mcu, mci = np.unique(mc, return_inverse=True)
TS = np.stack([Y[pti == k].sum(0) for k in range(len(ptu))]); MS = np.stack([Y[mci == k].sum(0) for k in range(len(mcu))])
days = pd.date_range("2024-01-01", "2026-12-31"); m = days.to_period("M").astype(str); dim = days.days_in_month.values
E = pd.read_parquet("data/events_month.parquet").set_index("m")
bf = E.bfcm.reindex(m).fillna(0).values / dim; oth = E.drop(columns=["bfcm", "prime_day"]).sum(1).reindex(m).fillna(0).values / dim
W = pd.read_parquet("data/weather_state_month.parquet"); S = pd.read_parquet("data/state_item_month.parquet")
sw = S.groupby("state").units.sum(); sw = sw / sw.sum()
tn = W[W.state.isin(sw.index)].assign(w=lambda d: d.state.map(sw)).groupby("cm").apply(lambda d: (d.tmean_norm * d.w).sum() / d.w.sum())
t_norm = tn.reindex(days.month).values
pad = len(days) - Y.shape[1]
np.savez_compressed("data/fk_fm_cov.npz", items=items, pti=pti, mci=mci, TS=np.pad(TS, ((0, 0), (0, pad))).astype(np.float32), MS=np.pad(MS, ((0, 0), (0, pad))).astype(np.float32),
                    ev_bfcm=bf.astype(np.float32), ev_other=oth.astype(np.float32), t_norm=t_norm.astype(np.float32), ndays_obs=np.array(Y.shape[1]))
print(len(ptu), "types,", len(mcu), "macro groups; covariate days", len(days), "| t_norm range", t_norm.min().round(1), t_norm.max().round(1))
