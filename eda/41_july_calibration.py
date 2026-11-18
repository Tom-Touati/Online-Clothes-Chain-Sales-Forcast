"""D061 July-share calibration. At the June 2025 origin the production method got Jul-Oct totals right (-0.4%) but under-allocated July:
predicted vs actual July share of Jul-Oct by age group 12m+ 0.282 vs 0.295, 3-12 months 0.273 vs 0.271, launch <3m 0.297 vs 0.335.
Correction: July share x (actual / predicted share) for the part of each item's shape that comes from the monthly model
(full factor where the shape is the model profile; half where it is 50/50 with the item's own 2025 shape; none for segment shapes,
World Cup items and S16079). Item totals are unchanged: August-October are scaled down proportionally."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
O = pd.read_csv("data/final_forecast_q3.csv", index_col=0)
M = pd.read_parquet("data/monthly_item.parquet")
FACTOR = {"12m+": 0.295 / 0.282, "3-12 months": 0.271 / 0.273, "launch <3m": 0.335 / 0.297}      # from eda/40_forecast_checks.py
WC = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW"]
grp = np.where(O.age_m < 3, "launch <3m", np.where(O.age_m < 12, "3-12 months", "12m+"))
seg_shape = O.seg.isin(["swimwear", "Homecoming", "Halloween"]) & ~O.mature
w = np.where(O.index.isin(WC + ["S16079_DARK-GRAY"]) | seg_shape, 0.0, np.where(O.mature, 0.5, 1.0))
f = pd.Series([1 + wi * (FACTOR[g] - 1) for wi, g in zip(w, grp)], index=O.index)
X = O[["f_jul", "f_aug", "f_sep", "f_oct"]].astype(float)
tot = X.sum(axis=1); jul_new = X.f_jul * f
rest = X[["f_aug", "f_sep", "f_oct"]].mul((tot - jul_new) / X[["f_aug", "f_sep", "f_oct"]].sum(axis=1), axis=0)
N = pd.concat([jul_new.rename("f_jul"), rest], axis=1).round(0)
N["f_oct"] += tot - N.sum(axis=1)                                                      # keep the item total exact after rounding
out = O.copy(); out[["f_jul", "f_aug", "f_sep", "f_oct"]] = N.values; out["july_factor"] = f.values
assert (out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum(axis=1) == O.final_total).all()
out.to_csv("data/final_forecast_q4.csv")
jun = M.loc[O.index, "2026-06"]
print("July/June before", round(O.f_jul.sum() / jun.sum(), 3), "-> after", round(out.f_jul.sum() / jun.sum(), 3), "| monthly totals:", out[["f_jul", "f_aug", "f_sep", "f_oct"]].sum().astype(int).to_dict(), "| total", int(out.final_total.sum()))
for g in ["12m+", "3-12 months", "launch <3m"]:
    s = O.index[grp == g]; print(f"{g}: July/June {O.loc[s, 'f_jul'].sum() / jun[s].sum():.3f} -> {out.loc[s, 'f_jul'].sum() / jun[s].sum():.3f} | July share {O.loc[s, 'f_jul'].sum() / O.loc[s, 'final_total'].sum():.3f} -> {out.loc[s, 'f_jul'].sum() / out.loc[s, 'final_total'].sum():.3f}")
print("items changed:", int((f != 1).sum()), "| largest July changes:", (out.f_jul - O.f_jul).sort_values().iloc[[0, -1, -2, -3]].astype(int).to_dict())
