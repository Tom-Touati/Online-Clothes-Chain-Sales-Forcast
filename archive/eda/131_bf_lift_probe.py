"""Can an item's Black Friday lift be predicted from what is known at the end of October? (D137)
Lift = units in the Thanksgiving..Cyber Monday days (as a weekly rate) / mean weekly units over the 4 weeks before. Features at the October
origin (data/features.parquet, h = 1 rows): channel shares, discount and relative discount, age, level, pace, trend, peak ratio, type,
group, occasion flags, colour number, own seasonal ratio. Time-honest: LightGBM trained on 2024 lifts (October 2024 features), tested on
2025 lifts (October 2025 features). Reports the correlation of predicted and actual log lift, and how much of the 2025 Black Friday
under-forecast an item-level lift multiplier would recover inside the three-way (October 2025 origin, 4-month totals)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)
DD = d.groupby(["style_color", "ny_date"]).qty.sum().unstack(0).asfreq("D").fillna(0)
BF = {2024: pd.Timestamp("2024-11-28"), 2025: pd.Timestamp("2025-11-27")}
def lifts(y):
    t = BF[y]; wk = DD.loc[t:t + pd.Timedelta(days=4)].sum() / 5 * 7; base = DD.loc[t - pd.Timedelta(days=28):t - pd.Timedelta(days=1)].mean() * 7
    return np.log((wk + 1) / (base + 1))[base >= 3]                      # items selling at least ~3 a week before
F = pd.read_parquet("data/features.parquet"); F = F[F.h == 1]
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
X = [c for c in ["sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "age_m", "lvl", "f_r7", "f_r28", "trend91", "peak_r", "ptype", "macro", "color_num", "own_sea", "lfl_type"]
     + [c for c in F.columns if c.startswith("occ_")] if c in F.columns]
tr = F[F.origin == "2024-10"].set_index("style_color"); te = F[F.origin == "2025-10"].set_index("style_color")
l24, l25 = lifts(2024), lifts(2025)
tr = tr.join(l24.rename("lift"), how="inner"); te = te.join(l25.rename("lift"), how="inner")
print("items: train (2024)", len(tr), "| test (2025)", len(te), "| mean log lift 2024", round(tr.lift.mean(), 3), "2025", round(te.lift.mean(), 3))
m = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0).fit(tr[X], tr.lift)
p = m.predict(te[X])
r = np.corrcoef(p, te.lift)[0, 1]; mse0 = ((te.lift - tr.lift.mean()) ** 2).mean(); mse1 = ((te.lift - p) ** 2).mean()
print(f"2025 test: corr(predicted, actual log lift) {r:.3f} | MSE vs the 2024 mean lift {mse1:.4f} vs {mse0:.4f} (R2 vs that {1 - mse1 / mse0:.3f})")
own = te.index.map(l24).values; ok = ~np.isnan(own.astype(float))
print(f"items with their own 2024 lift ({ok.sum()}): corr(own 2024 lift, 2025 lift) {np.corrcoef(own[ok].astype(float), te.lift[ok])[0, 1]:.3f} | corr(model, 2025) on them {np.corrcoef(p[ok], te.lift[ok])[0, 1]:.3f}")
imp = pd.Series(m.booster_.feature_importance("gain"), index=X).sort_values(ascending=False); print("top drivers:", (imp / imp.sum()).head(8).round(3).to_dict())
print("2025 lift by predicted-lift quintile (mean actual log lift):", pd.Series(te.lift.values).groupby(pd.qcut(p, 5, labels=False)).mean().round(3).to_dict())
