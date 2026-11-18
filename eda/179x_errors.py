"""D244 (user: "deep error analysis: what are the items we ain't good at, why and when"), final system F = p2c32a (data/dec_base.parquet).
1 Concentration: share of absolute error held by the worst x% of item-origins; error share vs unit share.
2 Who: WAPE / R3, bias and share of total error by macro group, product type, age band, volume band, Black Friday window, member spread,
  TikTok share, discount level, seasonality strength (|own_sea|), lifecycle phase (trend91), and the 100 targets.
3 When: by origin, and by target calendar month (monthly F = the 4-month total split by the LightGBM monthly shape).
4 Persistence: is the same item bad again? Spearman of |e| between the item's origins 4 months apart (non-overlapping windows), sign
  persistence of e, and the share of chronic items (|e| > 0.5 at >= half of their >= 5 origins).
5 Cascade diagnosis of each large miss (|e| > 0.5, y >= 20): consensus miss (actual outside the members' range: every member wrong on the
  same side), combination miss (actual inside the range: some member was right), and whether the seasonality layer moved toward or away
  from the actual (|log(y / season)| vs |log(y / R3)|).
6 Gallery: the 12 largest absolute misses (recent), monthly history and forecast -> eda/figs/err_final_worst.png.
Writes eda/errors_deep_output.txt (stdout) and data/errors_deep_items.parquet."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base.parquet"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]
Fe = pd.read_parquet("data/features.parquet"); se = Fe.assign(st=Fe.m3 * np.exp(Fe.lfl_type.fillna(0))).groupby(["origin", "style_color"]).st.sum().rename("season").reset_index()
B = B.merge(se, on=["origin", "style_color"], how="left")
R = B[B.origin >= "2025-06"].copy(); E = R.ae.sum(); R3E = (R.R3 - R.y).abs().sum()
def seg(col, bins=None, labels=None):
    x = pd.cut(R[col], bins, labels=labels) if bins is not None else R[col]
    g = R.groupby(x, observed=True)
    t = pd.DataFrame({"item_origins": g.size(), "unit_share": g.y.sum() / R.y.sum(), "error_share": g.ae.sum() / E,
                      "wape_r3": g.ae.sum() / g.apply(lambda d: (d.R3 - d.y).abs().sum()), "bias": g.F.sum() / g.y.sum() - 1})
    return t.round(3)
print("== 1 concentration (recent origins, 9,031 item-origins)")
s = R.ae.sort_values(ascending=False).cumsum() / E
for p in (0.01, 0.05, 0.10, 0.20): print(f"  worst {p:.0%} of item-origins hold {s.iloc[int(len(s) * p) - 1]:.1%} of the error")
print("\n== 2 who (recent): share of units vs share of error, WAPE / R3, bias")
R["age_band"] = pd.cut(R.age_m, [-1, 3, 6, 12, 24, 999], labels=["<3m", "3-6m", "6-12m", "12-24m", "24m+"])
R["vol_band"] = pd.qcut(np.log1p(R.R3), 5, labels=["v1 smallest", "v2", "v3", "v4", "v5 largest"])
R["spread_band"] = pd.qcut(R.spread, 5, labels=["agree", "q2", "q3", "q4", "disagree"])
R["tt_band"] = pd.cut(R.sh_tt.fillna(0), [-1, 0.02, 0.1, 0.25, 1.1], labels=["<2%", "2-10%", "10-25%", ">25%"])
R["disc_band"] = pd.cut(R.disc_item, [-1, 0.4, 0.55, 0.65, 1.1], labels=["<40%", "40-55%", "55-65%", ">65%"])
R["sea_band"] = pd.cut(R.own_sea.abs(), [-1, 0.2, 0.5, 1.0, 99], labels=["flat", "mild", "seasonal", "very seasonal"])
R["phase"] = pd.cut(R.trend91, [-99, -0.4, -0.1, 0.1, 0.4, 99], labels=["falling fast", "falling", "steady", "rising", "rising fast"])
for c in ["macro", "age_band", "vol_band", "bf_window", "spread_band", "tt_band", "disc_band", "sea_band", "phase", "target"]:
    print(f"-- {c}"); print(seg(c).to_string())
tp = seg("ptype").sort_values("error_share", ascending=False).head(12); print("-- product type (12 largest error shares)"); print(tp.to_string())
print("\n== 3 when: by origin")
print(B.groupby("origin").apply(lambda d: pd.Series({"wape_r3": d.ae.sum() / (d.R3 - d.y).abs().sum(), "bias": d.F.sum() / d.y.sum() - 1, "units": d.y.sum()})).round(3).to_string())
BT = pd.read_parquet("data/best_backtest_preds.parquet")[["origin", "style_color", "h", "pred"]]
sh = BT.assign(sh=BT.pred / BT.groupby(["origin", "style_color"]).pred.transform("sum").clip(lower=1e-9))
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
mm = sh.merge(B[["origin", "style_color", "F"]], on=["origin", "style_color"]); mm["Fm"] = mm.F * mm.sh
mm["tm"] = [str(pd.Period(o, "M") + h) for o, h in zip(mm.origin, mm.h)]; mm["ym"] = [M.at[s, t] if t in M.columns and s in M.index else np.nan for s, t in zip(mm.style_color, mm.tm)]
mm = mm[mm.origin >= "2025-06"].dropna(subset=["ym"]); mm["cal"] = mm.tm.str[-2:]
print("-- by target calendar month (recent origins): item-month WAPE and bias")
print(mm.groupby("cal").apply(lambda d: pd.Series({"item_month_wape": (d.Fm - d.ym).abs().sum() / d.ym.sum(), "bias": d.Fm.sum() / d.ym.sum() - 1, "units": d.ym.sum()})).round(3).to_string())
print("\n== 4 persistence: is the same item bad again?")
P = B[B.y >= 20][["style_color", "origin", "e"]].copy(); P["o"] = pd.PeriodIndex(P.origin, freq="M").astype(int)
Q = P.merge(P.assign(o=P.o + 4), on=["style_color", "o"], suffixes=("", "_lag4"))
print(f"  |e| rank correlation, same item 4 months apart: {Q[['e', 'e_lag4']].abs().corr(method='spearman').iloc[0, 1]:+.3f} | sign repeats {(np.sign(Q.e) == np.sign(Q.e_lag4)).mean():.3f} | e correlation {Q[['e', 'e_lag4']].corr().iloc[0, 1]:+.3f}")
ch = P.groupby("style_color").agg(n=("e", "size"), bad=("e", lambda v: (v.abs() > 0.5).mean())); ch = ch[ch.n >= 5]
print(f"  chronic items (|e| > 0.5 at >= half of >= 5 origins): {(ch.bad >= 0.5).sum()} of {len(ch)} ({(ch.bad >= 0.5).mean():.1%})")
print("\n== 5 cascade diagnosis of large misses (recent, |e| > 0.5, y >= 20)")
L = R[(R.e.abs() > 0.5) & (R.y >= 20)].copy(); lo = L[MEM].min(1); hi = L[MEM].max(1)
L["kind"] = np.where(L.y > hi, "consensus under (all members below)", np.where(L.y < lo, "consensus over (all members above)", "combination (actual inside the members' range)"))
L["season_dir"] = np.where(np.abs(np.log((L.y + 1) / (L.season + 1))) < np.abs(np.log((L.y + 1) / (L.R3 + 1))), "seasonality moved toward the actual", "seasonality moved away")
print(f"  large misses: {len(L)} item-origins, {L.ae.sum() / E:.1%} of all error")
print(L.groupby("kind").agg(n=("e", "size"), error_share=("ae", lambda v: v.sum() / E)).round(3).to_string())
print(L.groupby(["kind", "season_dir"]).ae.sum().div(E).round(3).to_string())
print("  which member was closest in the combination misses:", L[L.kind.str.startswith("combination")][MEM].sub(L[L.kind.str.startswith("combination")].y, axis=0).abs().idxmin(axis=1).value_counts(normalize=True).round(3).to_dict())
R.to_parquet("data/errors_deep_items.parquet", index=False)
W = R.sort_values("ae", ascending=False).head(12); fig, axs = plt.subplots(3, 4, figsize=(17, 10)); it = pd.read_parquet("data/items.parquet").set_index("style_color")
for ax, (_, r) in zip(axs.ravel(), W.iterrows()):
    ms = [str(pd.Period(r.origin, "M") + k) for k in range(-12, 5)]; ys = M.reindex(columns=ms).loc[r.style_color].values
    ax.plot(range(len(ms)), ys, marker="o", ms=3, color="k", label="actual"); fx = mm[(mm.origin == r.origin) & (mm.style_color == r.style_color)].sort_values("h")
    ax.plot(range(13, 13 + len(fx)), fx.Fm.values, marker="s", ms=3, color="tab:red", label="forecast"); ax.axvline(12.5, color="grey", lw=0.6)
    ax.set_title(f"{r.style_color} ({str(it.product_type.get(r.style_color, ''))[:12]}) @ {r.origin}\nF {r.F:,.0f} vs y {r.y:,.0f}; members {r[MEM].min():,.0f}-{r[MEM].max():,.0f}", fontsize=8)
    ax.set_xticks([0, 6, 12, 16]); ax.set_xticklabels([ms[0], ms[6], ms[12], ms[16]], fontsize=7)
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig("eda/figs/err_final_worst.png", dpi=110); print("\ngallery saved")
