"""Does the quantile model under-use an item's OWN seasonal pattern from last year?
z_act  = log((actual 4-month total + 1) / (4 x level + 1))
z_med  = same for the model's median;  z_own = item's own ratio last year: mean over the 4 target months of
         (units in that month last year + 5) / (mean of the 3 months ending a year before the origin + 5), in logs. Only items live then.
Test: slope of (z_act - z_med) on (z_own - z_med). A positive slope means the model shrinks the item's own seasonality too much.
Correction: shift every quantile by w x (z_own - z_med); w fitted on earlier origins only, then tau re-chosen on the representative window."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest.parquet")
M = pd.read_parquet("data/monthly_item.parquet"); cols = list(M.columns)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
def z_own(o, items):
    k = cols.index(o); base = [cols[k - 14], cols[k - 13], cols[k - 12]] if k >= 14 else None
    if base is None: return pd.Series(np.nan, index=items)
    tgt = [cols[k - 12 + h] for h in range(1, 5)]
    live = start.reindex(items) < pd.Period(base[0]).start_time
    r = np.log((M.loc[items, tgt].values + 5) / (M.loc[items, base].mean(axis=1).values[:, None] + 5)).mean(axis=1)
    return pd.Series(np.where(live.values, r, np.nan), index=items)
Q["z_own"] = np.nan
for o, g in Q.groupby("origin"): Q.loc[g.index, "z_own"] = z_own(o, g.style_color.values).values
lv = 4 * Q.level + 1
Q["z_act"] = np.log((Q.ytot + 1) / lv); Q["z_med"] = np.log((Q.q50 + 1) / lv); Q["gap"] = Q.z_own - Q.z_med; Q["res"] = Q.z_act - Q.z_med
Q["oo"] = Q.origin.map(lambda s: pd.Period(s, freq="M").ordinal)
REP = [str(p) for p in pd.period_range("2025-06", "2026-02", freq="M")]
H = Q[Q.z_own.notna()]
print("item-origins with own last-year ratio:", len(H), "of", len(Q), "| representative window:", int(H.origin.isin(REP).sum()))
for nm, g in [("all origins", H), ("representative", H[H.origin.isin(REP)])]:
    b = np.polyfit(g.gap, g.res, 1); c = np.corrcoef(g.gap, g.res)[0, 1]
    print(f"{nm}: slope of residual on (own - model) = {b[0]:.3f} (intercept {b[1]:.3f}), corr {c:.3f}")
    print("   residual by quintile of (own - model):", g.groupby(pd.qcut(g.gap, 5, labels=False)).agg(gap=("gap", "median"), res=("res", "median"), n=("res", "size")).round(3).to_dict("list"))
print("by origin slope:", {o: round(np.polyfit(g.gap, g.res, 1)[0], 3) for o, g in H.groupby("origin") if len(g) > 50})

def q_at(Lq, tau, k=1.0):
    med = Lq[:, 3:4]; Ls = med + k * (Lq - med); out = np.empty(len(Lq))
    for i in range(len(Lq)):
        out[i] = np.interp(tau, QS, Ls[i]) if QS[0] <= tau <= QS[-1] else (Ls[i, 0] if tau < QS[0] else Ls[i, -1])
    return out
def met(f, a):
    ape = np.abs(f - a) / np.clip(a, 1, None); return dict(wape=np.abs(f - a).sum() / a.sum(), bias=f.sum() / a.sum() - 1, w20=(ape <= .2).mean(), w50=(ape <= .5).mean())

# out-of-sample: w from earlier origins (origin + 4 <= o), applied to later; no shift for items without own history
rows = []
for o in sorted(Q.origin.unique()):
    oo = pd.Period(o).ordinal; past = H[H.oo + 4 <= oo]
    if past.origin.nunique() < 2: continue
    w = float(np.clip(np.polyfit(past.gap, past.res, 1)[0], 0, 1))
    g = Q[Q.origin == o]; sh = np.where(g.z_own.notna(), w * g.gap.fillna(0), 0)
    L0 = np.log1p(g[QC].values); L1 = np.log1p(np.clip(np.expm1(L0) * np.exp(sh)[:, None] + np.exp(sh)[:, None] - 1, 0, None))
    for tau in (0.5, 0.61):
        rows.append(dict(origin=o, w=w, tau=tau, version="current", **met(np.expm1(q_at(L0, tau)), g.ytot.values)))
        rows.append(dict(origin=o, w=w, tau=tau, version="own-season shift", **met(np.expm1(q_at(L1, tau)), g.ytot.values)))
R = pd.DataFrame(rows)
print(f"\n== out-of-sample: w fitted on earlier origins, {R.origin.nunique()} later origins")
print("w by origin:", R.drop_duplicates("origin").set_index("origin").w.round(3).to_dict())
print(R.groupby(["tau", "version"])[["wape", "bias", "w20", "w50"]].mean().round(3).to_string())
pw = R.pivot_table(index=["origin", "tau"], columns="version", values="wape"); print("origins where the shift lowers WAPE (tau 0.61):", int((pw.xs(0.61, level="tau")["own-season shift"] < pw.xs(0.61, level="tau")["current"]).sum()), "of", R.origin.nunique())
# only items that have own history (where the shift acts)
rows = []
for o in sorted(Q.origin.unique()):
    oo = pd.Period(o).ordinal; past = H[H.oo + 4 <= oo]
    if past.origin.nunique() < 2: continue
    w = float(np.clip(np.polyfit(past.gap, past.res, 1)[0], 0, 1)); g = H[H.origin == o]
    L0 = np.log1p(g[QC].values); sh = w * g.gap.values; L1 = np.log1p(np.clip((g[QC].values + 1) * np.exp(sh)[:, None] - 1, 0, None))
    rows.append(dict(origin=o, version="current", **met(np.expm1(q_at(L0, .61)), g.ytot.values))); rows.append(dict(origin=o, version="own-season shift", **met(np.expm1(q_at(L1, .61)), g.ytot.values)))
print("items with own history only, tau 0.61:"); print(pd.DataFrame(rows).groupby("version")[["wape", "bias", "w20", "w50"]].mean().round(3).to_string())
wR = float(np.clip(np.polyfit(H[H.origin.isin(REP)].gap, H[H.origin.isin(REP)].res, 1)[0], 0, 1)); wA = float(np.clip(np.polyfit(H.gap, H.res, 1)[0], 0, 1))
print(f"\nw on representative window: {wR:.3f} | w on all 13 origins: {wA:.3f}")
Q.to_parquet("data/q_backtest_own.parquet")
