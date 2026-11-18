"""D288, D289 (fork; user comments on the error study).
A  "is it specific times that yield an error on a lot of items or specific items?" (and: "if it's specific times, we can add statistics
   like how much on average do US fashion ecommerce websites rise at this time of year")
   The log miss of the submitted system split into: the time (everything at an origin moving together), the product type at that
   time, the item whenever it is forecast (persistent), and the item in one window. Variance components by one-way random-effects
   ANOVA; the persistent item part from windows that share no month (4-apart sets). In units: the score with totals known at each
   level; an item's own past miss used as a correction (time-honest); how many items and origins the worst forecasts are.
B  "we should show it both normalized by volume and not / for the evaluation sake, we need to see it normalized by the total"
   The same forecasts scored in units (WAPE, large items dominate) and per item in percentage terms (each item once: median and
   mean percentage error, share within +/-20%, mean absolute log error), for R3, R3 + rules and the system; concentration both ways;
   by volume fifth.
C  "some model features should also be given normalized and the loss in some cases shouldnt be on absolute items but on percentage
   wrong"  What the members use, and the stored tree variants along the scale from a percentage loss with every item equal to a loss
   on units (D206: rows weighted by volume^0.3 / 0.5 / 1, Tweedie on units), scored both ways.
Adds J["timeitem"], J["bothways"], J["lossscale"] to data/fk_err_analysis.json; stdout -> eda/time_item_percent_output.txt."""
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); ORIG = sorted(D.origin.unique()); REC = [o for o in ORIG if o >= "2025-06"]; D["rec"] = D.origin.isin(REC); D["oo"] = D.origin.map(lambda o: pd.Period(o, "M").ordinal)
J = json.load(open("data/fk_err_analysis.json")); y = D.y.values; F = D.F.values; ar = (D.R3 - D.y).abs().values; rec = D.rec.values; D["res"] = np.log((D.y + 1) / (D.F + 1)); D["ae"] = np.abs(F - y)
hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum())

# ---------- A time or item
def icc(df, key, val="res"):
    g = df.groupby(key)[val]; n = g.size().values; k = len(n); N = n.sum(); m = g.mean().values; gm = df[val].mean()
    msb = (n * (m - gm) ** 2).sum() / (k - 1); msw = ((df[val] - g.transform("mean")) ** 2).sum() / (N - k); n0 = (N - (n ** 2).sum() / N) / (k - 1); sb = max((msb - msw) / n0, 0); return sb / (sb + msw)
R = D[D.rec].copy(); R["ot"] = R.origin + "|" + R.ptype.astype(str); R["og"] = R.origin + "|" + R.macro.astype(str); R["otl"] = R.ot + "|" + R.look.astype(str)
time_share = icc(R, "origin"); type_share = icc(R, "ot"); group_share = icc(R, "og"); cell_share = icc(R, "otl")
# persistent item part: within each set of origins 4 months apart (no shared months), after removing the origin mean
D["res_c"] = D.res - D.groupby("origin").res.transform("mean"); it_shares = []
for k in range(4):
    Z = D[D.origin.isin(ORIG[k::4])]; Z = Z[Z.groupby("style_color").res_c.transform("size") >= 2]; it_shares.append(icc(Z, "style_color", "res_c"))
item_share = float(np.mean(it_shares)); R["res_c"] = R.res - R.groupby("origin").res.transform("mean"); item_overlap = icc(R[R.groupby("style_color").res_c.transform("size") >= 2], "style_color", "res_c")
def scaled(keys):
    g = D.groupby(keys); return F * g.y.transform("sum").values / np.maximum(g.F.transform("sum").values, 1e-9)
D["tl"] = D.ptype.astype(str) + "|" + D.look.astype(str); orc = []
for lab, keys in [("each origin's total", ["origin"]), ("each product group's total at each origin", ["origin", "macro"]), ("each product type's total at each origin", ["origin", "ptype"]), ("each type × look total at each origin", ["origin", "tl"])]:
    f = scaled(keys); orc.append(dict(known=lab, wape=r(wape(f)), hit=r(hit(f), 3), cut=r(1 - wape(f) / wape(F), 3), groups=int(D[D.rec].groupby(keys).ngroups / len(REC))))
# an item's own past miss as a correction (time-honest: origins whose windows have closed)
def past_corr(k, keep_total=True):
    f = F.copy()
    for o in REC:
        past = D[D.oo + 4 <= pd.Period(o).ordinal]; te = np.where((D.origin == o).values)[0]
        if not len(past): continue
        m = past.groupby("style_color").res_c.mean().reindex(D.style_color.values[te]).fillna(0).values; v = (F[te] + 1) * np.exp(k * m) - 1; f[te] = v * F[te].sum() / v.sum() if keep_total else v
    return f
pc_rows = [dict(k=k, wape=r(wape(past_corr(k))), hit=r(hit(past_corr(k)), 3)) for k in (0.25, 0.5, 1.0)]
# how many items and origins the worst forecasts are; episodes = runs of consecutive origins of one item
Rr = D[D.rec].sort_values("ae", ascending=False); n5 = int(round(0.05 * len(Rr))); W = Rr.head(n5); W100 = Rr.head(100)
def episodes(Z):
    e = 0
    for _, g in Z.groupby("style_color"):
        o = np.sort(g.oo.values); e += 1 + int((np.diff(o) > 1).sum())
    return e
by_item = D[D.rec].groupby("style_color").agg(ae=("ae", "sum"), y=("y", "sum"), n=("ae", "size")).sort_values("ae", ascending=False); ni = len(by_item); tot_ae = by_item.ae.sum(); tot_y = by_item.y.sum()
top_items = [dict(top=p_, items=int(round(p_ * ni)), err=r(by_item.ae.head(int(round(p_ * ni))).sum() / tot_ae, 3), units=r(by_item.y.head(int(round(p_ * ni))).sum() / tot_y, 3)) for p_ in (0.01, 0.05, 0.10)]
po = []
for o in REC:
    Z = D[D.origin == o]; f1 = Z.F * Z.y.sum() / Z.F.sum(); po.append(dict(origin=o, bias=r(Z.F.sum() / Z.y.sum() - 1, 3), under=r(float((Z.F < Z.y).mean()), 3), abs=r(Z.ae.sum() / Z.y.sum(), 3), cut=r(1 - (f1 - Z.y).abs().sum() / Z.ae.sum(), 3), worst5=r(float(Z.style_color.isin(W[W.origin == o].style_color).mean()), 3), n_worst=int((W.origin == o).sum())))
TI = dict(time=r(time_share, 3), type_time=r(type_share, 3), group_time=r(group_share, 3), cell_time=r(cell_share, 3), item=r(item_share, 3), item_sets=[r(v, 3) for v in it_shares], item_overlap=r(item_overlap, 3), oracles=orc, past_corr=pc_rows, system=r(wape(F)), system_hit=r(hit(F), 3),
          worst5=dict(n=n5, items=int(W.style_color.nunique()), origins=int(W.origin.nunique()), episodes=episodes(W), err=r(W.ae.sum() / Rr.ae.sum(), 3)), worst100=dict(items=int(W100.style_color.nunique()), origins=int(W100.origin.nunique()), episodes=episodes(W100), err=r(W100.ae.sum() / Rr.ae.sum(), 3)),
          top_items=top_items, n_items=ni, by_origin=po, under_range=[min(d["under"] for d in po), max(d["under"] for d in po)], bias_range=[min(d["bias"] for d in po), max(d["bias"] for d in po)])
print("A  time or item:"); [print("  ", k, v) for k, v in TI.items() if k != "by_origin"]; print(pd.DataFrame(po).to_string(index=False))

# ---------- B both ways
ape = lambda f: np.abs(f - y) / np.maximum(y, 1); male = lambda f, m: float(np.abs(np.log((f[m] + 1) / (y[m] + 1))).mean())
def score(f, m=rec):
    a = ape(f)[m]; e = np.abs(f - y)[m]; o = np.argsort(-e); oa = np.argsort(-np.minimum(a, 2)); n5_ = int(round(0.05 * m.sum()))
    return dict(wape_abs=r(e.sum() / y[m].sum(), 3), ratio=r(e.sum() / ar[m].sum(), 4), bias=r(f[m].sum() / y[m].sum() - 1, 3), med_ape=r(float(np.median(a)), 3), mean_ape=r(float(np.minimum(a, 2).mean()), 3), hit=r(float((e <= 0.2 * y[m]).mean()), 3),
                hit_units=r(float(y[m][e <= 0.2 * y[m]].sum() / y[m].sum()), 3), male=r(male(f, m), 3), top5_units=r(e[o][:n5_].sum() / e.sum(), 3), top5_pct=r(np.minimum(a, 2)[oa][:n5_].sum() / np.minimum(a, 2).sum(), 3))
SYS = [("R3", D.R3.values), ("R3 + rules", D.rules.values), ("Submitted system", F)]; BW = []
for lab, f in SYS:
    d = score(f); d["system"] = lab; BW.append(d)
for d in BW: d["male_ratio"] = r(d["male"] / BW[0]["male"], 3); d["med_ape_ratio"] = r(d["med_ape"] / BW[0]["med_ape"], 3); d["mean_ape_ratio"] = r(d["mean_ape"] / BW[0]["mean_ape"], 3)
D["fifth"] = D.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)); fif = []
for k in range(5):
    m = rec & (D.fifth.values == k); a = np.minimum(ape(F), 2); a3 = np.minimum(ape(D.R3.values), 2)
    fif.append(dict(fifth=k + 1, med_level=r(float(np.median(D.level.values[m])), 0), wape_abs=r(D.ae.values[m].sum() / y[m].sum(), 3), wape_abs_r3=r(ar[m].sum() / y[m].sum(), 3), med_ape=r(float(np.median(ape(F)[m])), 3), med_ape_r3=r(float(np.median(ape(D.R3.values)[m])), 3),
                    hit=r(hit(F, m), 3), err_units=r(D.ae.values[m].sum() / D.ae.values[rec].sum(), 3), err_pct=r(a[m].sum() / a[rec].sum(), 3), units=r(y[m].sum() / y[rec].sum(), 3)))
J["bothways"] = dict(systems=BW, fifths=fif); print("\nB  both ways:"); print(pd.DataFrame(BW).set_index("system").to_string()); print(pd.DataFrame(fif).to_string(index=False))

# ---------- C the loss, from percentage with every item equal to units
VARF = [("Percentage loss, every item counts the same (D198; how the members are trained)", "server_backup/edikted_fk/out/q_backtest_gb_base.parquet"), ("Percentage loss, items weighted by volume^0.3", "server_backup/edikted_fk/out/q_backtest_gb_wvol03.parquet"),
        ("Percentage loss, items weighted by volume^0.5", "server_backup/edikted_fk/out/q_backtest_gb_wvol05.parquet"), ("Percentage loss, items weighted by volume", "server_backup/edikted_fk/out/q_backtest_gb_wvol1.parquet"),
        ("Loss on units (Tweedie)", "server_backup/edikted_fk/out/q_backtest_gb_tw15.parquet"), ("New features, every item the same (the submitted tree member)", "data/q_backtest_nx_ec_ns.parquet"), ("New features, volume^0.3", "data/q_backtest_nx_w03ec.parquet")]
ix = pd.MultiIndex.from_arrays([D.origin, D.style_color]); LS = []
for lab, fn in VARF:
    q = pd.read_parquet(fn).set_index(["origin", "style_color"]).reindex(ix); f = q.q50.values; assert np.isfinite(f).all() and np.allclose(q.ytot.values, y); d = score(f); d["variant"] = lab; LS.append(d)
print("\nC  the tree member alone, by loss:"); print(pd.DataFrame(LS).set_index("variant")[["ratio", "wape_abs", "bias", "hit", "hit_units", "med_ape", "mean_ape", "male"]].to_string())
J["timeitem"] = TI; J["lossscale"] = LS; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
