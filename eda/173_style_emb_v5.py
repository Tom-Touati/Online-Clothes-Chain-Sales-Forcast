"""D167 (user goal): v5 of the content style embedding: v4 plus FORWARD-LOOKING time inputs, so region, temperature and event timing
describe the item's position in the coming 3 months (the future-cart target), all knowable at the origin:
  fwd_cal    the item's calendar share (24 months up to the origin) for the 3 coming calendar months, mean and each
  fwd_evt    its monthly event lifts (eda/170) weighted by each event's days in the coming 3 months; its daily 30-day-before lifts
             (eda/172) for the events falling in the coming 3 months
  fwd_temp   the normal temperature of the coming 3 months where the item sells (12-month state mix x state normals) minus the
             temperature at which it has sold (temperature gap); its temperature slope x that gap (expected temperature-driven change)
v5 = v4 + fwd block. Same gate as D164/D166 (held-out styles, 3 origins x 5 folds, past and future cart PPMI, style bootstrap).
Writes eda/style_emb_v5_output.txt and data/style_emb_v5_results.csv (v4 rerun alongside for paired comparison)."""
import os, sys
src = open("eda/172_style_emb_v4.py", encoding="utf-8").read()
_argv = sys.argv; sys.argv = sys.argv[:1]
exec(src[:src.index("CFG4 = {")])
sys.argv = _argv
RUN5 = sys.argv[1:] or ["v4", "v5", "v5_fwdonly_time"]
NORM = W.pivot(index="m", columns="state", values="tmean_norm").reindex(columns=states)


def fwd(o, TEMP2, EVT2):
    nm = [Pm(o, k) for k in (1, 2, 3)]; cmn = [int(m[-2:]) - 1 for m in nm]
    ms = [Pm(o, -k) for k in range(23, -1, -1)]; U = M.reindex(columns=ms).fillna(0).values; lv = np.cumsum(U, 1) > 0
    ub = np.where(lv, U, np.nan); cm = np.array([int(m[-2:]) - 1 for m in ms])
    cal = np.zeros((n, 12))
    for k in range(12): cal[:, k] = np.nansum(np.where(cm[None, :] == k, ub, np.nan), 1)
    cal = np.log((cal + 1) / (cal.sum(1, keepdims=True) / 12 + 1)); fcal = cal[:, cmn]
    E_hist = EV.reindex(ms).fillna(0).values; mean_u = np.nanmean(ub, 1, keepdims=True)
    aff = np.log((np.nan_to_num(ub) @ E_hist + 1) / (np.nan_to_num(mean_u) * (lv.astype(float) @ E_hist) + 1))
    En = EV.reindex(nm).fillna(0).values.sum(0); w_e = En / max(En.sum(), 1e-9)
    f_aff = aff @ w_e
    keys = list(KEY.keys()); lift = EVT2[:, :len(keys)]                     # standardised daily lifts
    soon = np.array([EV.reindex(nm).fillna(0)[e].sum() > 0 for e in keys]) if all(e in EV.columns for e in keys) else np.zeros(len(keys), bool)
    f_lift = lift[:, soon].mean(1) if soon.any() else np.zeros(n)
    w12 = [mi[m] for m in mons if Pm(o, -11) <= m <= o]; u12 = UST[w12].sum(0)
    mix = (u12 + 20 * (u12.sum(0) / max(u12.sum(), 1))) / (u12.sum(1, keepdims=True) + 20)
    tnext = mix @ NORM.reindex([f"2025-{c + 1:02d}" for c in cmn]).fillna(NORM.mean()).values.mean(0)   # calendar-month normals
    tsell, slope = TEMP2[:, 0], TEMP2[:, 1]                                                            # standardised in inputs2
    gap = zs(tnext) - tsell
    return np.nan_to_num(zs(np.column_stack([fcal.mean(1), fcal, f_aff, f_lift, gap, slope * gap])))


CFG5 = {"v4": ("nce", ["prod", "reg", "temp4", "evt4"]), "v5": ("nce", ["prod", "reg", "temp4", "evt4", "fwd"]),
        "v5_fwdonly_time": ("nce", ["prod", "reg", "fwd"])}
res = []; t0 = time.time()
for o in ORIG:
    REGION, TEMP, EVENTS = inputs(o); TEMP2, EVT2 = inputs2(o); FWD = fwd(o, TEMP2, EVT2); TR = truth(o)
    B = {"prod": np.hstack([ATT, COLL, NAME, LLM]), "reg": REGION, "temp4": np.hstack([TEMP, TEMP2]), "evt4": np.hstack([EVENTS, EVT2]), "fwd": FWD}
    raw_prod = rown(zs(PROD_RAW))
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    for k in range(NF):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]
        Hs = {"raw product": raw_prod}
        for v in RUN5:
            obj, blk = CFG5[v]; Hs[v] = train([B[b] for b in blk], tr, *TR["past"], obj)
        for tgt, (P_, ok_) in TR.items():
            for nm_, H in Hs.items():
                d = score(H, te, tr, P_, ok_); d["model"] = nm_; d["target"] = tgt; d["origin"] = o; d["fold"] = k; res.append(d)
        print(o, "fold", k, f"{time.time() - t0:.0f}s", flush=True)
R = pd.concat(res); R.to_csv("data/style_emb_v5_results.csv", index=False)
print("\n== mean over held-out items: Spearman with cart PPMI | top-20 precision | chance")
print(R.groupby(["target", "model"])[["rho", "prec", "chance"]].mean().round(4).to_string())
rng = np.random.default_rng(0)
for ref in ["raw product", "v4"]:
    print(f"\n== paired bootstrap over held-out styles, model minus {ref} (95% interval)")
    for tgt in ["past", "future"]:
        base = R[(R.target == tgt) & (R.model == ref)].groupby(["origin", "style"])[["rho", "prec"]].mean()
        for m in [x for x in R.model.unique() if x != ref]:
            x = R[(R.target == tgt) & (R.model == m)].groupby(["origin", "style"])[["rho", "prec"]].mean(); dlt = (x - base.reindex(x.index)).dropna()
            bs = np.array([dlt.values[rng.integers(0, len(dlt), len(dlt))].mean(0) for _ in range(1000)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f"  {tgt:6s} {m:18s} rho {dlt.rho.mean():+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | prec {dlt.prec.mean():+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
