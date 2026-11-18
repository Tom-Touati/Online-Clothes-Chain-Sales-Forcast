"""D192 (fork): zero-shot time-series foundation models on each item's weekly sales, on the remote GPU (Tesla T4).
Models (Apache-2.0 open weights): amazon/chronos-2, amazon/chronos-bolt-base, google/timesfm-2.5-200m-pytorch.
At each origin (13 backtest origins Feb 2025 .. Feb 2026, plus Jun 2026 for the deliverable), for every item with sales by the origin:
  context  daily units from the item's first sale to the origin's month end, summed into 7-day blocks counted back from that day
           (at most 104 weeks)
  forecast 18 weeks; each week's quantile is spread evenly over its 7 days and summed into the 4 target months (sum of quantiles:
           exact for the median path, an approximation for the tails)
  embed    Chronos-2 and Chronos-Bolt encoder outputs, mean over tokens (768 dims), saved for the trees
Writes out/fk_tsfm_<model>.parquet (model, origin, style_color, h, q10, q25, q50, q75, q90) and out/fk_tsfm_emb_<model>.npz."""
import sys, time, numpy as np, pandas as pd, torch
Z = np.load("data/fk_daily.npz"); items = Z["items"]; Y = Z["Y"]; day0 = pd.Timestamp(str(Z["start"]))
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")] + ["2026-06"]
QL = [0.1, 0.25, 0.5, 0.75, 0.9]; H = 18; MAXW = 104
first = np.where(Y.sum(1) > 0, (Y > 0).argmax(1), 10 ** 6)


def contexts(o):
    E = (pd.Period(o, "M").end_time.normalize() - day0).days            # index of the origin's last day
    ids, ctx = [], []
    for i in range(len(items)):
        if first[i] > E: continue
        n = min(MAXW, (E - first[i]) // 7 + 1, (E + 1) // 7); s = Y[i, E + 1 - 7 * n:E + 1].reshape(n, 7).sum(1); ids.append(i); ctx.append(s.astype(np.float32))
    return E, ids, ctx


def to_months(o, E, qw):                                               # qw: items x H x len(QL) weekly quantiles
    days = pd.date_range(day0 + pd.Timedelta(days=E + 1), periods=7 * H); mon = days.to_period("M")
    daily = np.repeat(qw / 7.0, 7, axis=1)                             # items x 7H x Q
    out = []
    for h in range(1, 5):
        m = mon == (pd.Period(o, "M") + h)
        out.append(daily[:, m, :].sum(1))
    return np.stack(out, 1)                                            # items x 4 x Q


def run(name, predict, embed=None, bs=256):
    rows, embs, t0 = [], {}, time.time()
    for o in ORIG:
        E, ids, ctx = contexts(o); q = np.zeros((len(ids), H, len(QL)), np.float32); em = []
        for b in range(0, len(ids), bs):
            q[b:b + bs] = predict(ctx[b:b + bs])
            if embed is not None: em.append(embed(ctx[b:b + bs]))
        Mq = np.clip(to_months(o, E, np.clip(q, 0, None)), 0, None); Mq.sort(axis=2)
        for h in range(4):
            d = pd.DataFrame(Mq[:, h, :], columns=[f"q{int(x * 100):02d}" for x in QL]); d.insert(0, "h", h + 1); d.insert(0, "style_color", items[ids]); d.insert(0, "origin", o)
            rows.append(d)
        if embed is not None: embs[o] = np.concatenate(em).astype(np.float16); embs[o + "_items"] = items[ids]
        print(name, o, len(ids), f"{time.time() - t0:.0f}s", flush=True)
    R = pd.concat(rows, ignore_index=True); R.insert(0, "model", name); R.to_parquet(f"out/fk_tsfm_{name}{'_all' if ALL else ''}.parquet", index=False)
    if embs: np.savez_compressed(f"out/fk_tsfm_emb_{name}{'_all' if ALL else ''}.npz", **embs)


ALL = "allorig" in sys.argv; WHICH = [a for a in sys.argv[1:] if a != "allorig"] or ["chronos2", "bolt", "timesfm"]
if ALL: ORIG = [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]      # every origin, for tree training rows (D198)
if "chronos2" in WHICH or "bolt" in WHICH:
    from chronos import BaseChronosPipeline
    for nm, hf in [("chronos2", "amazon/chronos-2"), ("bolt", "amazon/chronos-bolt-base")]:
        if nm not in WHICH: continue
        p = BaseChronosPipeline.from_pretrained(hf, device_map="cuda")
        def pred(ctx, p=p):
            q, _ = p.predict_quantiles([torch.tensor(c) for c in ctx], prediction_length=H, quantile_levels=QL)
            q = torch.stack([x.reshape(H, len(QL)) for x in q]) if isinstance(q, list) else q
            return q.float().cpu().numpy()
        def emb(ctx, p=p):
            e = p.embed([torch.tensor(c) for c in ctx]); e = e[0] if isinstance(e, tuple) else e
            return np.stack([x.float().reshape(-1, x.shape[-1]).mean(0).cpu().numpy() for x in e]) if isinstance(e, list) else e.float().mean(1).cpu().numpy()
        with torch.no_grad(): run(nm, pred, emb)
        del p; torch.cuda.empty_cache()
if "timesfm" in WHICH:
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=256, max_horizon=64, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    def pred_t(ctx):
        _, qf = m.forecast(horizon=H, inputs=[c.astype(np.float64) for c in ctx])  # qf: n x H x 10 = mean, then deciles 0.1 .. 0.9
        dec = qf[:, :, 1:]
        return np.stack([dec[:, :, 0], (dec[:, :, 1] + dec[:, :, 2]) / 2, dec[:, :, 4], (dec[:, :, 6] + dec[:, :, 7]) / 2, dec[:, :, 8]], axis=2)
    with torch.no_grad(): run("timesfm", pred_t)
if "chronos2mv" in WHICH:
    # D197: Chronos-2 with group context (item + its product type + its macro group as one 3-variate series; the item is NaN before
    # launch) and known covariates (Black Friday days, other event days, normal temperature), past and future, all in 7-day blocks.
    from chronos import BaseChronosPipeline
    CV = np.load("data/fk_fm_cov.npz"); pti, mci, TS, MS = CV["pti"], CV["mci"], CV["TS"], CV["MS"]
    COV = {"ev_bfcm": CV["ev_bfcm"], "ev_other": CV["ev_other"], "t_norm": CV["t_norm"]}
    p = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda"); CUR = {}
    def wk(a, s, n, agg="sum"):
        x = a[s:s + 7 * n].reshape(n, 7); return (x.mean(1) if agg == "mean" else x.sum(1)).astype(np.float32)
    def contexts(o):                                                    # overrides the univariate version for this model
        E = (pd.Period(o, "M").end_time.normalize() - day0).days; n = min(MAXW, (E + 1) // 7); s = E + 1 - 7 * n; ids, ctx = [], []
        past = {k: wk(v, s, n, "mean" if k == "t_norm" else "sum") for k, v in COV.items()}; fut = {k: wk(v, E + 1, H, "mean" if k == "t_norm" else "sum") for k, v in COV.items()}
        for i in range(len(items)):
            if first[i] > E: continue
            y = wk(Y[i], s, n); launch_blk = (first[i] - s) // 7
            if launch_blk > 0: y[:launch_blk] = np.nan
            ctx.append({"target": np.stack([y, wk(TS[pti[i]], s, n), wk(MS[mci[i]], s, n)]), "past_covariates": past, "future_covariates": fut}); ids.append(i)
        return E, ids, ctx
    def pred_mv(ctx):
        q, _ = p.predict_quantiles(ctx, prediction_length=H, quantile_levels=QL)
        return np.stack([x[0].float().cpu().numpy() for x in q])          # variate 0 = the item: H x Q
    with torch.no_grad(): run("chronos2mv", pred_mv, bs=64)
print("done")
