"""D307 (fork; user comment on the error study, at the panel 'Known at the origin': "add the good options then", and in the session:
"focus also on adding more recent features from the last month"). The tree member itself refitted with the new inputs, not only the
correction layer: Main's member 'ec_ns' (eda/180q_models.py: quantile LightGBM on the log ratio to the level, BASE and FULL sets
averaged, training origins o' + 4 <= o) with
  SIG  the origin-known signals of the panel that the member does not have and that relate to the miss: stock-gap days, newest colour,
       size-run completeness, size concentration, partner share, partners selling it, share of the style's units, live colours in the
       style, web discount, discount against the site's, largest day's share  (eda/198d axes(), computed for every training row)
  LM   the 20 last-month inputs of D304 (eda/198s), computed for every training row
  SIG + LM
Each variant is scored alone and inside the five-member average (replacing the tree member), before the correction layer and with the
submitted correction multiplier on top. 'ec_ns' is refitted first and checked against the stored member.
Writes data/fk_member_refit_preds.parquet; adds J["refit"] to data/fk_err_analysis.json; stdout -> eda/member_refit_output.txt."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
# 1 Main's member table, feature sets and fit function (no variant is run: argv is empty)
argv0 = sys.argv; sys.argv = ["eda/180q_models.py"]; ns = {}; exec(compile(open("eda/180q_models.py", encoding="utf-8").read(), "eda/180q_models.py", "exec"), ns); sys.argv = argv0
T = ns["T"].reset_index(drop=True); sets = ns["sets"]; fit = ns["fit"]; ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]; print("member table:", T.shape, "origins", T.origin.min(), "to", T.origin.max(), flush=True)
# 2 SIG: the origin-known axes for every row (eda/198d's axes(), hindsight off)
src = open("eda/198d_error_master.py", encoding="utf-8").read(); src = src[src.index("import sys, glob"):src.index("\ndef build(")]; ax = {}; exec(compile(src, "eda/198d_error_master.py", "exec"), ax)
AX = pd.concat([ax["axes"](g, o, hindsight=False) for o, g in T.groupby("origin")]).reindex(T.index).add_prefix("ax_")
SIG = ["ax_" + c for c in ["oos_28", "newest_colour", "size_run", "size_hhi", "sh_partner", "n_partners", "share_of_style", "n_siblings", "disc_web", "disc_vs_site", "top_day_share"]]
# 3 LM: the last-month inputs for every row (as eda/198s)
dd = pd.read_parquet("data/daily_item_chan.parquet"); dd["d"] = pd.to_datetime(dd.ny_date); ch = dd.chan.astype(str).str.lower(); dd["k"] = np.where(dd["class"].astype(str).str.upper() == "RETAIL", "ret", np.where(ch.str.contains("tiktok"), "tt", np.where(ch == "edikted.com", "web", "par")))
ok_ = dd[~dd.cancelled]; can = dd[dd.cancelled]; lr = lambda a, b: np.log((a + 1) / (b + 1)); parts = []
for o in sorted(T.origin.unique()):
    idx = np.where((T.origin == o).values)[0]; it = T.style_color.values[idx]; end = pd.Period(o, "M").end_time.normalize(); days = pd.date_range(end - pd.Timedelta(days=55), end)
    w = ok_[(ok_.d >= days[0]) & (ok_.d <= end) & ok_.style_color.isin(it)]; piv = lambda df, v: df.pivot_table(index="style_color", columns="d", values=v, aggfunc="sum").reindex(index=it, columns=days).fillna(0).values
    U = piv(w, "qty"); a, b = U[:, 28:], U[:, :28]; u28, p28 = a.sum(1), b.sum(1); f = pd.DataFrame(index=idx)
    f["lm_r7"] = lr(a[:, -7:].sum(1) * 4, u28); f["lm_r14"] = lr(a[:, -14:].sum(1) * 2, u28); f["lm_acc"] = lr(u28, p28); wk = np.log1p(np.stack([a[:, i * 7:(i + 1) * 7].sum(1) for i in range(4)], 1)); xw = np.arange(4) - 1.5; f["lm_slope"] = (wk * xw).sum(1) / (xw ** 2).sum()
    f["lm_vs_level"] = lr(u28 * 30.4 / 28, T.level.values[idx]); f["lm_days_sold"] = (a > 0).mean(1); last = np.where((a > 0).any(1), 27 - (a > 0)[:, ::-1].argmax(1), -1); f["lm_days_since"] = np.where(last >= 0, 27 - last, 28)
    f["lm_peak_ago"] = np.where(u28 > 0, 27 - a.argmax(1), np.nan); f["lm_peak_r"] = np.where(u28 > 0, a.max(1) / np.maximum(a.mean(1), 1e-9), np.nan); wd = np.array([d.dayofweek for d in days[28:]]); f["lm_wkend"] = np.where(u28 > 0, a[:, wd >= 5].sum(1) / np.maximum(u28, 1), np.nan)
    O = piv(w, "n_orders"); f["lm_basket"] = np.where(O[:, 28:].sum(1) > 0, u28 / np.maximum(O[:, 28:].sum(1), 1), np.nan); Dw = piv(w, "disc_w"); da = np.where(u28 > 0, Dw[:, 28:].sum(1) / np.maximum(u28, 1), np.nan); db = np.where(p28 > 0, Dw[:, :28].sum(1) / np.maximum(p28, 1), np.nan)
    f["lm_disc"] = da; f["lm_disc_chg"] = da - db
    for k in ("tt", "ret", "par"):
        Uk = piv(w[w.k == k], "qty"); sa = np.where(u28 > 0, Uk[:, 28:].sum(1) / np.maximum(u28, 1), np.nan); sb = np.where(p28 > 0, Uk[:, :28].sum(1) / np.maximum(p28, 1), np.nan); f[f"lm_sh_{k}"] = sa; f[f"lm_sh_{k}_chg"] = sa - sb
    C = piv(can[(can.d >= days[0]) & (can.d <= end) & can.style_color.isin(it)], "qty"); f["lm_cancel"] = C[:, 28:].sum(1) / np.maximum(C[:, 28:].sum(1) + u28, 1); parts.append(f)
LMF = pd.concat(parts).reindex(T.index); LM = list(LMF.columns); T = pd.concat([T, AX[SIG], LMF], axis=1); print("SIG", len(SIG), "| LM", len(LM), "| coverage SIG", round(float(T[SIG].notna().mean().mean()), 3), "LM", round(float(T[LM].notna().mean().mean()), 3), flush=True)
# 4 refit, time-honest
B0, F0_ = sets("ec_ns"); VAR = [("ec_ns (refit, check)", []), ("+ origin-known signals", SIG), ("+ last-month inputs", LM), ("+ signals and last-month inputs", SIG + LM)]; PRED = {}
for lab, add in VAR:
    t0 = time.time(); out = []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; te = T[T.origin == o]; assert (tr.o_ord + 4 <= oo).all() and len(te) > 0
        z = (fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2; out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
    PRED[lab] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, "|", len(F0_ + add), "features (full set) |", round(time.time() - t0), "s", flush=True)
# 5 score: alone and in the five-member average
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); rec = (D.origin >= "2025-06").values; early = ~rec; allm = np.ones(len(D), bool); y = D.y.values; ar = (D.R3 - D.y).abs().values
MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values; base = X.mean(1); rng = np.random.default_rng(0)
wape = lambda f, m=rec: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m=rec: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean()); bias = lambda f, m=rec: float(f[m].sum() / y[m].sum() - 1)
def boot(fa, fb_, B=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb_ - y), "r": ar, "ha": np.abs(fa - y) <= 0.2 * y, "hb": np.abs(fb_ - y) <= 0.2 * y, "n": 1.0})[rec].groupby("s").sum().values; n = len(Z); T_ = Z[rng.integers(0, n, (B, n))].sum(1)
    d = T_[:, 1] / T_[:, 2] - T_[:, 0] / T_[:, 2]; hd = (T_[:, 4] - T_[:, 3]) / T_[:, 5]; return [r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))], [r(np.percentile(hd, 2.5), 3), r(np.percentile(hd, 97.5), 3)]
chk = PRED[VAR[0][0]].reindex(K).values; print("refit check against the stored member: max relative difference", float(np.nanmax(np.abs(chk - D.qdyn.values) / np.maximum(D.qdyn.values, 1))), "| correlation", float(np.corrcoef(np.log1p(chk), np.log1p(D.qdyn.values))[0, 1]))
rows = []; save = D[["origin", "style_color"]].copy(); m0 = PRED[VAR[0][0]].reindex(K).values; s0 = (X.sum(1) - D.qdyn.values + m0) / 5
for lab, add in VAR:
    m = PRED[lab].reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5; save[lab] = m
    po = [wape(st, (D.origin == o).values) - wape(s0, (D.origin == o).values) for o in sorted(D.origin.unique())]; ci_m, _ = boot(m0, m) if add else ([0, 0], [0, 0]); ci_s, hci_s = boot(s0, st) if add else ([0, 0], [0, 0])
    rows.append(dict(variant=lab, n_add=len(add), alone=r(wape(m)), alone_all=r(wape(m, allm)), alone_hit=r(hit(m), 3), alone_ci=ci_m, stack=r(wape(st)), stack_all=r(wape(st, allm)), stack_early=r(wape(st, early)), stack_hit=r(hit(st), 3), stack_bias=r(bias(st), 3), stack_ci=ci_s, stack_hit_ci=hci_s,
                     corrected=r(wape(st * corr)), corrected_hit=r(hit(st * corr), 3), corrected_all=r(wape(st * corr, allm)), wins13=int(sum(v < 0 for v in po)), wins9=int(sum(v < 0 for v, o in zip(po, sorted(D.origin.unique())) if o >= "2025-06")), per_origin=[r(v) for v in po]))
print(pd.DataFrame(rows).drop(columns=["per_origin"]).to_string(index=False)); print("stored system for reference: stack", r(wape(base)), "| submitted", r(wape(D.F.values)), "| hit", r(hit(D.F.values), 3))
# importance of the new inputs in the last fit (full set, all 13 origins' training rows)
tr = T[(T.o_ord + 4 <= pd.Period("2026-06").ordinal) & T.ytot.notna()]; cols = F0_ + SIG + LM; mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **ns["P"]).fit(tr[cols], tr.z); g = pd.Series(mdl.booster_.feature_importance("gain"), index=cols); g = g / g.sum()
imp = [dict(key=c, gain=r(g[c], 4), rank=int((g > g[c]).sum()) + 1) for c in g[SIG + LM].sort_values(ascending=False).index[:10]]; print("share of gain held by the new inputs:", r(g[SIG + LM].sum(), 3), "| top:", imp[:6])
save.to_parquet("data/fk_member_refit_preds.parquet"); J = json.load(open("data/fk_err_analysis.json")); J["refit"] = dict(rows=rows, new_gain=r(g[SIG + LM].sum(), 3), importance=imp, n_sig=len(SIG), n_lm=len(LM), stored_stack=r(wape(base)), submitted=r(wape(D.F.values)), submitted_hit=r(hit(D.F.values), 3))
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
