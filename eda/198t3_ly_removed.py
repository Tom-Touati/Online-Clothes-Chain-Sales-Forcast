"""D315 (fork; proposal #10a of the error study, assigned in D314: "score every candidate a second time with last year's window removed and
prefer members that degrade least"; it gates Main's nx4 question). Main's tree member ec_ns and the D307 'sig' version (+ 11 origin-known
signals), each refitted at every backtest origin three ways:
  full      the time-honest pool (o' + 4 <= o)                                  (= D307)
  no LY     the pool without the origins that forecast the same calendar window a year earlier (o - 12 months, +/- 1), as D276
  control   the pool without as many other origins, drawn at random (3 draws, averaged), to separate the season from the training size
Scored alone and inside the five-member average (other members unchanged), with the submitted correction multiplier on top.
Inputs are built by eda/198t's own code (executed unchanged up to its backtest loop). Adds J["lyremoved"]; stdout -> eda/ly_removed_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
r = lambda x, k=4: None if x is None or not np.isfinite(x) else round(float(x), k)
src = open("eda/198t_member_refit.py", encoding="utf-8").read(); g = {"__name__": "lyremoved"}; exec(compile(src[:src.index("# 4 refit, time-honest")], "eda/198t_member_refit.py", "exec"), g)
T, sets, fit, SIG = g["T"], g["sets"], g["fit"], g["SIG"]; g["ns"]["P"]["n_jobs"] = 3
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]; B0, F0_ = sets("ec_ns"); rng = np.random.default_rng(0); PRED = {}; info = []
for lab, add in [("ec_ns", []), ("sig", SIG)]:
    for mode in ["full", "no LY", "control"]:
        out = []
        for o in ORIG:
            oo = pd.Period(o).ordinal; pool = sorted(T[(T.o_ord + 4 <= oo) & T.ytot.notna()].o_ord.unique()); same = [x for x in pool if abs(x - (oo - 12)) <= 1]; te = T[T.origin == o]
            if mode == "full" or not same or len(pool) - len(same) < 4: keeps = [pool]
            elif mode == "no LY": keeps = [[x for x in pool if x not in same]]
            else: keeps = [[x for x in pool if x not in set(rng.choice([y for y in pool if y not in same], len(same), replace=False))] for _ in range(3)]
            zs = []
            for keep in keeps:
                tr = T[T.o_ord.isin(keep) & T.ytot.notna()]; zs.append((fit(tr, te, B0 + add) + fit(tr, te, F0_ + add)) / 2)
            z = np.mean(zs, 0); out.append(pd.DataFrame({"origin": o, "style_color": te.style_color.values, "q50": np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)}))
            if lab == "ec_ns" and mode == "no LY": info.append(dict(origin=o, pool=len(pool), removed=len(same) if (same and len(pool) - len(same) >= 4) else 0))
        PRED[(lab, mode)] = pd.concat(out).set_index(["origin", "style_color"]).q50; print(lab, mode, "done", flush=True)
D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True); K = pd.MultiIndex.from_arrays([D.origin, D.style_color]); y = D.y.values; ar = (D.R3 - D.y).abs().values
rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; X = D[MEM].values.astype(float); corr = (D.F / D.F0).values
wape = lambda f, m: float(np.abs(f - y)[m].sum() / ar[m].sum()); hit = lambda f, m: float((np.abs(f - y)[m] <= 0.2 * y[m]).mean())
aff = D.origin.isin([d["origin"] for d in info if d["removed"] > 0]).values; rows = []
for (lab, mode), p in PRED.items():
    m = p.reindex(K).values; assert np.isfinite(m).all(); st = (X.sum(1) - D.qdyn.values + m) / 5 * corr
    rows.append(dict(member=lab, mode=mode, alone_rec=r(wape(m, rec)), alone_all=r(wape(m, allm)), alone_aff=r(wape(m, aff)), sys_rec=r(wape(st, rec)), sys_all=r(wape(st, allm)), sys_aff=r(wape(st, aff)), sys_hit=r(hit(st, rec), 3)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); print("origins where last year's window could be removed:", [d["origin"] for d in info if d["removed"] > 0])
deg = {}
for lab in ["ec_ns", "sig"]:
    a = R[(R.member == lab)].set_index("mode"); deg[lab] = dict(alone_ly=r(a.loc["no LY", "alone_aff"] - a.loc["full", "alone_aff"]), alone_ctrl=r(a.loc["control", "alone_aff"] - a.loc["full", "alone_aff"]), sys_ly=r(a.loc["no LY", "sys_aff"] - a.loc["full", "sys_aff"]), sys_ctrl=r(a.loc["control", "sys_aff"] - a.loc["full", "sys_aff"]))
print("degradation on the affected origins (no LY - full | control - full):", deg)
sv = D[["origin", "style_color"]].copy()
for (lab, mode), p in PRED.items(): sv[f"{lab} | {mode}"] = p.reindex(K).values
sv.to_parquet("data/fk_ly_removed_preds.parquet")
J = json.load(open("data/fk_err_analysis.json")); J["lyremoved"] = dict(rows=rows, degradation=deg, origins=info); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
