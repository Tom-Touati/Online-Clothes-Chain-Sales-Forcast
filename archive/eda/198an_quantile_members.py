"""D373 (fork; attempt 5 of Fork 2's delegation, user in Fork 2's session: "delegate these attempts"): quantile losses directly, instead of
patching the ranges afterwards.
  members   each member's own 4-month P25 / P75 as a forecast of the actual (Main's D364 file data/system_q2575_backtest.parquet):
              tree    quantile LightGBM at 0.25 / 0.75 (eda/187_losses.py q25 / q75)
              rc      Fork 2's recommender trained on pinball at 0.25 / 0.75 (eda/202 pin; f_honest_deepfm)
              pf2d32  TabPFN v2 dyn, its own predictive quantiles
              pf2c    conformal TabPFN (already conformal; for reference)
              rules   residual quantiles by segment on closed origins (calibrated by construction; for reference)
              nn      S13ecx6, monthly pinball quantiles summed (raw), and the fork's D327 calibrated version
            plus new NN runs (eda/197b_nn_ec.py on the CPU server, 3 seeds, all 13 origins):
              S13ecT    pinball head on the 4-month total directly (total=1)
              S13ecTi0  the same without the item and style embeddings (item=0: cannot memorise an item's own residuals)
              S13ecTd5  the same with dropout 0.5 and 10 epochs (less fitting)
              S13eci0   the monthly NN without the item and style embeddings
  system    the members' P25 (P75) averaged with the candidate's weights (1, 1, 1, 1, 1, 0.25) x the correction multiplier (as D364), with the
            NN raw or calibrated; and conformal on top: each side's log offset from P50 scaled per age x volume segment (as D366), left one
            origin out and purged; against the adopted rule (pf2c widths per segment)
Scores: below / inside / above, pinball in units / R3's (R3 as both quantiles), log interval score, inside by segment (min / max), and each
member's median WAPE / R3 for context. Writes the system bounds to data/fk_quantile_members_bounds.parquet; adds J["quantile_members"]; stdout -> eda/quantile_members_output.txt."""
import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
B = K.load(); key = ["origin", "style_color"]; KI = pd.MultiIndex.from_arrays([B.origin, B.style_color])
DB = pd.read_parquet("data/dec_base_ecnsnnhubrq.parquet").set_index(key).reindex(KI); GEO = pd.read_parquet("data/geo_ecnsnnhubrq_x_preds.parquet").set_index(key).reindex(KI)
mult = (GEO.geo / GEO.F.clip(lower=1e-9)).fillna(1).values; assert np.allclose(GEO.geo.values, B.P50.values)
tot = lambda f: pd.read_parquet(f).groupby(key)[["q10", "q25", "q50", "q75", "q90"]].sum().reindex(KI)
MQ = pd.read_parquet("data/fk_member_quantiles.parquet"); nnraw = MQ[MQ.member == "nn"].set_index(key).reindex(KI)
MEM = {"tree (quantile LightGBM)": ("tree_q25", "tree_q75", DB.qdyn.values), "rc (pinball recommender)": ("rc_q25", "rc_q75", DB.rc.values), "pf2d32 (TabPFN, own quantiles)": ("pf2d32_q25", "pf2d32_q75", DB.pf2d32.values),
       "pf2c (conformal TabPFN)": ("pf2c_q25", "pf2c_q75", DB.pf2c.values), "rules (segment residual quantiles)": ("rules_q25", "rules_q75", DB.rules.values)}
Q = {nm: (B[a].values, B[b].values, med) for nm, (a, b, med) in MEM.items()}
Q["nn S13ecx6, raw (monthly pinball, summed)"] = (nnraw.q25.values, nnraw.q75.values, DB.nn.values); Q["nn S13ecx6, calibrated (D327)"] = (B.nn_q25.values, B.nn_q75.values, DB.nn.values)
for tg, nm in [("_S13ecT", "nn total head"), ("_S13ecTi0", "nn total head, no item embeddings"), ("_S13ecTd5", "nn total head, dropout 0.5, 10 epochs"), ("_S13eci0", "nn monthly, no item embeddings")]:
    f = f"data/nn_backtest_preds{tg}.parquet"
    if os.path.exists(f): t = tot(f); Q[nm + f" ({tg[1:]})"] = (t.q25.values, t.q75.values, t.q50.values)
    else: print("missing", f)
y = B.y.values; R3 = B.R3.values; ar = np.abs(R3 - y); P = {"recent 9": (B.origin >= "2025-06").values, "all 13": np.ones(len(B), bool)}
lg = lambda v: np.log(np.maximum(v, 0) + 1)
rows = []
for nm, (q25, q75, med) in Q.items():
    q25, q75 = np.minimum(q25, q75), np.maximum(q25, q75)
    for pn, m in P.items():
        mm = m & np.isfinite(q25) & np.isfinite(q75); s = K.score(B, lg(q25), lg(q75), 0.5, mm)
        rows.append(dict(member=nm, period=pn, median_wape_r3=round(float(np.abs(med - y)[mm].sum() / ar[mm].sum()), 4), **{k: s[k] for k in ["n", "below", "inside", "above", "pinball_r3", "int_score", "log_width", "seg_min", "seg_max"]}))
R = pd.DataFrame(rows); print("=== members alone, their own P25 / P75 ==="); print(R.to_string(index=False))
W = {"rules (segment residual quantiles)": 1, "tree (quantile LightGBM)": 1, "pf2c (conformal TabPFN)": 1, "pf2d32 (TabPFN, own quantiles)": 1, "rc (pinball recommender)": 0.25}


def wavg(nn_name, j):
    cols = list(W) + [nn_name]; w = np.array([W.get(c, 1) for c in cols]); X = np.column_stack([np.clip(Q[c][j], 0, None) for c in cols]); ok = np.isfinite(X)
    return np.nansum(np.where(ok, X, 0) * w, 1) / np.maximum((ok * w).sum(1), 1e-9)


SYS = {}
for lab, nn_name in [("NN raw", "nn S13ecx6, raw (monthly pinball, summed)"), ("NN calibrated (D364 as delivered)", "nn S13ecx6, calibrated (D327)")] + [(f"NN {k}", k) for k in Q if k.startswith("nn total") or k.startswith("nn monthly")]:
    p25 = np.minimum(wavg(nn_name, 0) * mult, B.P50.values); p75 = np.maximum(wavg(nn_name, 1) * mult, B.P50.values); SYS["member average, " + lab] = (lg(p25), lg(p75))
lP = B.lP.values; e = B.e.values; ORIG = sorted(B.origin.unique())


def conformal(lo0, hi0, how, per_seg=True):
    wl, wu = np.maximum(lP - lo0, 0.02), np.maximum(hi0 - lP, 0.02); lo, hi = np.full(len(B), np.nan), np.full(len(B), np.nan)
    for o in ORIG:
        tr = K.train_mask(B, o, how); te = (B.origin == o).values; kg = K.factors(e[tr], wl[tr], wu[tr], 0.25)
        for sg in (B.seg[te].unique() if per_seg else [None]):
            ms = te & ((B.seg == sg).values if sg else True); mtr = tr & ((B.seg == sg).values if sg else True); k = K.factors(e[mtr], wl[mtr], wu[mtr], 0.25) if (sg is None or mtr.sum() >= 50) else kg
            lo[ms] = lP[ms] - k[0] * wl[ms]; hi[ms] = lP[ms] + k[1] * wu[ms]
    return lo, hi


WB = pd.read_parquet("data/fk_width_model_backtest.parquet"); srows = []; SAVE = {}
for how in ["loo", "purged"]:
    AD = WB[f"{how}|pf2c widths, per segment (adopted D366)|P25-P75|lo"].values, WB[f"{how}|pf2c widths, per segment (adopted D366)|P25-P75|hi"].values
    cand = {f"adopted: pf2c widths per segment ({how})": (lg(AD[0]), lg(AD[1]))}
    for lab, (lo0, hi0) in SYS.items():
        if how == "loo": cand[lab + " (raw)"] = (lo0, hi0)
        cand[lab + f" + conformal per segment ({how})"] = conformal(lo0, hi0, how)
    for lab, (lo, hi) in cand.items():
        SAVE[lab + "|lo"] = np.exp(lo) - 1; SAVE[lab + "|hi"] = np.exp(hi) - 1
        for pn, m in P.items():
            s = K.score(B, lo, hi, 0.5, m & np.isfinite(lo)); srows.append(dict(system=lab, period=pn, **{k: s[k] for k in ["n", "below", "inside", "above", "pinball_r3", "int_score", "log_width", "seg_min", "seg_max", "seg_sd"]}))
BOUNDS = {}
S = pd.DataFrame(srows).drop_duplicates(["system", "period"]); print("\n=== system P25-P75 ==="); print(S.to_string(index=False))
pd.concat([B[["origin", "style_color"]], pd.DataFrame(SAVE)], axis=1).to_parquet("data/fk_quantile_members_bounds.parquet")
J = json.load(open("data/fk_err_analysis.json")); J["quantile_members"] = dict(members=rows, system=S.to_dict("records")); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
