"""D363 (overnight goal, Main's share part 2): tuning of the correction layer on the candidate system (base data/dec_base_ecnsnn<TAG>.parquet,
default TAG = hubrq: Huber tree + rc_deepfm at a quarter weight, D358 / D361). Runs on Main's CPU server; no scoring suite needed.
Grids (each run time-honest, as the production scripts):
  segment re-allocation (eda/179x2_bias.py, NOSCORE): SHRK (shrink constant k in n / (n + k)) x VOLQ (volume bands)
  breakout model (eda/179b_breakout.py, XSIG=1, NOSCORE): BRK_LEAVES x BRK_ROUNDS
Post hoc, from the saved outputs: segment strength ls (from the stored 0.5 column: adj = 2 log(col / F)), breakout strength lb
(F x exp(lb x rhat)), each renormalised to the origin's F total, and the blend weight w in geo = seg^w x brk^(1 - w), renormalised.
Metrics straight from the base table (y, R3): WAPE / R3 on the tuning origins 2025-06 .. 2025-09 (the corrections need earlier closed
origins, so 2025-02 .. 05 are uncorrected), the hold-out 2025-10 .. 2026-02, all 13 and the four early origins.
Writes data/hpo_corr_trials.csv (one row per combination). Usage: python eda/192_hpo_corr.py [tag=hubrq] [threads=4]"""
import os, sys, itertools, subprocess, numpy as np, pandas as pd
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); TAG = KW.get("tag", "hubrq"); TH = KW.get("threads", "4"); PY = sys.executable
SEG = [(k, v) for k in (1, 2, 4, 8) for v in (3, 5, 7)]; BRK = [(7, 150), (7, 300), (15, 150), (15, 300), (15, 600), (31, 300), (31, 600)]
env0 = dict(os.environ, NOSCORE="1", NJOBS=TH, DEC=f"_ecnsnn{TAG}")
for k, v in SEG:
    out = f"data/bias_seg_preds_ecnsnn{TAG}_s{k}_v{v}.parquet"
    if not os.path.exists(out): subprocess.run([PY, "eda/179x2_bias.py"], env=dict(env0, SHRK=str(k), VOLQ=str(v), OUTTAG=f"_s{k}_v{v}"), check=True, capture_output=True); print("seg", k, v, flush=True)
for L, R in BRK:
    out = f"data/breakout_preds_ecnsnn{TAG}_x_l{L}_r{R}.parquet"
    if not os.path.exists(out): subprocess.run([PY, "eda/179b_breakout.py"], env=dict(env0, XSIG="1", BRK_LEAVES=str(L), BRK_ROUNDS=str(R), OUTTAG=f"_l{L}_r{R}"), check=True, capture_output=True); print("brk", L, R, flush=True)
B = pd.read_parquet(f"data/dec_base_ecnsnn{TAG}.parquet")[["origin", "style_color", "y", "R3", "F"]]; B = B[B.y.notna()].reset_index(drop=True); key = ["origin", "style_color"]
SETS = {"tune": B.origin.between("2025-06", "2025-09").values, "hold": (B.origin >= "2025-10").values, "all13": np.ones(len(B), bool), "early": (B.origin <= "2025-05").values,
        "recent": (B.origin >= "2025-06").values}
e3 = {k: np.abs(B.R3.values[m] - B.y.values[m]).sum() for k, m in SETS.items()}
norm = lambda x: x * B.groupby("origin").F.transform("sum").values / pd.Series(x).groupby(B.origin.values).transform("sum").values
F = B.F.values
segs = {}
for k, v in SEG:
    S = B[key].merge(pd.read_parquet(f"data/bias_seg_preds_ecnsnn{TAG}_s{k}_v{v}.parquet")[key + ["vol_age_disc_0.5"]], on=key, how="left")
    adj = 2 * np.log(np.clip(S["vol_age_disc_0.5"].values, 1e-9, None) / np.clip(F, 1e-9, None)); adj = np.nan_to_num(adj)
    for ls in (0.3, 0.5, 0.7): segs[(k, v, ls)] = norm(F * np.exp(ls * adj))
brks = {}
for L, R in BRK:
    K = B[key].merge(pd.read_parquet(f"data/breakout_preds_ecnsnn{TAG}_x_l{L}_r{R}.parquet")[key + ["rhat"]], on=key, how="left"); rh = np.nan_to_num(K.rhat.values)
    for lb in (0.3, 0.5, 0.7): brks[(L, R, lb)] = norm(F * np.exp(lb * rh))
rows = []
for (sk, sv), (sb_) in itertools.product(segs.items(), brks.items()):
    (k, v, ls), s = sk, sv; (L, R, lb), b = sb_
    for w in (0.3, 0.5, 0.7):
        g = norm(np.exp(w * np.log(np.clip(s, 1e-9, None)) + (1 - w) * np.log(np.clip(b, 1e-9, None))))
        r = {"shrk": k, "volq": v, "seg_lam": ls, "brk_leaves": L, "brk_rounds": R, "brk_lam": lb, "w_seg": w}
        for nm, m in SETS.items(): r[nm] = round(float(np.abs(g[m] - B.y.values[m]).sum() / e3[nm]), 5)
        rows.append(r)
D = pd.DataFrame(rows); D.to_csv("data/hpo_corr_trials.csv", index=False)
dflt = D[(D.shrk == 2) & (D.volq == 5) & (D.seg_lam == 0.5) & (D.brk_leaves == 15) & (D.brk_rounds == 300) & (D.brk_lam == 0.5) & (D.w_seg == 0.5)]
print("combinations", len(D)); print("default:"); print(dflt.to_string(index=False))
best = D.sort_values("tune").head(10); print("best 10 on the tuning origins:"); print(best.to_string(index=False))
