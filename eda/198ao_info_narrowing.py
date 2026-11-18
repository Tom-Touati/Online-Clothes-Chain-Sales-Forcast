"""D372 (fork; attempt 4 of Fork 2's delegation, user in Fork 2's session: "delegate these attempts"): how much the ranges narrow, at the same
coverage, once more is known.
  month 1   after the first month has sold (D322, eda/198af / data/fk_month1_preds.parquet, recent origins only): the 4-month total becomes
            month 1's actual + the updated months 2-4 ('scaled' by the month-1 surprise, or the D322 pick 'half-strength model, total from the
            scaled method'); against the original forecast at the origin (the submitted system, nx4 basis)
  plan      the D308 setting: the tree member refitted with the window's store, partner and launch plan as known-future inputs (realised values,
            an exact plan, so an upper bound), inside the five-member system with the submitted correction multiplier; against the same
            system without the plan (all 13 origins)
Ranges for each forecast by the same rules: (a) one width per segment (the per-segment quantiles of the log miss; D370's best on pinball) and
(b) the conformal TabPFN widths per segment (the adopted D366 rule), factors left one origin out, and purged (4+ months away) as a check.
Scores: below / inside / above, mean log width, interval score, pinball / R3's, and the median width in units as a share of the forecast
((P75 - P25) / P50). Adds J["info_narrowing"]; writes data/fk_plan_tree_preds.parquet; stdout -> eda/info_narrowing_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
B = K.load(); D = pd.read_parquet("data/fk_err_master.parquet").reset_index(drop=True)
assert (D.origin.values == B.origin.values).all() and (D.style_color.values == B.style_color.values).all()
# ---- the plan forecasts (D308's 'stores, partners and launches' variant), refitted here and saved
import os
if not os.path.exists("data/fk_plan_tree_preds.parquet"):
    src = open("eda/198u_member_future_stores.py", encoding="utf-8").read(); src = src[:src.index('\nKFP = ["kf_new_style"')]; ns = {"__name__": "plan"}
    exec(compile(src, "eda/198u_member_future_stores.py", "exec"), ns); ns["run"]("plan", ns["KF2"] + ["kf_new_style", "kf_new_type"]); ns["run"]("noplan", [])
    pd.DataFrame({"plan": ns["PRED"]["plan"], "noplan": ns["PRED"]["noplan"]}).reset_index().to_parquet("data/fk_plan_tree_preds.parquet")
PT = pd.read_parquet("data/fk_plan_tree_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color]))
corr = (D.F / D.F0).values; X = D[["rules", "nn", "pf2c", "pf2d32"]].values.astype(float).sum(1)
FC = {"system without the plan (tree refitted here)": (X + PT.noplan.values) / 5 * corr, "system with the plan (tree with stores, partners, launches)": (X + PT.plan.values) / 5 * corr}
# ---- month 1
M = pd.read_parquet("data/fk_month1_preds.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([D.origin, D.style_color]))
FM = {"at the origin (submitted, nx4 basis)": D.F.values, "after month 1: scaled by the surprise": M.m1.values + M.scaled.values, "after month 1: D322 pick (half model, scaled total)": M.m1.values + M.model_half_lev.values}


def ranges(Bx, lP, src, how, tail):
    e = Bx.ly.values - lP; lo, hi = np.full(len(Bx), np.nan), np.full(len(Bx), np.nan)
    wl = np.ones(len(Bx)) if src == "one width" else Bx["pw_q25" if tail == 0.25 else "pw_q10"].values
    wu = np.ones(len(Bx)) if src == "one width" else Bx["pw_q75" if tail == 0.25 else "pw_q90"].values
    ok = np.isfinite(lP)
    for o in sorted(Bx.origin[ok].unique()):
        tr = K.train_mask(Bx, o, how) & ok; te = (Bx.origin == o).values
        if Bx.origin[tr].nunique() < 2: continue
        kg = K.factors(e[tr], wl[tr], wu[tr], tail)
        for sg in Bx.seg[te].unique():
            ms = te & (Bx.seg == sg).values; mtr = tr & (Bx.seg == sg).values; k = K.factors(e[mtr], wl[mtr], wu[mtr], tail) if mtr.sum() >= 50 else kg
            lo[ms] = lP[ms] - k[0] * wl[ms]; hi[ms] = lP[ms] + k[1] * wu[ms]
    return lo, hi


rows = []
for part, FS, periods in [("month 1", FM, {"recent 9": (B.origin >= "2025-06").values}), ("plan", FC, {"recent 9": (B.origin >= "2025-06").values, "all 13": np.ones(len(B), bool)})]:
    common = np.logical_and.reduce([np.isfinite(f) for f in FS.values()])
    for lab, f in FS.items():
        lP = np.where(common, np.log(np.maximum(f, 0) + 1), np.nan); wr = float(np.abs(f - B.y.values)[common & (B.origin >= "2025-06").values].sum() / np.abs(B.R3.values - B.y.values)[common & (B.origin >= "2025-06").values].sum())
        for src in ["one width", "pf2c widths"]:
            for how in ["loo", "purged"]:
                for rng, tail, a in [("P25-P75", 0.25, 0.5), ("P10-P90", 0.10, 0.2)]:
                    lo, hi = ranges(B, lP, src, how, tail)
                    for pn, pm in periods.items():
                        m = pm & common & np.isfinite(lo); s = K.score(B, lo, hi, a, m); uw = float(np.median(((np.exp(hi) - np.exp(lo)) / np.exp(lP))[m]))
                        rows.append(dict(part=part, forecast=lab, wape_r3_recent=round(wr, 4), widths=src, protocol=how, range=rng, period=pn, **{k: s[k] for k in ["n", "below", "inside", "above", "log_width", "int_score", "pinball_r3", "seg_min", "seg_max"]}, unit_width=round(uw, 3)))
R = pd.DataFrame(rows)
for part in ["month 1", "plan"]:
    V = R[R.part == part]; print(f"\n=== {part} ==="); print(V.drop(columns=["part"]).to_string(index=False))
# narrowing: width after / width before, same rule, same protocol, same rows
nar = []
for (part, src, how, rng, pn), g in R.groupby(["part", "widths", "protocol", "range", "period"], sort=False):
    b = g.iloc[0]
    for _, x in g.iloc[1:].iterrows():
        nar.append(dict(part=part, widths=src, protocol=how, range=rng, period=pn, forecast=x.forecast, log_width_ratio=round(x.log_width / b.log_width, 3), unit_width_ratio=round(x.unit_width / b.unit_width, 3),
                        int_score_ratio=round(x.int_score / b.int_score, 3), pinball_ratio=round(x.pinball_r3 / b.pinball_r3, 3), inside_before=b.inside, inside_after=x.inside))
N = pd.DataFrame(nar); print("\nnarrowing (after / before, same rule and rows):"); print(N.to_string(index=False))
J = json.load(open("data/fk_err_analysis.json")); J["info_narrowing"] = dict(rows=rows, narrowing=nar); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
