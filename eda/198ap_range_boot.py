"""D373b (fork; Fork 2's range attempts): item bootstrap of the pinball differences behind D370 and D373, so the small gains carry intervals.
Each comparison: (sum of the challenger's pinball - sum of the reference's pinball) / sum of R3's pinball, at 0.25 / 0.75 (or 0.10 / 0.90),
on the same rows; items resampled with replacement (1,000 draws), 95% interval. Negative = the challenger is better.
Sources: data/fk_width_model_backtest_l1.parquet (D370 second pass), data/fk_quantile_members_bounds.parquet (D373).
Adds J["range_boot"]; stdout -> eda/range_boot_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
B = K.load(); W = pd.read_parquet("data/fk_width_model_backtest_l1.parquet"); Q = pd.read_parquet("data/fk_quantile_members_bounds.parquet")
assert (W.style_color.values == B.style_color.values).all() and (Q.style_color.values == B.style_color.values).all()
y, R3 = B.y.values, B.R3.values; rng = np.random.default_rng(0)
AD = "pf2c widths, per segment (adopted D366)"
CMP = [("width model (L1, one k per side) vs adopted, purged", W, "purged|width model, global|{r}", "purged|" + AD + "|{r}", ["P25-P75", "P10-P90"], ["recent 9", "all 13"]),
       ("width model (L1, one k per side) vs adopted, honest", W, "honest|width model, global|{r}", "honest|" + AD + "|{r}", ["P25-P75", "P10-P90"], ["2025-07..2026-02"]),
       ("member average, NN raw (no conformal) vs adopted (loo)", Q, "member average, NN raw (raw)", "adopted: pf2c widths per segment (loo)", ["P25-P75"], ["recent 9", "all 13"]),
       ("member average, NN monthly without item embeddings (no conformal) vs adopted (loo)", Q, "member average, NN nn monthly, no item embeddings (S13eci0) (raw)", "adopted: pf2c widths per segment (loo)", ["P25-P75"], ["recent 9", "all 13"]),
       ("member average as delivered (D364) + conformal per segment vs adopted (loo)", Q, "member average, NN calibrated (D364 as delivered) + conformal per segment (loo)", "adopted: pf2c widths per segment (loo)", ["P25-P75"], ["recent 9", "all 13"])]
PER = {"recent 9": (B.origin >= "2025-06").values, "all 13": np.ones(len(B), bool), "2025-07..2026-02": (B.origin >= "2025-07").values}
rows = []
for lab, src, a, b, rngs, pers in CMP:
    for r in rngs:
        t = 0.25 if r == "P25-P75" else 0.10; ca, cb = a.format(r=r), b.format(r=r)
        la, ha, lb, hb = src[ca + "|lo"].values, src[ca + "|hi"].values, src[cb + "|lo"].values, src[cb + "|hi"].values
        pa = K.pin(y, la, t) + K.pin(y, ha, 1 - t); pb = K.pin(y, lb, t) + K.pin(y, hb, 1 - t); pr = K.pin(y, R3, t) + K.pin(y, R3, 1 - t)
        for pn in pers:
            m = PER[pn] & np.isfinite(pa) & np.isfinite(pb)
            Z = pd.DataFrame({"s": B.style_color.values[m], "a": pa[m], "b": pb[m], "r": pr[m]}).groupby("s").sum().values; n = len(Z)
            T = Z[rng.integers(0, n, (1000, n))].sum(1); d = (T[:, 0] - T[:, 1]) / T[:, 2]
            rows.append(dict(comparison=lab, range=r, period=pn, challenger=round(float(Z[:, 0].sum() / Z[:, 2].sum()), 4), reference=round(float(Z[:, 1].sum() / Z[:, 2].sum()), 4),
                             diff=round(float((Z[:, 0].sum() - Z[:, 1].sum()) / Z[:, 2].sum()), 4), ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]))
R = pd.DataFrame(rows); print(R.to_string(index=False))
J = json.load(open("data/fk_err_analysis.json")); J["range_boot"] = rows; json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float); print("saved")
