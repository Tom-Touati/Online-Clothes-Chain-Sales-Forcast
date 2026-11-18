"""Segment bias of the current answer (61st percentile) at the summer origins (Jun, Jul 2025: windows cover Jul-Nov 2025),
and the monthly shape of the monthly model at the June 2025 origin, by category group and occasion tag."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
QS = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]); QC = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
Q = pd.read_parquet("data/q_backtest_own.parquet")
Q["ans"] = np.expm1([np.interp(0.61, QS, r) for r in np.log1p(Q[QC].values)])
F = pd.read_parquet("data/features.parquet"); h1 = F[F.h == 1][["origin", "style_color", "macro", "ptype"] + [c for c in F.columns if c.startswith("occ_")]]
Q = Q.merge(h1, on=["origin", "style_color"])
Q["tag"] = np.select([Q.occ_halloween == 1, Q.occ_homecoming == 1, Q.occ_4th == 1, Q.occ_valentine == 1, Q.occ_graduation == 1, Q.occ_new == 1], ["Halloween", "Homecoming", "4th of July", "Valentine", "Graduation", "New Year"], "no tag")
Q["ageb"] = pd.cut(Q.age_m, [0, 3, 6, 12, 24, 99], labels=["<3", "3-6", "6-12", "12-24", "24+"])
def seg(d, by):
    g = d.groupby(by, observed=True)
    return pd.DataFrame({"n": g.size(), "bias": g.apply(lambda x: x.ans.sum() / x.ytot.sum() - 1), "med_ratio": g.apply(lambda x: (x.ans / x.ytot.clip(lower=1)).median()),
                         "w20": g.apply(lambda x: ((x.ans - x.ytot).abs() / x.ytot.clip(lower=1) <= .2).mean())}).round(3)
for nm, sel in [("summer origins Jun+Jul 2025", Q.origin.isin(["2025-06", "2025-07"])), ("all 9 representative origins", Q.origin.isin([str(p) for p in pd.period_range("2025-06", "2026-02", freq="M")]))]:
    d = Q[sel]; print(f"\n== {nm} (n={len(d)}): answer / actual")
    for by in ["tag", "macro", "ageb"]: print(seg(d, by).to_string()); print()
# monthly shape of the monthly model at the June 2025 origin
B = pd.read_parquet("data/best_backtest_preds.parquet"); B = B[B.origin == "2025-06"].merge(h1[h1.origin == "2025-06"], on=["origin", "style_color"])
B["tag"] = np.where(B.occ_halloween == 1, "Halloween", np.where(B.occ_homecoming == 1, "Homecoming", "other"))
for by in ["macro", "tag"]:
    p = B.pivot_table(index=by, columns="h", values=["pred", "y"], aggfunc="sum")
    sh = pd.DataFrame({f"pred_m{h}/m1": p["pred"][h] / p["pred"][1] for h in (2, 3, 4)}).join(pd.DataFrame({f"act_m{h}/m1": p["y"][h] / p["y"][1] for h in (2, 3, 4)}))
    print(f"monthly shape at the June 2025 origin by {by} (month h / July):"); print(sh.round(2).to_string()); print()
