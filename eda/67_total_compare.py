"""4-month-total level: production quantile model (q_backtest) vs NN total model (eda/57 total=1) vs NN monthly model summed
(eda/57 monthly, B13) and their quantile averages (Vincentization in log1p space at the 5 shared quantile levels).
Scores P50 WAPE / bias, pinball loss over the 5 levels, and [P25,P75] / [P10,P90] coverage; all 13 origins and the 9 representative ones."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
TAG = sys.argv[1] if len(sys.argv) > 1 else "_T13"
QL = ["q10", "q25", "q50", "q75", "q90"]; QV = np.array([.10, .25, .50, .75, .90])
Q = pd.read_parquet("data/q_backtest.parquet").set_index(["origin", "style_color"])
NT = pd.read_parquet(f"data/nn_backtest_preds{TAG}.parquet").set_index(["origin", "style_color"])
NM = pd.read_parquet(f"data/{sys.argv[2] if len(sys.argv) > 2 else 'nn_backtest_preds_B13'}.parquet").groupby(["origin", "style_color"])[QL].sum()   # sum of monthly quantiles (not a total quantile; P50 only is used)
idx = Q.index.intersection(NT.index).intersection(NM.index)
A = Q.loc[idx, "ytot"].values; L = {"qmodel": np.log1p(Q.loc[idx, QL].values), "nn_total": np.log1p(NT.loc[idx, QL].values)}
L["avg qmodel+nn_total"] = 0.5 * L["qmodel"] + 0.5 * L["nn_total"]
L["qmodel recentred on nn_monthly (D085)"] = L["qmodel"] + 0.5 * (np.log1p(NM.loc[idx, "q50"].values) - L["qmodel"][:, 2])[:, None]
L["avg of 3 medians"] = L["qmodel"] + ((np.log1p(NM.loc[idx, "q50"].values) + L["nn_total"][:, 2] - 2 * L["qmodel"][:, 2]) / 3)[:, None]
org = idx.get_level_values(0).values; T = set(pd.read_csv("data/targets.csv").style_color); tg = np.isin(idx.get_level_values(1), list(T))
rows = []
for k, l in L.items():
    f = np.expm1(l)
    for o in np.unique(org):
        for seg, m0 in [("all", np.ones(len(A), bool)), ("targets", tg)]:
            m = (org == o) & m0; a = A[m]; e = np.log1p(a)[:, None] - l[m]
            rows.append(dict(model=k, origin=o, seg=seg, wape=np.abs(f[m, 2] - a).sum() / a.sum(), bias=f[m, 2].sum() / a.sum() - 1,
                             pinball=np.maximum(QV * e, (QV - 1) * e).mean(), cov50=((e[:, 1] <= 0) & (e[:, 3] >= 0)).mean() if False else ((np.log1p(a) >= l[m, 1]) & (np.log1p(a) <= l[m, 3])).mean(),
                             cov80=((np.log1p(a) >= l[m, 0]) & (np.log1p(a) <= l[m, 4])).mean()))
R = pd.DataFrame(rows)
for seg in ["all", "targets"]:
    x = R[R.seg == seg]; rep = x.origin >= "2025-06"
    S = pd.concat({"13 origins": x.groupby("model")[["wape", "bias", "pinball", "cov50", "cov80"]].mean(), "9 representative": x[rep].groupby("model")[["wape", "bias", "pinball", "cov50", "cov80"]].mean()}, axis=1)
    base = x[x.model == "qmodel"].set_index("origin").wape
    S[("wins vs qmodel", "13")] = [int((x[x.model == k].set_index("origin").wape < base).sum()) for k in S.index]
    S[("wins vs qmodel", "rep9")] = [int((x[(x.model == k) & rep].set_index("origin").wape < base[base.index >= "2025-06"]).sum()) for k in S.index]
    print(f"\n== {seg}"); print(S.round(3).to_string())
print("\nper origin WAPE (all):"); print(R[R.seg == "all"].pivot(index="model", columns="origin", values="wape").round(3).to_string())
