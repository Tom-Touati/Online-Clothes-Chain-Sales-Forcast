"""Each model against two baselines (D111): R3 (last 3 months x 4) and R3 + rules (the rule agents: recent daily rate x rule multipliers).
For every model and baseline: ratio = model / baseline (< 1 = model better) of WAPE, MAE + |bias| (per origin), pinball (P10-P90),
buying-simulation loss (production policy, eda/87), plus relMAE (typical-item error ratio, Davydenko-Fildes) and the item win rate.
95% item bootstrap intervals for the WAPE ratio and relMAE. Uses eda/86 definitions and data/sim_systems_windows.csv."""
import numpy as np, pandas as pd, contextlib, io
src = open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
MODELS = {"qmodel": "Quantile model", "lgb": "LightGBM", "nn": "Embedding NN", "lgb_nn": "LightGBM + NN", "q_nn": "Quantile + NN",
          "r_q": "Rules + quantile", "r_q_nn": "Rules + quantile + NN"}
BASE = {"R3": "R3", "rules": "R3 + rules"}
SIM = pd.read_csv("data/sim_systems_windows.csv"); SIM = SIM[SIM.policy == "production"].pivot(index="start", columns="system", values="loss_pct")
rng = np.random.default_rng(0)
def mae_bias(F, c):
    e = F[c] - F.y; return (e.abs().groupby(F.origin).sum() + e.groupby(F.origin).sum().abs()).div(F.y.groupby(F.origin).sum()).mean()
def table(F, boot=True):
    g = [x for _, x in F.groupby("origin")]; B = [pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g]) for _ in range(300)] if boot else []
    rows = []
    for b, bn in BASE.items():
        eb = (F[b] - F.y).abs(); pb = pin(F, b)[0]
        for c, cn in MODELS.items():
            e = (F[c] - F.y).abs(); r = {"baseline": bn, "model": cn}
            r["WAPE ratio"] = e.sum() / eb.sum()
            r["relMAE"] = np.exp(np.mean(np.log((e + 1) / (eb + 1))))
            r["win rate"] = ((e < eb) + 0.5 * (e == eb)).mean()
            r["MAE+|bias| ratio"] = mae_bias(F, c) / mae_bias(F, b)
            r["pinball ratio"] = pin(F, c)[0] / pb
            if boot:
                w = [((s[c] - s.y).abs().sum() / (s[b] - s.y).abs().sum()) for s in B]
                m = [np.exp(np.mean(np.log(((s[c] - s.y).abs() + 1) / ((s[b] - s.y).abs() + 1)))) for s in B]
                r["WAPE ratio 95%"] = f"[{np.percentile(w, 2.5):.3f}, {np.percentile(w, 97.5):.3f}]"; r["relMAE 95%"] = f"[{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"
            if F is TR_ and c in SIM:
                r["sim loss ratio"] = SIM[c].mean() / SIM[b].mean(); r["sim windows won"] = f"{int((SIM[c] < SIM[b]).sum())}/11"
            rows.append(r)
    return pd.DataFrame(rows)
REC_ = [o for o in ORIGINS if o >= "2025-06"]; TR_ = T[T.origin.isin(REC_)]
pd.set_option("display.width", 250)
for nm, F in [("NINE RECENT ORIGINS (simulation: 11 windows Mar 2025 - Jan 2026)", TR_), ("ALL 13 ORIGINS", T), ("RECENT, THE 100 TARGETS", T[T.origin.isin(REC_) & T.target])]:
    t = table(F)
    print(f"\n######## {nm}: {len(F)} item-origins")
    for bn in BASE.values():
        print(f"-- against {bn}"); print(t[t.baseline == bn].drop(columns="baseline").set_index("model").round(3).to_string())
