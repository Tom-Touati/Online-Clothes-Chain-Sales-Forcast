"""Score the D112 context-feature variants of the quantile model (data/q_backtest_ctx_*.parquet) against the stored production
backtest (data/q_backtest.parquet, optionally its rerun q_backtest_ctx_base) on the same items as eda/86/88.
For each variant: the quantile model alone and the three-way combination rebuilt with it (rules x quantile x NN), scored by WAPE,
MAE + |bias| (per origin), pinball (each variant's own P10-P90, widened k = 1.25 around its median), and ratios to R3 and to R3 + rules.
Subsets: all items, occasion-tagged items, the 100 targets; nine recent origins and all 13. Paired item bootstrap against the base."""
import sys, glob, numpy as np, pandas as pd, contextlib, io
PFX = sys.argv[1] if len(sys.argv) > 1 else "ctx"          # ctx = D112 context features, mrl = D113 Matryoshka embeddings
BASEF = sys.argv[2] if len(sys.argv) > 2 else "data/q_backtest.parquet"          # optional baseline file (default: the stored production backtest)
OUT_SIM = sys.argv[1:]; sys.argv = sys.argv[:1]          # eda/86 (exec'd below) must not see the prefix argument
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
src = open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
QC = ["q10", "q25", "q50", "q75", "q90"]; TAU = [0.1, 0.25, 0.5, 0.75, 0.9]
V = {"base (stored)": BASEF} | {"+" + p.replace("\\", "/").split(PFX + "_")[-1][:-8]: p for p in sorted(glob.glob(f"data/q_backtest_{PFX}_*.parquet"))}
key = pd.MultiIndex.from_arrays([T.origin, T.style_color])
for nm, p in V.items():
    q = pd.read_parquet(p).set_index(["origin", "style_color"]).reindex(key)
    assert q.q50.notna().all(), nm
    T[f"q|{nm}"] = q.q50.values
    L = np.log1p(q[QC].values); L = L[:, [2]] + K * (L - L[:, [2]])
    for j, c in enumerate(QC): T[f"{c}|{nm}"] = np.expm1(L[:, j])
    T[f"rqn|{nm}"] = np.exp((np.log(T.rules.clip(lower=0.5)) + np.log(T[f"q|{nm}"].clip(lower=0.5)) + np.log(T.nn.clip(lower=0.5))) / 3)
T["occ_tagged"] = T.occ == "tagged"
def pinball(F, nm, centre):
    shift = np.log1p(F[centre].values) - np.log1p(F[f"q50|{nm}"].values); tot = 0
    for c, tau in zip(QC, TAU):
        Q = np.expm1(np.log1p(F[f"{c}|{nm}"].values) + shift); u = F.y.values - Q; tot += np.maximum(tau * u, (tau - 1) * u).sum()
    return tot / (len(QC) * F.y.sum())
def mb(F, c): e = F[c] - F.y; return (e.abs().groupby(F.origin).sum() + e.groupby(F.origin).sum().abs()).div(F.y.groupby(F.origin).sum()).mean()
def rows(F):
    """Every metric as a ratio to R3 (< 1 = better than recency); R3 + rules (the rule agents) is one of the rows.
    Pinball for R3 and R3 + rules uses the stored quantile model's spread around their own forecast."""
    e3 = (F.R3 - F.y).abs(); eru = (F.rules - F.y).abs(); mb3 = mb(F, "R3"); p3 = pinball(F, "base (stored)", "R3")
    def r(c, nm):
        e = (F[c] - F.y).abs()
        return {"WAPE / R3": e.sum() / e3.sum(), "relMAE vs R3": np.exp(np.mean(np.log((e + 1) / (e3 + 1)))), "win rate vs R3": ((e < e3) + 0.5 * (e == e3)).mean(), "win rate vs R3+rules": ((e < eru) + 0.5 * (e == eru)).mean(),
                "MAE+|bias| / R3": mb(F, c) / mb3, "pinball / R3": pinball(F, nm, c) / p3, "bias (context)": F[c].sum() / F.y.sum() - 1}
    out = {("R3 + rules (rule agents)", ""): r("rules", "base (stored)") | {"sim loss / R3": SL("rules")}}
    for nm in V:
        out[("quantile", nm)] = r(f"q|{nm}", nm) | {"sim loss / R3": SL("qmodel", nm)}
        out[("rules+quantile+NN", nm)] = r(f"rqn|{nm}", nm) | {"sim loss / R3": SL("r_q_nn", nm)}
    print(f"   R3 itself: WAPE {e3.sum() / F.y.sum():.3f} | MAE+|bias| {mb3:.3f} | pinball {p3:.3f} | bias {F.R3.sum() / F.y.sum() - 1:+.3f}")
    return pd.DataFrame(out).T
# buying-simulation loss (production policy, 11 windows Mar 2025 - Jan 2026, all items): mean loss / R3's mean loss
S0 = pd.read_csv("data/sim_systems_windows.csv"); S0 = S0[S0.policy == "production"].groupby("system").loss_pct.mean()
try: S1 = pd.read_csv(f"data/sim_{PFX}_windows.csv").groupby(["system", "variant"]).loss_pct.mean()
except FileNotFoundError: S1 = pd.Series(dtype=float)
def SL(system, nm="base (stored)"):
    if not SIMCOL["on"]: return np.nan
    if nm == "base (stored)": return S0[system] / S0["R3"]
    k = (system, nm.lstrip("+")); return S1[k] / S0["R3"] if k in S1.index else np.nan
SIMCOL = {"on": False}
print(f"R3 buying-simulation loss: {S0['R3']:.4f} of perfect-foresight profit (production policy, 11 windows)")
REC_ = [o for o in ORIGINS if o >= "2025-06"]
SUB = {"nine recent origins, all items": T.origin.isin(REC_), "nine recent, occasion-tagged items": T.origin.isin(REC_) & T.occ_tagged,
       "nine recent, the 100 targets": T.origin.isin(REC_) & T.target, "all 13 origins, all items": np.ones(len(T), bool), "all 13, occasion-tagged": T.occ_tagged}
for s, m in SUB.items():
    SIMCOL["on"] = s == "nine recent origins, all items"      # the simulation covers all items, Mar 2025 - Jan 2026
    F = T[m]; print(f"\n######## {s}: {len(F)} item-origins"); print(rows(F).round(4).to_string())
rng = np.random.default_rng(0)
print("\n== paired bootstrap, variant minus base (stored), nine recent origins: WAPE | pinball  (quantile model alone; three-way)")
for s in ["nine recent origins, all items", "nine recent, occasion-tagged items"]:
    F = T[SUB[s]]; g = [x for _, x in F.groupby("origin")]; B = [pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in g]) for _ in range(400)]
    print(f"-- {s}")
    for nm in [v for v in V if v != "base (stored)"]:
        res = []
        for lab, pre in [("q", "q|"), ("3way", "rqn|")]:
            dw = [((x[pre + nm] - x.y).abs().sum() - (x[pre + "base (stored)"] - x.y).abs().sum()) / x.y.sum() for x in B]
            dp = [pinball(x, nm, pre + nm) - pinball(x, "base (stored)", pre + "base (stored)") for x in B]
            res.append(f"{lab}: WAPE [{np.percentile(dw, 2.5):+.4f}, {np.percentile(dw, 97.5):+.4f}] pinball [{np.percentile(dp, 2.5):+.4f}, {np.percentile(dp, 97.5):+.4f}]")
        print(f"  {nm:8s} " + " | ".join(res))
