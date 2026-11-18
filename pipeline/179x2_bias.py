"""D245: are the systematic biases found in D244 stable enough to correct? (final system F = p2c32a, data/dec_base.parquet)
D244: small and mid items under-forecast (-17% to -28%), young items -17%, low-discount items -12%, heavily discounted +9%.
Segments: volume quintile (by R3, cut points from each origin's own items), age band, discount band, and their crossings.
Time-honest correction: at origin o, each segment's log ratio of summed actual to summed F over earlier origins with known outcomes
(o' + 4 <= o), shrunk by k / (k + n_origins) toward 0 and applied as F x exp(lambda x ratio), lambda in 0.5 / 1.
Also the stability check: correlation of a segment's bias across consecutive origins. Standard table + bootstrap against F."""
import sys, contextlib, io, importlib.util, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
import os; SUF = os.environ.get("DEC", ""); B = pd.read_parquet(f"data/dec_base{SUF}.parquet")   # D257: DEC=_<tag> refits on another base system
B["vol"] = B.groupby("origin").R3.transform(lambda v: pd.qcut(np.log1p(v), int(os.environ.get("VOLQ", 5)), labels=False, duplicates="drop"))   # D363: VOLQ volume bands
B["age_b"] = pd.cut(B.age_m, [-1, 3, 6, 12, 999], labels=False); B["disc_b"] = pd.cut(B.disc_item.fillna(0.55), [-1, 0.55, 0.65, 1.1], labels=False)
SEGS = {"vol": ["vol"], "age": ["age_b"], "disc": ["disc_b"], "vol_age": ["vol", "age_b"], "vol_age_disc": ["vol", "age_b", "disc_b"]}
oo = pd.PeriodIndex(B.origin, freq="M").astype(int)
print("== stability: correlation of segment log-bias between consecutive origins")
for nm, cols in SEGS.items():
    sb = B.groupby(["origin"] + cols).apply(lambda d: np.log((d.y.sum() + 1) / (d.F.sum() + 1))).unstack(cols)
    c = [np.corrcoef(sb.iloc[i].values, sb.iloc[i + 4].values)[0, 1] for i in range(len(sb) - 4) if np.isfinite(sb.iloc[i].values).sum() > 3]
    print(f"  {nm:13s} lag-4 (non-overlapping) correlation of segment biases: mean {np.nanmean(c):+.3f}")
for nm, cols in SEGS.items():
    for lam in (0.5, 1.0):
        col = f"{nm}_{lam}"; B[col] = B.F; B[col + "n"] = B.F
        for o in sorted(B.origin.unique()):
            tr = oo + 4 <= pd.Period(o).ordinal
            if B.origin[tr].nunique() < 2: continue
            g = B[tr].groupby(cols).apply(lambda d: pd.Series({"r": np.log((d.y.sum() + 1) / (d.F.sum() + 1)), "n": d.origin.nunique()}))
            g["r"] = g.r * g.n / (g.n + float(os.environ.get("SHRK", 2))); m = (B.origin == o).values   # D363: SHRK shrink constant
            adj = B[m][cols].merge(g.reset_index(), on=cols, how="left").r.fillna(0).values
            B.loc[m, col] = B.F[m] * np.exp(lam * adj)
            B.loc[m, col + "n"] = B.loc[m, col] * B.F[m].sum() / B.loc[m, col].sum()   # redistribution only: origin total kept
if os.environ.get("NOSCORE") == "1": B[["origin", "style_color", "F"] + [f"{nm}_{lam}{x}" for nm in SEGS for lam in (0.5, 1.0) for x in ("", "n")]].to_parquet("data/bias_seg_preds" + SUF + os.environ.get("OUTTAG", "") + ".parquet", index=False); print("saved without scoring (NOSCORE)"); sys.exit()   # D352: server runs
with contextlib.redirect_stdout(io.StringIO()): exec(open("pipeline/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
spec = importlib.util.spec_from_file_location("sl", "pipeline/score_lib.py"); sl = importlib.util.module_from_spec(spec); spec.loader.exec_module(sl)
key = pd.MultiIndex.from_arrays([T.origin, T.style_color]); Bi = B.set_index(["origin", "style_color"]).reindex(key)
COLS = ["F"] + [f"{nm}_{lam}{x}" for nm in SEGS for lam in (0.5, 1.0) for x in ("", "n")]
for c in COLS: T[c] = Bi[c].values
REC_ = [o for o in ORIGINS if o >= "2025-06"]
for nm, m in [("nine recent origins", T.origin.isin(REC_)), ("all 13 origins", np.ones(len(T), bool))]:
    print(f"\n######## {nm}"); print(sl.table(T[m], COLS, D).round(4).to_string())
F_ = T[T.origin.isin(REC_)]; print("\n== bootstrap vs F, recent"); [print(f"  {c:18s}", np.round(sl.boot(F_, c, "F"), 4)) for c in COLS[1:]]
B[["origin", "style_color", "F"] + [c for c in COLS if c != "F"]].to_parquet("data/bias_seg_preds" + SUF + os.environ.get("OUTTAG", "") + ".parquet", index=False)
