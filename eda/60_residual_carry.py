"""Online residual correction of the LightGBM ensemble ("the model's own recent miss carries over").
At origin o the miss of the forecast made one month earlier for month o (h=1, log((y+1)/(pred+1))) is known for every item.
Correction: pred(o,h) * exp(b_h * miss_item + c_h * miss_style), where miss_style = mean miss of the item's sibling colours
(hierarchical: item -> style), and b_h, c_h are least-squares fits on earlier origins only (pairs whose target month <= o).
Also reports the persistence of misses at the product-type level. Writes data/carry_backtest_preds.parquet (pred column)."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
SHR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
B = pd.read_parquet("data/best_backtest_preds.parquet")
B["o"] = pd.PeriodIndex(B.origin, freq="M").asi8; B["tm"] = B.o + B.h
B["r"] = np.log((B.y + 1) / (B.pred + 1)); B["style"] = B.style_color.str.split("_").str[0]
m1 = B[B.h == 1][["style_color", "style", "tm", "r"]].rename(columns={"r": "miss"})      # miss of forecast for month tm, known at origin tm
m1["miss_style"] = m1.groupby(["style", "tm"]).miss.transform(lambda v: (v.sum() - v) / max(len(v) - 1, 1) if len(v) > 1 else np.nan)
B = B.merge(m1[["style_color", "tm", "miss", "miss_style"]].rename(columns={"tm": "o"}), on=["style_color", "o"], how="left")
B[["miss", "miss_style"]] = B[["miss", "miss_style"]].fillna(0)
out = []; coefs = []
for o in sorted(B.o.unique()):
    te = B[B.o == o].copy(); tr = B[(B.tm <= o) & (B.o < o)]
    te["adj"] = 0.0
    for h in range(1, 5):
        t = tr[tr.h == h]
        if len(t) < 300: continue
        X = t[["miss", "miss_style"]].values; b = np.linalg.lstsq(X, t.r.values, rcond=None)[0]
        # robust: shrink toward 0 by half (few origins), clip
        b = np.clip(b * SHR, -0.5, 0.8); coefs.append(dict(origin=te.origin.iloc[0], h=h, b_item=b[0], b_style=b[1], n=len(t)))
        mk = te.h == h; te.loc[mk, "adj"] = te.loc[mk, ["miss", "miss_style"]].values @ b
    te["pred_c"] = (te.pred + 1) * np.exp(te.adj) - 1
    out.append(te)
X = pd.concat(out); print(pd.DataFrame(coefs).round(3).to_string(index=False))
def sc(g, c):
    t = g.groupby("style_color")[[c, "y"]].sum(); return (t[c] - t.y).abs().sum() / t.y.sum()
R = X.groupby("origin").apply(lambda g: pd.Series({"lgb": sc(g, "pred"), "carry": sc(g, "pred_c")}))
print(R.round(3).T.to_string()); print("mean", R.mean().round(4).to_dict(), "| origins improved", int((R.carry < R.lgb).sum()), "of", len(R))
T = set(pd.read_csv("data/targets.csv").style_color)
Rt = X[X.style_color.isin(T)].groupby("origin").apply(lambda g: pd.Series({"lgb": sc(g, "pred"), "carry": sc(g, "pred_c")}))
print("targets mean", Rt.mean().round(4).to_dict(), "| improved", int((Rt.carry < Rt.lgb).sum()), "of", len(Rt))
X[["origin", "style_color", "h", "pred_c", "y"]].rename(columns={"pred_c": "pred"}).to_parquet("data/carry_backtest_preds.parquet")
