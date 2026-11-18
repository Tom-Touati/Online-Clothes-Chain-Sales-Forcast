"""D198 (fork): features from the dynamic co-occurrence network (pipeline/181; passed the D199 cart gate) for the trees. At each origin o,
from the network fitted on months <= o (v(o), and its 3-month move v(o) - v(o-3)); 20 nearest neighbours of other styles alive in the
last 3 months, weights = cosine similarity clipped at 0:
  dy_nb_sea4, dy_nb_lvl, dy_nb_mom3, dy_nb_yoy   as pipeline/153 (neighbours' last-year ramp into the target months, level, momentum, growth)
  dy_vel        |v(o) - v(o-3)| / |v(o)|: how far the item's co-purchase position moved in 3 months
  dy_fut_lvl, dy_fut_mom3   level and momentum of the neighbours of the projected position v(o) + (v(o) - v(o-3)) / 3 (one month on)
Writes data/fk_dyn_feats.parquet."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
Z = np.load("data/fk_cooc_dyn.npz", allow_pickle=True); items = Z["items"].astype(str); n = len(items); sty = pd.Series(items).str.split("_").str[0].values
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]; M = M.reindex(items).fillna(0)
Pm = lambda o, k: str(pd.Period(o, "M") + k); U = lambda o, ks: M.reindex(columns=[Pm(o, k) for k in ks]).fillna(0).values
rown = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
rows = []
for o in [k for k in Z.files if len(k) == 7 and k[4] == "-"]:
    V = Z[o].astype(np.float32); vel = Z[o + "_vel"].astype(np.float32)
    l3 = U(o, [-2, -1, 0]); p3 = U(o, [-5, -4, -3]); y3 = U(o, [-14, -13, -12]); yt = U(o, [-11, -10, -9, -8]); live = l3.sum(1) > 0
    lvl = l3.mean(1); mom3 = np.log((l3.sum(1) + 1) / (p3.sum(1) + 1)); yoy = np.where(y3.sum(1) > 0, np.log((l3.sum(1) + 1) / (y3.sum(1) + 1)), np.nan)
    def nbr(A):
        S = rown(A) @ rown(V).T; S[:, ~live] = -9; S[sty[:, None] == sty[None, :]] = -9
        nb = np.argsort(-S, 1)[:, :20]; w = np.clip(np.take_along_axis(S, nb, 1), 0, None); return nb, w
    nb, w = nbr(V); ws = w.sum(1); wm = lambda x: np.where(ws > 0, (w * x[nb]).sum(1) / np.maximum(ws, 1e-9), np.nan)
    wy = w * np.isfinite(yoy)[nb]; sy3 = (w * y3.sum(1)[nb]).sum(1); syt = (w * yt.sum(1)[nb]).sum(1)
    nf, wf = nbr(V + vel / 3); wfs = wf.sum(1); wfm = lambda x: np.where(wfs > 0, (wf * x[nf]).sum(1) / np.maximum(wfs, 1e-9), np.nan)
    rows.append(pd.DataFrame({"origin": o, "style_color": items, "dy_nb_sea4": np.where(sy3 > 0, np.log((syt + 1) / (sy3 * 4 / 3 + 1)), np.nan), "dy_nb_lvl": np.log1p(wm(lvl)),
                              "dy_nb_mom3": wm(mom3), "dy_nb_yoy": np.where(wy.sum(1) > 0, (wy * np.nan_to_num(yoy)[nb]).sum(1) / np.maximum(wy.sum(1), 1e-9), np.nan),
                              "dy_vel": np.linalg.norm(vel, axis=1) / np.maximum(np.linalg.norm(V, axis=1), 1e-9), "dy_fut_lvl": np.log1p(wfm(lvl)), "dy_fut_mom3": wfm(mom3)}))
X = pd.concat(rows, ignore_index=True); X.to_parquet("data/fk_dyn_feats.parquet", index=False); print(X.shape, X.origin.min(), X.origin.max()); print(X.describe().round(3).T[["mean", "std"]].to_string())
