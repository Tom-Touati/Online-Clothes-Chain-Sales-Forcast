"""D177 (overnight M5 goal, direction 7), runs on the GPU server: multi-horizon neural forecasters with known-future inputs
(neuralforecast). Weekly item series (data/weekly_item.parquet), known-future weekly inputs (events, normal temperature at the site
and where the item sells, site campaign intensity, week of year), static inputs (product type, LLM scores).
At each of the 13 backtest origins (month ends 2025-02 .. 2026-02) every model is refitted on weeks ending by the origin and forecasts
the weeks up to the end of the 4th target month; weekly quantiles are spread over days and summed over the 4 target months.
Models (argv; default all): nhits, tft, deepar (and *_log: trained on log1p(units) with standard scaling, back-transformed) (MQLoss / NegativeBinomial), and *_emb variants adding the origin's v4 style embedding
(8 principal components) as static inputs. Writes data/q_backtest_nf_<model>.parquet (origin, style_color, q10..q90)."""
import os, sys, time, numpy as np, pandas as pd, torch, logging
torch.backends.cudnn.enabled = False   # cuDNN sub-library fails to load on the server build
os.chdir(os.path.expanduser("~/edikted_main")); logging.getLogger("pytorch_lightning").setLevel(logging.ERROR); logging.getLogger("lightning").setLevel(logging.ERROR)
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT, DeepAR
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
RUN = sys.argv[1:] or ["nhits", "tft", "deepar", "nhits_emb", "tft_emb"]
Y = pd.read_parquet("data/weekly_item.parquet"); FU = pd.read_parquet("data/weekly_futr.parquet"); FI = pd.read_parquet("data/weekly_futr_item.parquet")
ST = pd.read_parquet("data/static_item.parquet"); EMB = np.load("data/style_emb_v4.npz", allow_pickle=True); emb_items = EMB["items"].astype(str)
FUT = [c for c in FU.columns if c != "ds"] + ["t_norm_item"]; STAT = [c for c in ST.columns if c != "unique_id"]
Y = Y.merge(FU, on="ds", how="left").merge(FI, on=["unique_id", "ds"], how="left")
LEVEL = [50, 80]
def model(name, h, stat):
    common = dict(h=h, input_size=52, futr_exog_list=FUT, stat_exog_list=stat, max_steps=600, scaler_type=("standard" if "log" in name else "robust"), start_padding_enabled=True,
                  enable_progress_bar=False, logger=False, enable_model_summary=False, random_seed=0)
    if name.startswith("nhits"): return NHITS(loss=MQLoss(level=LEVEL), batch_size=64, windows_batch_size=256, **common)
    if name.startswith("tft"): return TFT(loss=MQLoss(level=LEVEL), hidden_size=64, batch_size=64, windows_batch_size=256, **common)
    return DeepAR(loss=DistributionLoss("NegativeBinomial", level=LEVEL), batch_size=64, windows_batch_size=256, **{k: v for k, v in common.items() if k != "start_padding_enabled"}, start_padding_enabled=True)
for name in RUN:
    out, t0 = [], time.time()
    for o in [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]:
        E = pd.Period(o, "M").to_timestamp(how="end").normalize(); E4 = (pd.Period(o, "M") + 4).to_timestamp(how="end").normalize()
        tr = Y[Y.ds + pd.Timedelta(days=6) <= E]; last = tr.ds.max(); h = int(np.ceil(((E4 - last).days - 6) / 7)) + 1
        stat = ST.copy(); sl = STAT
        if name.endswith("_emb") and o in EMB.files:
            V = pd.DataFrame(EMB[o].astype(float), index=emb_items).reindex(stat.unique_id).fillna(0).values
            Vc = V - V.mean(0); u, s, _ = np.linalg.svd(Vc, full_matrices=False); pcs = u[:, :8] * s[:8] / np.sqrt(len(V))
            for j in range(8): stat[f"emb{j}"] = pcs[:, j]
            sl = STAT + [f"emb{j}" for j in range(8)]
        ids = tr.unique_id.unique(); stat = stat[stat.unique_id.isin(ids)]
        nf = NeuralForecast(models=[model(name, h, sl)], freq="W-MON")
        trf = tr.assign(y=np.log1p(tr.y)) if "log" in name else tr       # *_log: model log1p(units), back-transform the quantiles
        nf.fit(df=trf[["unique_id", "ds", "y"] + FUT], static_df=stat[["unique_id"] + sl])
        fd = pd.DataFrame([(u, last + pd.Timedelta(weeks=k)) for u in ids for k in range(1, h + 1)], columns=["unique_id", "ds"])
        fd = fd.merge(FU, on="ds", how="left").merge(FI, on=["unique_id", "ds"], how="left")
        P = nf.predict(futr_df=fd[["unique_id", "ds"] + FUT]).reset_index()
        mn = [c for c in P.columns if c not in ("unique_id", "ds", "index")]; base = mn[0].split("-")[0]
        col = {"q50": f"{base}-median" if f"{base}-median" in P else base, "q25": f"{base}-lo-50", "q75": f"{base}-hi-50", "q10": f"{base}-lo-80", "q90": f"{base}-hi-80"}
        # fraction of each forecast week's days that fall in the 4 target months
        frac = np.array([sum(1 for k in range(7) if E < d + pd.Timedelta(days=k) <= E4) / 7 for d in P.ds])
        for q, c in col.items(): P[q] = np.clip(np.expm1(P[c].values) if "log" in name else P[c].values, 0, None) * frac
        R = P.groupby("unique_id")[list(col)].sum().reset_index().rename(columns={"unique_id": "style_color"}); R.insert(0, "origin", o)
        out.append(R); print(name, o, "h", h, "series", len(ids), f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(out).to_parquet(f"data/q_backtest_nf_{name}.parquet", index=False); print("saved", name, flush=True)
