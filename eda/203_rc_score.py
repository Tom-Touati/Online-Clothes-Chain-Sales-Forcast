"""D332 (fork 2): scores the location recommender (eda/202, data/rc_recsys_preds_<tag>.parquet) the way the three sessions agreed:
WAPE / R3 on the nine recent origins (from 2025-06) and on all 13, share of items within +/-20%, bias; alone, as a sixth member of the
arithmetic mean of the five (rules, qdyn = ec_ns, nn, pf2c, pf2d32; data/dec_base_ecnsnn.parquet), and swapped in for each member;
with an item bootstrap of the difference against the five-member mean. The submitted system (geo, D273 corrections on top) is shown
for reference; Main re-fits the corrections before a member is judged in the full system.
Usage: python eda/203_rc_score.py tag=_e10   (stdout -> eda/rc_score_<tag>.txt)"""
import sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220)
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); TAG = KW.get("tag", "")
B = pd.read_parquet("data/dec_base_ecnsnn.parquet"); G = pd.read_parquet("data/geo_ecnsnn_x_preds.parquet")[["origin", "style_color", "geo"]]; R = pd.read_parquet(f"data/rc_recsys_preds{TAG}.parquet").drop(columns="y")
D = B.merge(G, on=["origin", "style_color"]).merge(R, on=["origin", "style_color"], how="left"); MEM = ["rules", "qdyn", "nn", "pf2c", "pf2d32"]; y = D.y.values; rec = (D.origin >= "2025-06").values; allm = np.ones(len(D), bool)
ar = np.abs(D.R3.values - y); rng = np.random.default_rng(0); FC = [c for c in R.columns if c.startswith("f_")]
print("coverage of the recommender (share of backtest rows with a forecast):", {c: round(float(D[c].notna().mean()), 3) for c in FC})
for c in FC: D[c] = D[c].fillna(D.R3)   # an item with no location history gets R3
def sc(f, m): e = np.abs(f - y)[m]; return e.sum() / ar[m].sum(), float((e <= 0.2 * y[m]).mean()), f[m].sum() / y[m].sum() - 1
def boot(fa, fb, B_=1000):
    Z = pd.DataFrame({"s": D.style_color, "a": np.abs(fa - y), "b": np.abs(fb - y), "r": ar})[rec].groupby("s").sum().values; n = len(Z); T = Z[rng.integers(0, n, (B_, n))].sum(1); d = T[:, 1] / T[:, 2] - T[:, 0] / T[:, 2]
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
base = D[MEM].values.mean(1); rows = []
def add(lab, f, ref=True):
    w, h, b = sc(f, rec); wa, ha, _ = sc(f, allm); rows.append(dict(system=lab, recent=round(w, 4), all13=round(wa, 4), hit=round(h, 3), hit_all=round(ha, 3), bias=round(b, 3), ci=boot(base, f) if ref else None))
add("R3", D.R3.values, False); add("R3 + rules", D.rules.values, False); add("five-member mean (E:ecnsnn, before corrections)", base, False); add("submitted system (geo)", D.geo.values, False)
for c in FC: add(f"recommender alone: {c[2:]}", D[c].values)
for c in [c for c in FC if c.startswith("f_honest")]:
    add(f"six-member mean, + {c[2:]}", (D[MEM].values.sum(1) + D[c].values) / 6)
    for m in MEM: add(f"  {c[2:]} in place of {m}", (D[MEM].values.sum(1) - D[m].values + D[c].values) / 5)
for c in [c for c in FC if c.startswith("f_plan")]: add(f"six-member mean, + {c[2:]} (plan known: bound)", (D[MEM].values.sum(1) + D[c].values) / 6)
T = pd.DataFrame(rows); print(T.to_string(index=False))
po = []
for o in sorted(D.origin.unique()):
    m = (D.origin == o).values; po.append(dict(origin=o, **{k: round(sc(f, m)[0], 3) for k, f in [("R3+rules", D.rules.values), ("mean5", base), ("rc_honest_avg", D.f_honest_avg.values), ("six", (D[MEM].values.sum(1) + D.f_honest_avg.values) / 6), ("rc_plan_avg", D.f_plan_avg.values)]}))
print(pd.DataFrame(po).to_string(index=False)); corr = np.corrcoef(np.log1p(D.loc[rec, MEM + ["f_honest_avg"]].values).T)[-1, :-1]; print("log correlation of the recommender with each member (recent):", dict(zip(MEM, corr.round(3))))
json.dump(dict(rows=rows, by_origin=po, corr=dict(zip(MEM, corr.round(3).tolist()))), open(f"data/rc_score{TAG}.json", "w"), indent=0)
