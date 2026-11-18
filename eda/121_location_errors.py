"""Per-location error analysis of the bottom-up model (D125c): where does its -20% bias come from? Recent origins, Tweedie version.
By location type and by location: units share, bias of the location forecast on covered rows, actual / (4 x location level) growth vs
predicted growth; plus the units the bottom-up cannot see (new locations for the item) by type. Figure: eda/figs/err_locations.png."""
import numpy as np, pandas as pd, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
L = pd.read_parquet("data/bu_loc_tw.parquet"); L = L[L.origin >= "2025-06"]; L["ltype"] = L.ltype.astype(str); L["loc"] = L["loc"].astype(str)
def agg(x): return pd.Series({"rows": len(x), "actual units": x.ytot_loc.sum(), "bias": x.pred.sum() / x.ytot_loc.sum() - 1,
                              "actual growth vs 4x level": x.ytot_loc.sum() / (4 * x.lv.sum()), "predicted growth": x.pred.sum() / (4 * x.lv.sum()),
                              "WAPE": (x.pred - x.ytot_loc).abs().sum() / x.ytot_loc.sum()})
print("== by location type (covered rows, recent origins)"); print(L.groupby("ltype").apply(agg).round(3).to_string())
print("\n== by location (largest 20)"); G = L.groupby("loc").apply(agg).sort_values("actual units", ascending=False); print(G.head(20).round(3).to_string())
print("\n== by origin x type: bias"); print(L.groupby(["origin", "ltype"]).apply(lambda x: x.pred.sum() / x.ytot_loc.sum() - 1).unstack().round(2).to_string())
print("\n== by origin x type: actual growth vs 4 x level"); print(L.groupby(["origin", "ltype"]).apply(lambda x: x.ytot_loc.sum() / (4 * x.lv.sum())).unstack().round(2).to_string())
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
P = L.groupby(["origin", "ltype"]).apply(lambda x: x.pred.sum() / x.ytot_loc.sum() - 1).unstack()
for c in P.columns: ax[0].plot(P.index, P[c], marker="o", label=c)
ax[0].axhline(0, color="k", lw=0.6); ax[0].set_title("Bottom-up bias by origin and location type (covered rows)"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=45)
Gs = G[G.index.str.startswith("store_")]
ax[1].scatter(Gs["actual growth vs 4x level"], Gs["predicted growth"], s=np.sqrt(Gs["actual units"]) / 3)
for n, r in Gs.iterrows(): ax[1].annotate(n.replace("store_", ""), (r["actual growth vs 4x level"], r["predicted growth"]), fontsize=7)
lim = [0, max(Gs["actual growth vs 4x level"].max(), Gs["predicted growth"].max()) * 1.1]; ax[1].plot(lim, lim, color="k", lw=0.6)
ax[1].set_xlabel("actual 4-month units / (4 x level)"); ax[1].set_ylabel("predicted"); ax[1].set_title("Stores: realised vs predicted growth (dot size = units)")
fig.tight_layout(); fig.savefig("eda/figs/err_locations.png", dpi=60)
