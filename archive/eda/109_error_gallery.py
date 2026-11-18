"""Visual error gallery (D123): weekly sales histories behind the largest misses of the three-way combination (rules x quantile x NN),
nine recent origins, 4-month totals. Each panel: weekly units (all channels, black), online units (grey), retail units (orange),
the origin (vertical line), the forecast window shaded; the three-way's predicted weekly rate over the window (blue dashed), the members'
(rules green, quantile purple, NN red, thin), and the actual weekly rate (black dashed); weekly online discount on a second axis (light).
Usage: python eda/109_error_gallery.py [under|over|typical|consensus|young] -> eda/figs/err_<kind>.png (24 panels)."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
KIND = sys.argv[1] if len(sys.argv) > 1 else "under"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 24; SKIP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
import os
T = pd.read_parquet("data/error_map_items.parquet"); T = T[T.origin >= "2025-06"].copy()
ONLY = os.environ.get("GALLERY_ORIGIN")                       # optional: restrict to one origin (e.g. 2025-06, the deliverable analogue)
if ONLY: T = T[T.origin == ONLY].copy()
T["err"] = T.r_q_nn - T.y
if KIND == "under": P = T.sort_values("err").iloc[SKIP:SKIP + N]
elif KIND == "over": P = T.sort_values("err", ascending=False).iloc[SKIP:SKIP + N]
elif KIND == "consensus":                                   # members agree (lowest 40% disagreement) yet off by more than 50%: what no model knows
    lo = T.disagree <= T.disagree.quantile(0.4); off = (np.log((T.r_q_nn + 1) / (T.y + 1))).abs() > np.log(1.5)
    P = T[lo & off].assign(ae=lambda x: x.err.abs()).sort_values("ae", ascending=False).iloc[SKIP:SKIP + N]
    print("consensus misses:", int((lo & off).sum()), "of", int(lo.sum()), "low-disagreement item-origins; share of all abs error:", round(T[lo & off].err.abs().sum() / T.err.abs().sum(), 3),
          "| under:", int(((T.r_q_nn < T.y) & lo & off).sum()), "over:", int(((T.r_q_nn > T.y) & lo & off).sum()))
elif KIND == "young":                                       # launches under 3 months old at the origin, largest absolute errors
    P = T[T.age_m < 3].assign(ae=lambda x: x.err.abs()).sort_values("ae", ascending=False).iloc[SKIP:SKIP + N]
    Y = T[T.age_m < 3]; print("young item-origins:", len(Y), "| under:", int((Y.err < 0).sum()), "over:", int((Y.err > 0).sum()), "| bias", round(Y.r_q_nn.sum() / Y.y.sum() - 1, 3))
else: P = T[(T.y > 500)].sample(N, random_state=1)
d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[(~d.cancelled) & d.style_color.isin(P.style_color)]; d["ny_date"] = pd.to_datetime(d.ny_date)
d["wk"] = d.ny_date.dt.to_period("W").dt.start_time
W = d.groupby(["style_color", "wk"]).qty.sum().unstack(0).fillna(0)
Wo = d[d["class"] == "ONLINE"].groupby(["style_color", "wk"]).qty.sum().unstack(0).reindex(W.index).fillna(0)
Wr = d[d["class"] == "RETAIL"].groupby(["style_color", "wk"]).qty.sum().unstack(0).reindex(W.index).fillna(0)
on = d[d.chan == "edikted.com"]; Wd = (on.groupby(["style_color", "wk"]).disc_w.sum() / on.groupby(["style_color", "wk"]).qty.sum()).unstack(0).reindex(W.index)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
fig, axs = plt.subplots(6, 4, figsize=(24, 26)); axs = axs.ravel()
for ax, (_, r) in zip(axs, P.iterrows()):
    s = r.style_color; O = pd.Period(r.origin, "M"); t0, t1 = O.end_time.normalize(), (O + 4).end_time.normalize(); days = (t1 - t0).days
    lo = t0 - pd.Timedelta(days=420); x = W.index[(W.index >= lo) & (W.index <= t1 + pd.Timedelta(days=60))]
    ax.plot(x, W.loc[x, s] if s in W else 0, color="black", lw=1.4, label="all")
    if s in Wo: ax.plot(x, Wo.loc[x, s], color="grey", lw=0.8, label="online")
    if s in Wr: ax.plot(x, Wr.loc[x, s], color="darkorange", lw=0.8, label="retail")
    ax.axvline(t0, color="k", lw=0.8); ax.axvspan(t0, t1, color="#eef", alpha=0.6)
    for c, col, lw, ls in [("r_q_nn", "blue", 1.8, "--"), ("rules", "green", 0.9, "-"), ("qmodel", "purple", 0.9, "-"), ("nn", "red", 0.9, "-"), ("y", "black", 1.8, ":")]:
        ax.hlines(r[c] / days * 7, t0, t1, color=col, lw=lw, ls=ls)
    if s in Wd:
        a2 = ax.twinx(); a2.plot(x, Wd.loc[x, s], color="teal", lw=0.6, alpha=0.5); a2.set_ylim(0, 1); a2.tick_params(labelsize=6)
    nm = str(it.product_name.get(s, ""))[:28]
    ax.set_title(f"{s} | {nm}\n{it.product_type.get(s, '')} | age {r.age_m:.0f}m | origin {r.origin} | occ {it.occasion.get(s, '') if pd.notna(it.occasion.get(s, np.nan)) else '-'}\n"
                 f"actual {r.y:,.0f} | 3-way {r.r_q_nn:,.0f} (rules {r.rules:,.0f} q {r.qmodel:,.0f} nn {r.nn:,.0f})", fontsize=8)
    ax.tick_params(labelsize=7)
axs[0].legend(fontsize=7, loc="upper left")
fig.suptitle(f"Largest {KIND}-forecasts (three-way, recent origins), ranks {SKIP + 1}-{SKIP + N}: weekly units; blue dashed = forecast weekly rate, black dotted = actual; teal = online discount", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(f"eda/figs/err_{KIND}{'_' + str(SKIP) if SKIP else ''}{'_' + ONLY if ONLY else ''}.png", dpi=55)
print("saved", KIND, "| share of all absolute error in these panels:", round(P.err.abs().sum() / T.err.abs().sum(), 3))
