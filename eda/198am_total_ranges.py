"""D371 (fork; attempts 2 and 3 of Fork 2's delegation, user in Fork 2's session: "delegate these attempts"): ranges for totals, at the level
where they can be met, and a shared origin-level term so item ranges are not summed as if independent.
  levels    product group (macro), product type, look (groups of 3+ items), the total of all items; the candidate system's P50 (D358 / nx5)
  item rng  the adopted per-segment item ranges (D366: pf2c widths, factors left one origin out; from data/fk_width_model_backtest.parquet)
  methods   T1 sum of item ranges          group P25 = sum of item P25 (as if every item missed in the same direction)
            T2 independent items           units spread from each item's half-ranges (normal approximation), added in quadrature
            T3 T2 + shared origin term     log spread^2 = (T2 spread / F)^2 + c^2, where c = the root mean square of the all-item total's log miss
                                           on the training origins (the part every item shares at an origin)
            T4 studentised conformal       r = log miss / T3 spread; bounds log(F+1) + quantiles of r on the training groups (all levels pooled),
                                           so the shift and the asymmetry come from the data
            T5 level x size-third quantiles (the D320 method): the log miss's quantiles by level and size third of the forecast total
  protocols loo / purged (4+ months away) / honest (closed origins only, scored 2025-07..2026-02), as D370
Scores per level: P25-P75 and P10-P90 below / inside / above, mean log width, pinball in units / R3's, and the share within +/-20% (the
promise level of D320). Also the correlation of item misses within an origin (why summing as independent is too narrow).
If a method holds coverage at every level, writes its June 2026 total ranges (data/fk_total_ranges_2026.csv). Adds J["total_ranges"];
stdout -> eda/total_ranges_output.txt."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "eda"); import fk_rangekit as K
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
B = K.load(); ORIG = sorted(B.origin.unique())
WB = pd.read_parquet("data/fk_width_model_backtest.parquet"); assert (WB.style_color.values == B.style_color.values).all() and (WB.origin.values == B.origin.values).all()
AD = "loo|pf2c widths, per segment (adopted D366)|"
for rng, a, b in [("P25-P75", "L50", "H50"), ("P10-P90", "L80", "H80")]: B[a] = WB[AD + rng + "|lo"].values; B[b] = WB[AD + rng + "|hi"].values
assert B[["L50", "H50", "L80", "H80"]].notna().all().all()
Z50, Z80 = 0.6745, 1.2816
B["v50l"] = ((B.P50 - B.L50) / Z50) ** 2; B["v50h"] = ((B.H50 - B.P50) / Z50) ** 2; B["v80l"] = ((B.P50 - B.L80) / Z80) ** 2; B["v80h"] = ((B.H80 - B.P50) / Z80) ** 2
B["all"] = "all items"
LV = {"product group": "macro", "product type": "ptype", "look": "look", "total": "all"}; parts = []
for lab, key in LV.items():
    g = B.groupby(["origin", key]).agg(F=("P50", "sum"), y=("y", "sum"), R3=("R3", "sum"), n=("y", "size"), L50=("L50", "sum"), H50=("H50", "sum"), L80=("L80", "sum"), H80=("H80", "sum"),
                                      v50l=("v50l", "sum"), v50h=("v50h", "sum"), v80l=("v80l", "sum"), v80h=("v80h", "sum")).reset_index().rename(columns={key: "group"})
    g = g[g.n >= 3]; g["level"] = lab; g["third"] = g.groupby("origin").F.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else 0 * s); parts.append(g)
G = pd.concat(parts, ignore_index=True); G["oo"] = pd.PeriodIndex(G.origin, freq="M").asi8; G["lF"] = np.log(G.F + 1); G["ly"] = np.log(G.y + 1); G["e"] = G.ly - G.lF; G["seg"] = G.level
print("groups per origin:", G.groupby("level").size().div(13).round(1).to_dict())
TOT = G[G.level == "total"].set_index("origin").e
print("all-item total, log miss by origin:", TOT.round(3).to_dict())
# within-origin correlation of item misses: the share of the item log-miss variance explained by the origin's mean miss
ib = B.assign(em=B.groupby("origin").e.transform("mean")); print("item log miss: variance", round(float(B.e.var()), 3), "| between-origin share", round(float(ib.em.var() / B.e.var()), 3),
                                                            "| origin mean miss", ib.groupby("origin").e.mean().round(3).to_dict())
s_ind = {w: (np.sqrt(G[f"v{w}l"]) / (G.F + 1)).values for w in ["50", "80"]}; s_ind_h = {w: (np.sqrt(G[f"v{w}h"]) / (G.F + 1)).values for w in ["50", "80"]}


def tmask(o, how):
    d = G.oo.values - pd.Period(o).ordinal
    return {"loo": d != 0, "purged": np.abs(d) >= 4, "honest": d <= -4}[how]


def bounds(how, meth, rng):
    w, z, tail = ("50", Z50, 0.25) if rng == "P25-P75" else ("80", Z80, 0.10)
    lo, hi = np.full(len(G), np.nan), np.full(len(G), np.nan); cs = {}
    for o in ORIG:
        tr = tmask(o, how); te = (G.origin == o).values; tro = G.origin[tr & (G.level == "total").values]
        if tro.nunique() < 2: continue
        c = float(np.sqrt(np.mean(TOT.reindex(tro).values ** 2))); cs[o] = round(c, 3)
        if meth == "T1": lo[te] = np.log(G[f"L{w}"].values[te] + 1); hi[te] = np.log(G[f"H{w}"].values[te] + 1)
        elif meth == "T2": lo[te] = np.log(np.maximum(G.F.values[te] * (1 - z * s_ind[w][te]), 0) + 1); hi[te] = np.log(G.F.values[te] * (1 + z * s_ind_h[w][te]) + 1)
        elif meth == "T3": lo[te] = G.lF.values[te] - z * np.sqrt(s_ind[w][te] ** 2 + c ** 2); hi[te] = G.lF.values[te] + z * np.sqrt(s_ind_h[w][te] ** 2 + c ** 2)
        elif meth == "T4":
            s = np.sqrt(((s_ind["50"] + s_ind_h["50"]) / 2) ** 2 + c ** 2); r = G.e.values / s
            lo[te] = G.lF.values[te] + np.quantile(r[tr], tail) * s[te]; hi[te] = G.lF.values[te] + np.quantile(r[tr], 1 - tail) * s[te]
        elif meth == "T5":
            for (lv, t), mm in G[te].groupby(["level", "third"]).groups.items():
                ref = tr & (G.level == lv).values & (G.third == t).values; ix = np.asarray(G.index.get_indexer(mm))
                if ref.sum() < 5: ref = tr & (G.level == lv).values
                lo[ix] = G.lF.values[ix] + np.quantile(G.e.values[ref], tail); hi[ix] = G.lF.values[ix] + np.quantile(G.e.values[ref], 1 - tail)
    return lo, hi, cs


def sc(lo, hi, a, m):
    ly, y, R3 = G.ly.values[m], G.y.values[m], G.R3.values[m]; lo, hi = lo[m], hi[m]; below, above = ly < lo, ly > hi; t = a / 2
    pr = (K.pin(y, np.exp(lo) - 1, t).sum() + K.pin(y, np.exp(hi) - 1, 1 - t).sum()) / (K.pin(y, R3, t).sum() + K.pin(y, R3, 1 - t).sum())
    return dict(n=int(m.sum()), below=round(float(below.mean()), 3), inside=round(float((~below & ~above).mean()), 3), above=round(float(above.mean()), 3), log_width=round(float((hi - lo).mean()), 3), pinball_r3=round(float(pr), 3))


MN = {"T1": "sum of item ranges", "T2": "independent items", "T3": "independent + shared origin term", "T4": "studentised conformal", "T5": "level x size-third quantiles (D320)"}
P = {"recent 9": (G.origin >= "2025-06").values, "all 13": np.ones(len(G), bool), "2025-07..2026-02": (G.origin >= "2025-07").values}
rows, CS, BND = [], {}, {}
for how in ["loo", "purged", "honest"]:
    for meth in MN:
        for rng, a in [("P25-P75", 0.5), ("P10-P90", 0.2)]:
            lo, hi, cs = bounds(how, meth, rng); CS[how] = cs; BND[(how, meth, rng)] = (lo, hi)
            for pn, pm in P.items():
                if (how == "honest") != (pn == "2025-07..2026-02"): continue
                for lv in LV:
                    m = pm & (G.level == lv).values & np.isfinite(lo)
                    rows.append(dict(protocol=how, method=MN[meth], range=rng, period=pn, level=lv, **sc(lo, hi, a, m)))
R = pd.DataFrame(rows)
hit = G.assign(h=(G.F - G.y).abs() <= 0.2 * G.y).groupby("level").h.agg(lambda s: round(float(s.mean()), 3)); hit_r = G[P["recent 9"]].assign(h=lambda d: (d.F - d.y).abs() <= 0.2 * d.y).groupby("level").h.mean().round(3)
print("within +/-20% (all 13):", hit.to_dict(), "| recent 9:", hit_r.to_dict())
print("shared origin term c by test origin:", CS)
for rng in ["P25-P75", "P10-P90"]:
    for pn in P:
        V = R[(R.range == rng) & (R.period == pn)]
        print(f"\n=== {rng}, {pn} ==="); print(V.pivot_table(index=["protocol", "method"], columns="level", values="inside", sort=False).round(3).to_string())
        print(V.pivot_table(index=["protocol", "method"], columns="level", values="log_width", sort=False).round(3).add_prefix("width ").to_string())
print("\nfull table (recent 9 and honest):"); print(R[R.period != "all 13"].to_string(index=False))
J = json.load(open("data/fk_err_analysis.json")); J["total_ranges"] = dict(rows=rows, c_by_origin=CS, hit20=hit.to_dict(), hit20_recent=hit_r.to_dict(), total_miss=TOT.round(4).to_dict())
json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0, default=float)
G.to_parquet("data/fk_total_ranges_backtest.parquet"); print("saved")
