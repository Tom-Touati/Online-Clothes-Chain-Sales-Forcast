"""D206 F4b (fork): stack-aware feature pruning. As eda/193c 'select' (backward group elimination, then single features; four EARLY
origins only), but the criterion is the units WAPE of the whole stack, the geometric mean of rules, the candidate tree median, NN
and TabPFN v2 (dyn, 32 estimators) (data/fk_members_for_select.parquet), so features are kept for what they add next to TabPFN.
Writes data/fk_feat_select_stack.json (BASE, FULL for eda/193b 'sels')."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
JOBS = int(dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a).get("jobs", 8))
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
for c in META["CAT"]: T[c] = T[c].astype("category")
BASE, FULL = META["BASE"] + DY, META["FULL"] + DY; P0 = dict(META["PARAMS"]); P0["n_jobs"] = JOBS
MB = pd.read_parquet("data/fk_members_for_select.parquet").set_index(["origin", "style_color"])
lg = lambda x: np.log(np.maximum(x, 0) + 0.5)
SPL = []
for o in ["2025-02", "2025-03", "2025-04", "2025-05"]:
    oo = pd.Period(o).ordinal; te = T[T.origin == o]; mb = MB.reindex(pd.MultiIndex.from_arrays([te.origin, te.style_color]))
    SPL.append((T[(T.o_ord + 4 <= oo) & T.ytot.notna()], te, (lg(mb.rules.values) + lg(mb.nn.values) + lg(mb.pf2d32.values)), mb.y.values))
def crit(cols):
    num = den = 0.0
    for tr, te, other, y in SPL:
        z = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P0).fit(tr[cols], tr.z).predict(te[cols]); f = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)
        s = np.exp((other + lg(f)) / 4) - 0.5; num += np.abs(s - y).sum(); den += y.sum()
    return num / den
t0 = time.time(); ref = crit(FULL); print("stack early WAPE with the full D198 trees", round(ref, 4), flush=True)
G = META["GROUP"]; grp = {}
for f in FULL:
    g = G.get(f, "dynamic co-purchase neighbours" if f in DY else "other")
    if g == "special events / occasions": g = "events: upcoming" if f.endswith("_t") or f in ("occ_next", "occ_n4") else ("events: past" if f.endswith("_o") or f == "occ_last" else "events: interactions")
    grp.setdefault(g, []).append(f)
keep, cur, log = dict(grp), ref, []
while len(keep) > 1:
    res = {g: crit([f for gg, fs in keep.items() if gg != g for f in fs]) for g in keep}
    g, w = min(res.items(), key=lambda kv: kv[1]); print("round:", {k: round(v, 4) for k, v in res.items()}, flush=True)
    if w < cur - 0.001: keep.pop(g); log.append(("group", g, round(cur, 4), round(w, 4))); cur = w
    else: break
feats = [f for fs in keep.values() for f in fs]
for f in list(feats):
    w = crit([x for x in feats if x != f])
    if w < cur - 0.001: feats.remove(f); log.append(("feature", f, round(cur, 4), round(w, 4))); cur = w
FULLs = [f for f in FULL if f in feats]; BASEs = [f for f in BASE if f in feats]
print("dropped:", log); print("stack early WAPE", round(ref, 4), "->", round(cur, 4), "| FULL", len(FULL), "->", len(FULLs), "| BASE", len(BASE), "->", len(BASEs), f"| {time.time() - t0:.0f}s")
json.dump({"BASE": BASEs, "FULL": FULLs, "dropped": log, "early_wape": cur, "ref_early_wape": ref}, open("data/fk_feat_select_stack.json", "w"), indent=1)
