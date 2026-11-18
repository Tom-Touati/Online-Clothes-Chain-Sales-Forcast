"""Generalised copy of eda/49_rules_in_sim.py: pass the origin month (default 2025-06). Business test of the final-forecast rules (open points: mature 50/50 YoY blend, Homecoming/Halloween floors, July/June split).
Re-creates the deliverable's situation one year earlier: one forecast made at the end of June 2025 for Jul-Oct 2025, built
with each rule as it would have been built then (2024 history only, no 2025 actuals after June), and used for every buying
decision from Jul 1 to Sep 30 in the simulation of eda/47_decision_sim.py. Each rule is kept only if it raises profit.
Leakage note: the ageing factor 0.843 was fitted on origins that include Jun 2025 (sensitivity with 1.0 reported)."""
import importlib.util, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("sim", "eda/47_decision_sim.py"); sim = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(sim); sys.argv = _a
M = pd.read_parquet("data/monthly_item.parquet"); it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
occ = it.occasion.fillna(""); SW = ["Swimwear Top", "Swimwear Bottom"]
mo = lambda y, ms: [f"{y}-{m:02d}" for m in ms]
def seg(idx):
    s = pd.Series("other", index=idx)
    s[occ.reindex(idx).str.contains("Homecoming").values] = "Homecoming"; s[occ.reindex(idx).str.contains("Halloween").values] = "Halloween"
    s[it.product_type.reindex(idx).isin(SW).values] = "swimwear"
    return s

O = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("20") else "2025-06"
PO = pd.Period(O); MS = [str(PO + h) for h in range(1, 5)]; LY = [str(PO + h - 12) for h in range(1, 5)]
w = sim.Window(str((PO + 1).start_time.date())); I = pd.Index(w.items)
b = sim.BT[sim.BT.origin == O].pivot(index="style_color", columns="h", values="pred").reindex(I)
q = sim.QB.loc[O].reindex(I)
base = b.values * (q.q50.values / b.sum(axis=1).values)[:, None]          # model P50 split, Jul..Oct 2025
act = M.loc[I, MS].values
jun = M.loc[I, O].values
sg = seg(I).values
mature = ((start.reindex(I) < (PO - 14).start_time).values & (M.loc[I, LY].sum(axis=1).values >= 50))
# segment history from 2024 only (what was known at the June 2025 origin)
OLY = str(PO - 12)
live = start[start < (PO - 12).start_time - pd.Timedelta(days=15)].index.intersection(M.index); live = live[M.loc[live, OLY] >= 50]
r24 = M.loc[live, LY].sum(axis=1) / M.loc[live, OLY]; s24 = seg(live).values
X24 = M.loc[live, LY]
SHAPE = {s: (X24[s24 == s].sum() / X24[s24 == s].sum().sum()).values for s in ["Homecoming", "Halloween", "swimwear"]}
FLOOR = {s: float(r24[s24 == s].quantile(0.30)) for s in ["Homecoming", "Halloween"]}
CAP = float(np.median(r24[s24 == "swimwear"]))
print("floors (2024 p30):", {k: round(v, 2) for k, v in FLOOR.items()}, "| swim cap (2024 median):", round(CAP, 2), "| mature items:", int(mature.sum()), "of", len(I))


AGE0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.843
def blend(F, age=None):
    age = AGE0 if age is None else age
    F = F.copy(); ly = M.loc[I, LY].values
    yn = ly.sum(1) * (M.loc[I, [str(PO - j) for j in (2, 1, 0)]].sum(1).values + 5) / (M.loc[I, [str(PO - 12 - j) for j in (2, 1, 0)]].sum(1).values + 5) * age
    tot = F.sum(1); new = np.exp(0.5 * np.log(tot + 1) + 0.5 * np.log(yn + 1)) - 1
    shp = 0.5 * F / np.maximum(tot, 1e-9)[:, None] + 0.5 * ly / np.maximum(ly.sum(1), 1e-9)[:, None]
    F[mature] = (new[:, None] * shp)[mature]; return F

def floors(F):
    F = F.copy()
    for s, r in FLOOR.items():
        m = (sg == s) & ~mature & (F.sum(1) < jun * r)
        F[m] = (jun[m] * r)[:, None] * SHAPE[s][None, :]
    return F

def swimcap(F):
    F = F.copy(); m = (sg == "swimwear") & (F.sum(1) > jun * CAP)
    F[m] = F[m] * (jun[m] * CAP / F[m].sum(1))[:, None]; return F

def july_to_benchmark(F):
    """Raise July so the aggregate July/June equals the 2024 same-mix benchmark (age groups weighted by June 2025 units); totals kept."""
    F = F.copy(); agep = (PO.end_time.normalize() - start.reindex(I)).dt.days.values / 30.44
    grp = np.where(agep >= 12, "12m+", np.where(agep >= 3, "3-12", "<3"))
    a24 = ((PO - 12).end_time.normalize() - start.reindex(live)).dt.days.values / 30.44
    g24 = np.where(a24 >= 12, "12m+", np.where(a24 >= 3, "3-12", "<3"))
    rj = {g: M.loc[live[g24 == g], str(PO - 11)].sum() / M.loc[live[g24 == g], OLY].sum() for g in np.unique(g24)}
    bench = sum(rj.get(g, np.nan) * jun[grp == g].sum() for g in rj) / jun[np.isin(grp, list(rj))].sum()
    cur = F[:, 0].sum() / jun.sum(); f = bench / cur
    tot = F.sum(1); F[:, 0] *= f; rest = F[:, 1:].sum(1); F[:, 1:] *= ((tot - F[:, 0]) / np.maximum(rest, 1e-9))[:, None]
    print(f"July/June: forecast {cur:.3f}, 2024 same-mix benchmark {bench:.3f}, actual 2025 {act[:, 0].sum() / jun.sum():.3f}")
    return F


def run(F, alpha, mask=None):
    dmon = w.days.to_period("M"); D = np.zeros((len(I), w.T))
    for t, m in enumerate(dmon):
        h = (m - pd.Period(O)).n
        if 1 <= h <= 4: D[:, t] = F[:, h - 1] / m.days_in_month
    for o in w.origins: w.F[o] = D; w.SIG[o] = sim.K_IQR * np.log((q.q75.values + 1) / (q.q25.values + 1)) / 1.349
    r = sim.simulate(w, "model", tau=0.8, alpha=alpha)
    # per-item profit so each rule is judged only on the items it changes
    buy = np.bincount(r["orders"].i.astype(int), weights=r["orders"].q, minlength=len(I)) + w.pre.sum(1)
    prof = r["sales"].sum(1) * sim.P - buy * sim.C - r["inv"].sum(1) * sim.H + r["inv"][:, -1] * sim.S_SALV
    return prof, r["sales"].sum(1)

ideal = sim.simulate(w, "ideal"); ip = ideal["sales"].sum(1) * sim.P - (np.bincount(ideal["orders"].i.astype(int), weights=ideal["orders"].q, minlength=len(I)) + w.pre.sum(1)) * sim.C - ideal["inv"].sum(1) * sim.H

V = {"model P50": base}
V["+ mature blend"] = blend(base); V["+ mature blend (ageing 1.0)"] = blend(base, 1.0)
V["+ floors"] = floors(base); V["+ swim cap"] = swimcap(base)
V["all rules"] = swimcap(floors(blend(base)))
V["all rules + July at benchmark"] = july_to_benchmark(V["all rules"])
groups = {"mature": mature, "floored (Hc/Hw young)": np.isin(sg, ["Homecoming", "Halloween"]) & ~mature, "swimwear": sg == "swimwear", "all": np.ones(len(I), bool)}
rows = []
for alpha in (0.0, 0.5):
    P = {k: run(F, alpha)[0] for k, F in V.items()}
    for k, F in V.items():
        for gname, m in groups.items():
            tot = F.sum(1)[m]; a = act[m].sum(1)
            rows.append(dict(alpha=alpha, variant=k, items=gname, n=int(m.sum()), loss_pct=(ip[m].sum() - P[k][m].sum()) / ip[m].sum(),
                             bias=tot.sum() / a.sum() - 1, wape=np.abs(tot - a).sum() / a.sum()))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for gname, keep in [("mature", ["model P50", "+ mature blend", "+ mature blend (ageing 1.0)"]), ("floored (Hc/Hw young)", ["model P50", "+ floors"]),
                    ("swimwear", ["model P50", "+ swim cap"]), ("all", list(V))]:
    x = R[(R["items"] == gname) & R.variant.isin(keep)].pivot_table(index=["variant", "n"], columns="alpha", values=["loss_pct", "bias", "wape"]).round(3)
    print(f"\n== items: {gname}  (loss = ideal profit - profit, % of ideal; bias/WAPE on the Jul-Oct total)"); print(x.to_string())
# July split, monthly check
print("\nmonthly totals, all items: actual", act.sum(0).astype(int).tolist())
for k in ["model P50", "all rules", "all rules + July at benchmark"]: print(k, V[k].sum(0).astype(int).tolist())
R.to_csv(f"data/sim_rules_test_{O}.csv", index=False)
