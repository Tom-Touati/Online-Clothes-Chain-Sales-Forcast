"""D239: where do demand variation and forecast error live in the hierarchy? (data/dec_base.parquet, D238; final system F = p2c32a.)
1. Oracle rescaling: F scaled so that each group's total equals its actual total, at the level of site, macro group, product type,
   look cluster, type x look, style; raw WAPE and WAPE / R3 after each (nine recent origins). The drop from one level to the next is the
   error that lives between those levels.
2. Nested variance decomposition (weights = sqrt(y + F)), for the log growth g = log((y + 1) / (R3 + 1)) and the log error
   e = log((y + 1) / (F + 1)): share of the total weighted variance between groups at each successive level, along two paths:
   A site > macro > type > style > colour, and B site > look > style > colour. Recent origins, averaged.
3. The same split by origin month (when) and by the Black Friday window.
Writes eda/dec_levels_output.txt (stdout) and data/dec_levels.csv."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
B = pd.read_parquet("data/dec_base.parquet"); B["site"] = "all"; B["type_look"] = B.ptype + "|" + B.look
REC = B[B.origin >= "2025-06"].copy()
def oracle(D, lev):
    if lev is None: Fh = D.F
    else: Fh = D.F * (D.groupby(["origin", lev]).y.transform("sum") / D.groupby(["origin", lev]).F.transform("sum").clip(lower=1e-9))
    return (Fh - D.y).abs().sum()
den = (REC.R3 - REC.y).abs().sum(); ysum = REC.y.sum(); rows = []
print("== 1. oracle rescaling of the final system (recent origins): error left if each level's totals were known")
for lev in [None, "site", "macro", "ptype", "look", "type_look", "style"]:
    a = oracle(REC, lev); rows.append({"analysis": "oracle", "level": lev or "as forecast", "wape": a / ysum, "wape_r3": a / den})
    print(f"  {str(lev or 'as forecast'):12s} WAPE {a / ysum:.4f} | WAPE / R3 {a / den:.4f}")
def nested(D, path, col):
    w = np.sqrt(D.y + D.F + 1); out = {}; prev = None
    tot = np.average((D[col] - np.average(D[col], weights=w)) ** 2, weights=w)
    for k in range(1, len(path) + 1):
        grp = D.groupby(path[:k] if path[:k] != ["site"] else ["site"])
        mk = grp[col].transform(lambda v: np.average(v, weights=w.loc[v.index]))
        between = np.average((mk - np.average(D[col], weights=w)) ** 2, weights=w)
        out[path[k - 1]] = between / tot - (prev or 0); prev = between / tot
    out["colour (item)"] = 1 - prev; return out
print("\n== 2. nested variance shares (weights sqrt(y + F)), mean over recent origins")
for col, nm in [("g", "growth vs R3"), ("e", "final-system error")]:
    for pname, path in [("A", ["macro", "ptype", "style"]), ("B", ["look", "style"])]:
        res = pd.DataFrame([nested(g, path, col) for _, g in REC.groupby("origin")]).mean()
        print(f"  {nm:20s} path {pname}: " + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))
        for k, v in res.items(): rows.append({"analysis": f"nested {pname} {col}", "level": k, "share": v})
print("\n== 3. when: error variance shares by origin (path B) and WAPE / R3")
for o, g in B.groupby("origin"):
    r = nested(g, ["look", "style"], "e"); w = (g.F - g.y).abs().sum() / (g.R3 - g.y).abs().sum()
    print(f"  {o}  WAPE/R3 {w:.3f} | look {r['look']:.3f} style {r['style']:.3f} colour {r['colour (item)']:.3f}")
    rows.append({"analysis": "by origin", "level": o, "wape_r3": w, "look": r["look"], "style": r["style"], "colour": r["colour (item)"]})
pd.DataFrame(rows).to_csv("data/dec_levels.csv", index=False)
