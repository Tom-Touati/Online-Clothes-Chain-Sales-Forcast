"""Seasonal-turn errors (D128): for warm and swim items at the origins where they swung hardest, compare the 4-month ratio to R3
(units in the window / R3) that each member implied with what was knowable (the like-for-like type ratio and the item's own ratio a year
earlier, from eda/06 features: sum over the 4 months of exp(lfl_type) and exp(own_sea)) and with what happened. Aggregates over items
(units-weighted, so ratio of sums), by origin and group."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
T = pd.read_parquet("data/fixes_items.parquet")
F = pd.read_parquet("data/features.parquet", columns=["origin", "style_color", "h", "m3", "lfl_type", "own_sea"])
F["st"] = F.m3 * np.exp(F.lfl_type.fillna(0)); F["so"] = np.where(F.own_sea.notna(), F.m3 * np.exp(F.own_sea), np.nan)
G = F.groupby(["origin", "style_color"]).agg(sea_type=("st", "sum"), sea_own=("so", lambda x: x.sum() if x.notna().all() else np.nan)).reset_index()
T = T.merge(G, on=["origin", "style_color"], how="left")
rows = []
for (o, g), x in T.groupby(["origin", "macro"]):
    if g not in ("warm", "swim", "dress", "bottom", "top"): continue
    r3 = x.R3.sum(); xo = x[x.sea_own.notna()]
    rows.append({"origin": o, "group": g, "items": len(x), "actual/R3": x.y.sum() / r3, "rules": x.rules.sum() / r3, "quantile": x.qmodel.sum() / r3, "nn": x.nn.sum() / r3,
                 "3-way": x.r_q_nn.sum() / r3, "type LY ratio": x.sea_type.sum() / r3,
                 "own LY (items with own)": xo.sea_own.sum() / xo.R3.sum() if len(xo) else np.nan, "actual (same items)": xo.y.sum() / xo.R3.sum() if len(xo) else np.nan})
R = pd.DataFrame(rows)
for g in ["warm", "swim", "dress", "bottom", "top"]:
    print(f"\n== {g}: 4-month units / R3"); print(R[R.group == g].drop(columns="group").set_index("origin").round(2).to_string())
