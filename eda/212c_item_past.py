"""Fork 2 (user: "to the explorer please add the past predictions as well"): the submitted system's backtest forecasts for each explorer item.
From the new nx5 backtest (data/dec_base_nx5r26.parquet + geo_nx5r_x_preds.parquet + ranges_nx5r_loo.parquet; was fk_err_master; 13 origins, 2025-02 .. 2026-02; origin o = data through month o, forecast = units in o+1 .. o+4):
the system F (five-member mean x submitted correction multiplier), its P10 / P90, R3 (recent rate x 4), the rule agents and the four other
members, and the actual units y. Items without a backtest row (launched too late, or below the scoreboard's sales floor) get none.
Adds 'past' and 'past_stats' to data/item_explorer.json, then writes eda/item_explorer.html from eda/item_explorer_template.html."""
import json, numpy as np, pandas as pd
J = json.load(open("data/item_explorer.json", encoding="utf-8"))
D = pd.read_parquet("data/dec_base_nx5r26.parquet"); D = D[D.y.notna()].merge(pd.read_parquet("data/geo_nx5r_x_preds.parquet")[["origin", "style_color", "geo"]], on=["origin", "style_color"])   # D389i: the new nx5's backtest
D = D.merge(pd.read_parquet("data/ranges_nx5r_loo.parquet"), on=["origin", "style_color"], how="left").rename(columns={"geo": "F_new"}); D["F"] = D.F_new   # F = corrected system; p10 / p90 = its leave-one-out ranges
r0 = lambda v: None if v is None or not np.isfinite(v) else int(round(float(v)))
n_with = 0
for it in J["items"]:
    d = D[D.style_color == it["id"]].sort_values("origin")
    it["past"] = [dict(o=o, y=r0(y), F=r0(F), R3=r0(R3), rules=r0(ru), p10=r0(a), p90=r0(b), qnx=r0(q), nn=r0(n), pf2c=r0(c), pf2d32=r0(e), rc=r0(rc_))
                  for o, y, F, R3, ru, a, b, q, n, c, e, rc_ in d[["origin", "y", "F", "R3", "rules", "p10", "p90", "qdyn", "nn", "pf2c", "pf2d32", "rc"]].itertuples(index=False)]
    if len(d):
        n_with += 1; y, F, R3 = d.y.values, d.F.values, d.R3.values
        den = np.abs(R3 - y).sum()
        it["past_stats"] = dict(n=int(len(d)), wape_r3=(round(float(np.abs(F - y).sum() / den), 3) if den > 0 else None),
                                wape=(round(float(np.abs(F - y).sum() / y.sum()), 3) if y.sum() > 0 else None),
                                bias=(round(float(F.sum() / y.sum() - 1), 3) if y.sum() > 0 else None),
                                inside=round(float(((y >= d.p10.values) & (y <= d.p90.values)).mean()), 3),
                                w20=round(float((np.abs(F - y) <= 0.2 * y).mean()), 3))
    else:
        it["past_stats"] = None
json.dump(J, open("data/item_explorer.json", "w", encoding="utf-8"), ensure_ascii=False)
html = open("eda/item_explorer_template.html", encoding="utf-8").read()
assert "/*DATA*/null" in html
open("eda/item_explorer.html", "w", encoding="utf-8").write(html.replace("/*DATA*/null", json.dumps(J, ensure_ascii=False).replace("</", "<\\/")))
print("items with a backtest:", n_with, "of", len(J["items"]))
S = pd.DataFrame([dict(id=it["id"], **it["past_stats"]) for it in J["items"] if it["past_stats"]])
print(S.describe().round(3).to_string())
