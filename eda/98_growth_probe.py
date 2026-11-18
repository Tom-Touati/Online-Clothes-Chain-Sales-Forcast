"""Catalogue growth probe (D116): how does the rising number of items show up in the errors?
For each origin: items live at the origin (the forecast set), items launched in the next 4 months, the launches' share of all units
sold in the target window, the existing items' share of site units before and after, and the aggregate bias of each system on the
existing items. If new launches cannibalise existing items more than in the past, bias should fall as the launch share rises.
Also: site units growth vs existing-item growth (like-for-like), by origin."""
import numpy as np, pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250)
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
X = pd.read_parquet("data/fixes_items.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
first = M.gt(0).idxmax(axis=1).where(M.gt(0).any(axis=1))        # first month with sales
rows = []
for o, x in X.groupby("origin"):
    P = pd.Period(o, "M"); win = [str(P + h) for h in range(1, 5)]; pre = [str(P - k) for k in range(0, 3)]
    site_win = M[win].sum().sum(); site_pre = M[pre].sum().sum() * 4 / 3
    launched = first[(first > str(P)) & (first <= win[-1])].index
    live_pre = M.index[M[pre].sum(axis=1) > 0]
    r = {"origin": o, "forecast items": len(x), "live items at origin": len(live_pre), "launches next 4m": len(launched),
         "launch share of window units": M.loc[launched, win].sum().sum() / site_win,
         "site growth (window / 4/3 x last 3m)": site_win / site_pre,
         "forecast-set growth (actual / R3)": x.y.sum() / x.R3.sum()}
    for c in ["rules", "qmodel", "nn", "r_q_nn"]: r[f"bias {c}"] = x[c].sum() / x.y.sum() - 1
    rows.append(r)
R = pd.DataFrame(rows).set_index("origin")
print(R.round(3).to_string())
print("\ncorrelation across origins of the launch share with each system's bias on existing items:")
print({c: round(float(np.corrcoef(R["launch share of window units"], R[f"bias {c}"])[0, 1]), 2) for c in ["rules", "qmodel", "nn", "r_q_nn"]})
print("correlation of launch share with existing-item growth (actual / R3):", round(float(np.corrcoef(R["launch share of window units"], R["forecast-set growth (actual / R3)"])[0, 1]), 2))
