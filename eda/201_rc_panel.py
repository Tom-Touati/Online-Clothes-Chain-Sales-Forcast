"""D330 (fork 2; user: "the ideal model should be able to get future discounts and a location and predict the sales, then sum over
regions and channels"). Builds the item x location x month panel the location recommender is trained on.
Locations (from Dataset.parquet, cancelled lines dropped):
  web:<region>   edikted.com orders by US census region of the shipping state (Northeast, Midwest, South, West) and web:other
                 (no state, outside the US, or unrecognised)
  tiktok         TikTok shop (one location)
  store:<name>   each own store (o_type of RETAIL lines)
  partner:<name> each wholesale / marketplace partner (o_type of DROP lines, and Flip)
Per item x location x month: units, orders, and the units-weighted discount (avg_discounts).
Per location: kind, first and last month with any sale.
Writes data/rc_panel.parquet, data/rc_locations.parquet. Run on the CPU server (~/edikted_rc)."""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
REG = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD", "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX PR",
       "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
AB = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "washington dc": "DC", "florida": "FL", "georgia": "GA",
      "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
      "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
      "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
      "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR"}
S2R = {s: r for r, ss in REG.items() for s in ss.split()}
def region(state, country):
    s = str(state).strip()
    if str(country).strip() not in ("United States", "US", "USA", "nan", "None", ""): return "other"
    ab = s.upper() if len(s) == 2 else AB.get(s.lower())
    return S2R.get(ab, "other")
cols = ["ny_date", "style_color", "qty_ordered", "avg_discounts", "class", "o_type", "state", "country", "cancelled"]
pf = pq.ParquetFile("data/Dataset.parquet"); parts = []
for g in range(pf.metadata.num_row_groups):
    b = pf.read_row_group(g, columns=cols).to_pandas(); b = b[~b.cancelled.astype(bool)]
    reg = pd.Series([region(s, c) for s, c in zip(b.state, b.country)], index=b.index) if len(b) else pd.Series(dtype=str)
    loc = np.where(b.o_type == "edikted.com", "web:" + reg, np.where(b.o_type == "Tiktok", "tiktok", np.where(b["class"] == "RETAIL", "store:" + b.o_type.astype(str), "partner:" + b.o_type.astype(str))))
    b = b.assign(loc=loc, m=pd.to_datetime(b.ny_date).dt.strftime("%Y-%m"), dq=b.avg_discounts.astype(float).clip(-1, 1) * b.qty_ordered)
    parts.append(b.groupby(["style_color", "loc", "m"], observed=True).agg(units=("qty_ordered", "sum"), orders=("qty_ordered", "size"), dq=("dq", "sum")).reset_index())
    if g % 20 == 0: print("row group", g, "of", pf.metadata.num_row_groups, flush=True)
P = pd.concat(parts).groupby(["style_color", "loc", "m"], observed=True).sum().reset_index(); P["disc"] = P.dq / P.units.where(P.units > 0); P = P.drop(columns="dq")
P = P[P.units > 0]; P.to_parquet("data/rc_panel.parquet", index=False)
L = P.groupby("loc").agg(first_m=("m", "min"), last_m=("m", "max"), units=("units", "sum"), items=("style_color", "nunique")).reset_index(); L["kind"] = L["loc"].str.split(":").str[0]
L.to_parquet("data/rc_locations.parquet", index=False); print(L.sort_values("units", ascending=False).to_string(index=False)); print("panel", P.shape, P.m.min(), P.m.max())
