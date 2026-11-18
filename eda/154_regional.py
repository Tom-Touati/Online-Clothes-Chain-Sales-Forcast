"""D152: regional structure of online demand (state of each online order; data/state_item_month.parquet from eda/152).
Regions: the four US Census regions, plus a climate split (warm = FL, TX, AZ, HI, LA, GA, AL, MS, SC, NV, PR and other territories;
cold = the Northeast and Midwest regions plus AK, MT, WY, ID, ND, SD, CO, UT, WA, OR, NH, VT, ME).
1. Share of units by region; seasonal timing by region for warm and summer product groups.
2. Are item trends national? At each origin, each item's 4-month ratio per region z_r = log((next 4 months + 1) / (4 x mean of last 3 + 1));
   correlation of z across regions, and the share of the regional variance explained by the national z.
3. Lead-lag: does a region's recent momentum (last 3 months vs the 3 before, minus the national one) predict the national z?
4. Time-honest features per item and origin (data <= origin): relative regional mix (item share minus site share, last 3 months),
   warm-state share, cold-state share, regional momentum dispersion, and the momentum of each region minus national.
Writes data/region_feats.parquet; diagnostics to stdout."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 200)
NE = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"]
MW = ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"]
SO = ["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia",
      "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"]
WE = ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"]
REG = {s: r for r, L in [("NE", NE), ("MW", MW), ("S", SO), ("W", WE)] for s in L}
WARM = {"Florida", "Texas", "Arizona", "Hawaii", "Louisiana", "Georgia", "Alabama", "Mississippi", "South Carolina", "Nevada", "Puerto Rico", "Guam", "Virgin Islands",
        "Northern Mariana Islands", "American Samoa", "Palau"}
COLD = set(NE) | set(MW) | {"Alaska", "Montana", "Wyoming", "Idaho", "Colorado", "Utah", "Washington", "Oregon"}
S = pd.read_parquet("data/state_item_month.parquet"); S["reg"] = S.state.map(REG).fillna("other")
S["clim"] = np.where(S.state.isin(WARM), "warm", np.where(S.state.isin(COLD), "cold", "mid"))
items = np.sort(pd.read_parquet("data/items.parquet").style_color.values)
F0 = pd.read_parquet("data/features.parquet", columns=["style_color", "macro", "ptype"]).drop_duplicates("style_color").set_index("style_color")
grp = np.array([str(x) if isinstance(x, str) else "na" for x in F0.macro.reindex(items).astype(object)])
print("== 1. share of online units"); print((S.groupby("reg").units.sum() / S.units.sum()).round(3).to_dict(), (S.groupby("clim").units.sum() / S.units.sum()).round(3).to_dict())
R = S.groupby(["m", "reg", "it"]).units.sum().unstack(["reg", "it"], fill_value=0)
months = list(R.index); RG = ["NE", "MW", "S", "W"]
print("\n== seasonal timing: each region's monthly units of a group / its own Jan-Dec 2025 mean (warm and summer-type groups)")
G = S.assign(g=grp[S.it.values]).groupby(["m", "reg", "g"]).units.sum().unstack(["reg"]).fillna(0)
for g in sorted(set(grp)):
    if g not in G.index.get_level_values(1): continue
    x = G.xs(g, level=1)[RG]; x = x / x.loc["2025-01":"2025-12"].mean(); print(f"-- {g}"); print(x.loc["2025-01":"2026-06"].T.round(2).to_string())
Pm = lambda o, k: str(pd.Period(o, "M") + k)
def win(o, ks, reg=None):
    cols = [Pm(o, k) for k in ks]; X = R.reindex(cols).fillna(0)
    return X[reg].values.sum(0) if reg else sum(X[r].values for r in RG).sum(0)
print("\n== 2. are item trends national?  (items with >= 30 units in every region in the last 3 months)")
rows, diag = [], []
for o in [str(p) for p in pd.period_range("2024-04", "2026-06", freq="M")]:
    nat3 = win(o, [-2, -1, 0]); natp = win(o, [-5, -4, -3]); natm = np.log((nat3 + 1) / (natp + 1))
    r3 = {r: win(o, [-2, -1, 0], r) for r in RG}; rp = {r: win(o, [-5, -4, -3], r) for r in RG}
    rmom = {r: np.log((r3[r] + 1) / (rp[r] + 1)) - natm for r in RG}
    tot = sum(r3.values()); site = {r: r3[r].sum() / tot.sum() for r in RG}
    f = pd.DataFrame({"origin": o, "style_color": items})
    for r in RG: f[f"reg_{r}_rel"] = np.where(tot > 0, r3[r] / np.maximum(tot, 1) - site[r], np.nan); f[f"reg_{r}_mom"] = np.where(tot >= 30, rmom[r], np.nan)
    cw = S[(S.m >= Pm(o, -2)) & (S.m <= o)].groupby(["it", "clim"]).units.sum().unstack(fill_value=0).reindex(range(len(items))).fillna(0)
    ct = cw.sum(1).replace(0, np.nan); sw = cw.sum(); f["clim_warm_rel"] = (cw.get("warm", 0) / ct - sw.get("warm", 0) / sw.sum()).values
    f["clim_cold_rel"] = (cw.get("cold", 0) / ct - sw.get("cold", 0) / sw.sum()).values
    f["reg_mom_sd"] = np.where(tot >= 30, np.std([rmom[r] for r in RG], axis=0), np.nan)
    rows.append(f)
    if Pm(o, 4) <= months[-1]:
        nxt = {r: win(o, [1, 2, 3, 4], r) for r in RG}; zr = {r: np.log((nxt[r] + 1) / (4 * r3[r] / 3 + 1)) for r in RG}
        zn = np.log((sum(nxt.values()) + 1) / (4 * tot / 3 + 1)); ok = np.all([r3[r] >= 30 for r in RG], axis=0)
        if ok.sum() >= 30:
            Z = np.array([zr[r][ok] for r in RG]); C = np.corrcoef(Z); off = C[np.triu_indices(4, 1)].mean()
            expl = 1 - np.mean([np.var(zr[r][ok] - zn[ok]) / np.var(zr[r][ok]) for r in RG])
            lead = {r: np.corrcoef(rmom[r][ok], zn[ok] - np.polyval(np.polyfit(natm[ok], zn[ok], 1), natm[ok]))[0, 1] for r in RG}
            diag.append(dict(origin=o, items=int(ok.sum()), corr_between_regions=off, var_explained_by_national=expl, **{f"lead_{r}": v for r, v in lead.items()}))
D = pd.DataFrame(diag); print(D.round(3).to_string(index=False)); print("mean:", D.drop(columns="origin").mean().round(3).to_dict())
pd.concat(rows, ignore_index=True).to_parquet("data/region_feats.parquet", index=False); print("saved data/region_feats.parquet")
