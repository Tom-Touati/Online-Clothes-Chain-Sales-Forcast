"""Collects every number shown on the project-map page into data/project_map.json (no hand-typed figures for charts and tables)."""
import json, re, glob, os, openpyxl, pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
J = {}
d = pd.read_parquet("data/daily_item_chan.parquet"); d["ny_date"] = pd.to_datetime(d.ny_date)
ok = d[~d.cancelled].copy(); ok["m"] = ok.ny_date.dt.to_period("M").astype(str)
it = pd.read_parquet("data/items.parquet").set_index("style_color")
it["pub"] = pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
M = pd.read_parquet("data/monthly_item.parquet")
T = pd.read_csv("data/targets.csv").style_color.tolist()

# 1 monthly by class + cancellation
mc = ok.groupby(["m", "class"]).qty.sum().unstack(fill_value=0)
canc = d.assign(m=d.ny_date.dt.to_period("M").astype(str)).groupby(["m", "cancelled"]).qty.sum().unstack(fill_value=0)
J["monthly"] = [dict(m=m, online=int(r.ONLINE), retail=int(r.RETAIL), drop=int(r.DROP), cancel_rate=round(float(canc.loc[m, True] / canc.loc[m].sum()), 4),
                     items_live=int(((M[m] > 0)).sum())) for m, r in mc.iterrows()]
# 2 channels by month (k)
ch = ok.groupby(["m", "chan"]).qty.sum().unstack(fill_value=0)
keep = ["edikted.com", "Tiktok", "RETAIL", "Pacsun", "Nordstrom", "Macys"]
ch["other DROP"] = ch.drop(columns=keep).sum(axis=1)
J["channels"] = {"months": list(ch.index), "series": {c: [int(v) for v in ch[c]] for c in keep + ["other DROP"]}}
st = pd.read_parquet("data/daily_otype.parquet"); st["ny_date"] = pd.to_datetime(st.ny_date); st = st[~st.cancelled]
so = st[st["class"] == "RETAIL"].groupby("otype").agg(first=("ny_date", "min"), last=("ny_date", "max"), qty=("qty", "sum")).sort_values("first")
J["stores"] = [dict(code=k, first=str(r["first"].date()), last=str(r["last"].date()), units=int(r.qty)) for k, r in so.iterrows()]
# 3 weekly units + edikted.com discount + like-for-like units
wk = ok.groupby(pd.Grouper(key="ny_date", freq="W-SUN")).qty.sum()
web = ok[ok.chan == "edikted.com"].groupby(pd.Grouper(key="ny_date", freq="W-SUN")).agg(q=("qty", "sum"), dw=("disc_w", "sum"))
J["weekly"] = [dict(w=str(k.date()), units=int(v), disc=round(float(web.dw[k] / web.q[k]), 4)) for k, v in wk.items()]
dd = ok.groupby("ny_date").qty.sum()
J["top_days"] = [dict(d=str(k.date()), units=int(v)) for k, v in dd.nlargest(6).items()]
dow = dd.groupby(dd.index.dayofweek).mean(); J["dow"] = [round(float(x), 3) for x in dow / dow.mean()]
# 4 type seasonality 2025 (index = month / type's monthly mean)
ty = M.groupby(it.reindex(M.index).product_type.values).sum()
big = ty.sum(axis=1).sort_values(ascending=False).head(22).index
cols25 = [f"2025-{m:02d}" for m in range(1, 13)]
prof = ty.loc[big, cols25]; prof = prof.div(prof.mean(axis=1), axis=0)
J["type_season"] = [dict(type=t, units=int(ty.loc[t].sum()), idx=[round(float(x), 2) for x in prof.loc[t]], n_items=int((it.product_type == t).sum()), n_targets=int(it.loc[T].product_type.eq(t).sum())) for t in big]
# 5 occasion index 2025
occ = it.occasion.fillna("")
J["occasion"] = []
for o, lab in [("Halloween", "Halloween"), ("Homecoming", "Homecoming"), ("4th of July", "4th of July"), ("Valentine", "Valentine's Day"), ("New Year", "New Year's Eve"), ("Graduation", "Graduation"), ("St. Patrick", "St. Patrick's Day"), ("", "No tag")]:
    ids = occ[occ == ""].index if o == "" else occ[occ.str.contains(o)].index
    s = M.loc[M.index.isin(ids), cols25].sum()
    J["occasion"].append(dict(tag=lab, n=int(len(ids)), n_targets=int(len(set(ids) & set(T))), idx=[round(float(x), 2) for x in s / s.mean()]))
J["worldcup_n"] = int(occ.str.contains("World Cup").sum())
# 6 launch curve
lc = []
for sc, p in it.pub.items():
    if pd.Timestamp("2024-02-01") <= p <= pd.Timestamp("2025-06-30") and sc in M.index:
        s = M.loc[sc]; s = s[s.index >= p.strftime("%Y-%m")]; base = s.iloc[1:4].mean()
        if base >= 30 and len(s) >= 13: lc.append(s.values[:13] / base)
L = pd.DataFrame(lc)
J["launch"] = dict(n=len(L), p25=[round(float(x), 3) for x in L.quantile(.25)], med=[round(float(x), 3) for x in L.median()], p75=[round(float(x), 3) for x in L.quantile(.75)])
# 7 size mix
sz = pd.read_parquet("data/monthly_item_size.parquet")
MAC = {"Top": "Tops", "Tank Top": "Tops", "T-Shirt": "Tops", "Crop Top": "Tops", "Corset": "Tops", "Jeans": "Bottoms", "Shorts": "Bottoms", "Skirt": "Bottoms", "Skort": "Bottoms", "Pants": "Bottoms", "Hoodie": "Warm", "Sweater": "Warm", "Sweatpants": "Warm", "Dress": "Dresses"}
sz["grp"] = it.reindex(sz.style_color).product_type.map(MAC).fillna("Other").values
tab = sz.groupby(["grp", "size"]).qty.sum().unstack(fill_value=0)
order = ["XS", "S", "M", "L", "XL", "XS/S", "M/L", "OS"]
tab = tab.reindex(columns=order, fill_value=0); tab = tab.div(tab.sum(axis=1), axis=0)
J["sizes"] = {"sizes": order, "rows": [dict(grp=g, share=[round(float(x), 3) for x in r]) for g, r in tab.iterrows()]}
# 8 item master facts
J["published_years"] = {str(y): [int(((it.pub.dt.year == y) & ~it.index.isin(T)).sum()), int(((it.pub.dt.year == y) & it.index.isin(T)).sum())] for y in range(2020, 2027)}
J["colour_num"] = {str(k): int(v) for k, v in it.color_num.value_counts().sort_index().items()}
# 9 multiplier drift and backtest per origin
F = pd.read_parquet("data/features.parquet"); A = F[(F.h == 1) & F.ytot.notna()].copy()
A["yr"] = A.origin.str[:4]; A["mo"] = A.origin.str[5:]
agg = A.groupby(["mo", "yr"]).apply(lambda x: x.ytot.sum() / (4 * x.level.sum())).unstack()
mat = A[A.age_m >= 12].groupby(["mo", "yr"]).apply(lambda x: x.ytot.sum() / (4 * x.level.sum())).unstack()
J["drift"] = [dict(mo=mo, all={y: (None if pd.isna(agg.loc[mo, y]) else round(float(agg.loc[mo, y]), 2)) for y in agg.columns},
                   mature={y: (None if pd.isna(mat.loc[mo, y]) else round(float(mat.loc[mo, y]), 2)) for y in mat.columns}) for mo in agg.index]
r3 = pd.read_csv("data/round3_by_fold.csv"); r2 = pd.read_csv("data/round2_by_fold.csv"); r1 = pd.read_csv("data/m5_experiments_by_fold.csv")
def per(df, model, seg="all"):
    x = df[(df.model == model) & (df.seg == seg)].set_index("origin"); return {o: dict(wape=round(float(r.wape), 3), bias=round(float(r.bias), 3), w20=round(float(r.w20), 3), w50=round(float(r.w50), 3), n=int(r.n)) for o, r in x.iterrows()}
J["per_origin"] = {"Monthly LightGBM pair (rounds 2-3)": per(r3, "ens (L1 base + L1 full) [current best]"), "Tweedie on units": per(r1, "loss:tw_units"),
                   "Flat 91-day run-rate": per(r1, "flat (91-day run-rate)"), "Structured baseline": per(r3, "B0 level x like-for-like season x age")}
J["per_origin_targets"] = per(r3, "ens (L1 base + L1 full) [current best]", "targets")
# 10 feature importance of the final full model
OCC = [c for c in F.columns if c.startswith("occ_")]
for c in ["ptype", "macro"] + [c for c in F.columns if c.startswith("a_")]: F[c] = F[c].astype("category")
BASE = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "age_m", "ptype", "tmonth", "h", "color_num", "own_sea"] + OCC
FULL = BASE + ["f_r7", "f_r28", "trend91", "peak_r", "days_live", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ"] + [c for c in F.columns if c.startswith("a_")]
tr = F[(F.o_ord + F.h <= pd.Period("2026-06").ordinal) & F.y.notna()]
m = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=0, n_jobs=4).fit(tr[FULL], np.log((tr.y + 1) / (tr.level + 1)))
imp = pd.Series(m.booster_.feature_importance("gain"), index=FULL); imp = imp / imp.sum()
grp = lambda c: "occasion tags" if c.startswith("occ_") else ("tagger attributes" if c.startswith("a_") else ("last 6 months (lags)" if c.startswith("lagr") else c))
gi = imp.groupby([grp(c) for c in imp.index]).sum().sort_values(ascending=False)
J["importance"] = [dict(f=k, g=round(float(v), 4)) for k, v in gi.items()]
# 11 targets table: the SUBMITTED workbook (D350: a copy of candidate_nx4 = candidate_nx3 + the D318 confidence rule and the D317 ranges, D299) with its cascade (data/final_2026_nx3_cascade.csv);
#    prev = the original 12 Nov submission (kept as a backup file). Workbooks are opened read-only.
C = pd.read_csv("data/final_2026_nx12_cascade.csv", index_col=0); W0 = pd.read_csv("data/final_forecast.csv", index_col=0); tv = pd.read_csv("data/targets_overview.csv", index_col=0)
def wbrows(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True); r = {x[1]: x for x in wb["Forecast"].iter_rows(min_row=2, max_row=101, values_only=True)}; wb.close(); return r
WB = wbrows("Edikted_Forecasting_Exercise_filled.xlsx"); WB0 = wbrows("candidates/Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx")
nn_ = lambda v, d=0: (None if v is None or pd.isna(v) else (int(round(v)) if d == 0 else round(float(v), d)))
J["targets"] = []
for sc in T:
    w, t, cz = WB[sc], tv.loc[sc], C.loc[sc]; f = [int(round(x)) for x in w[6:10]]
    J["targets"].append(dict(sc=sc, name=t.product_name, type=t.product_type, pub=str(t.published_date)[:10], occ=(t.occasion if isinstance(t.occasion, str) else ""),
        age=round(float(W0.loc[sc, "age_m"]), 1), level=int(round(W0.loc[sc, "level"])), jun=int(M.loc[sc, "2026-06"]), l12=int(t.L12), jo25=int(t.JulOct25),
        f=f, fin=int(sum(f)), raw=nn_(w[13]), prev=int(round(sum(WB0[sc][6:10]))), conf=int(w[11]), cn=int(it.loc[sc, "color_num"]), expl=w[12] or "",
        r3=nn_(cz.R3), season=nn_(cz.season), rules=nn_(cz.rules), tree=nn_(cz.qnx), nnm=nn_(cz.nn), pf2c=nn_(cz.pf2c), pf2d=nn_(cz.pf2d32), avg5=nn_(cz.avg5),
        corr=nn_(cz.correction, 3), buy=nn_(cz.buying), p10=nn_(cz.P10), p90=nn_(cz.P90), spread=nn_(cz.member_spread, 4)))
assert sum(t["fin"] for t in J["targets"]) == 630254, sum(t["fin"] for t in J["targets"])
H = pd.read_csv("data/final_2026_nx12_cascade_hierarchy.csv"); h0 = H[H.level == "total"].iloc[0]
J["system"] = dict(total=int(sum(t["fin"] for t in J["targets"])), months=[int(sum(t["f"][j] for t in J["targets"])) for j in range(4)], buy=int(round(h0.buying)),
                   orig=int(sum(t["prev"] for t in J["targets"])), conf=[int(sum(t["conf"] == k for t in J["targets"])) for k in (1, 2, 3)], n_expl=int(sum(bool(t["expl"]) for t in J["targets"])),
                   cascade=[dict(step=k, total=int(round(h0[c]))) for k, c in [("R3: last 3 months x 4", "R3"), ("x type seasonality", "season"), ("member: rule agents", "rules"), ("member: LightGBM (events + launch curves)", "qnx"),
                            ("member: sales-curve NN", "nn"), ("member: TabPFN v2, conformal", "pf2c"), ("member: TabPFN v2, 32 estimators", "pf2d32"), ("arithmetic mean of the five", "avg5"),
                            ("x segment and breakout corrections = forecast", "forecast"), ("with the two hand rules (out of stock, World Cup) = submitted", "final"), ("buying quantity (quantile members at P65)", "buying")]])
J["links"] = dict(scoreboard="https://claude.ai/artifact/9ZPDJvYzHq2cUqELfcXdVY", error_analysis="https://claude.ai/artifact/NkoNJTZUBirD4soyyH9sDv")   # the Edikted Error Study (D277)
# 12 decision log. Ids such as D125b; repeated ids (addenda, one corrected entry) get a unique key "D253·2"; sub-bullets and tables are kept
#    in the body. Entries after the last "## Turn" heading are grouped by id range (two sessions wrote in parallel from D161 on).
log = open("log.md", encoding="utf-8").read(); turn = ""; J["log"] = []
entries = []
for line in log.splitlines():
    if line.startswith("## Turn"): turn = line[3:].strip(); continue
    mm = re.match(r"- \*\*(D\d{3}[a-z]?) (.+?)\*\*\s*(.*)", line)
    if mm: entries.append([mm.group(1), mm.group(2), mm.group(3), turn]); continue
    if entries and line.startswith("  ") and line.strip():
        st = line.strip()
        if st.startswith("|"):
            if set(st) <= set("|-: "): continue
            st = " · ".join(x.strip() for x in st.strip("|").split("|")) + ";"
        entries[-1][2] += " " + re.sub(r"^\s*-\s*", "", st)   # sub-bullets belong to the entry
LAST = turn
GROUPS = [(148, 163, "Carts, style embeddings, weather and events, clusters, the scoreboard"), (164, 189, "Main session: content-embedding gate and the M5 round"),
          (190, 229, "Fork session: explainability, foundation models, dynamic co-purchase network, TabPFN, M5 part, submission inputs"),
          (230, 251, "Main session: arithmetic stacks, final chain, decomposition and error analysis"), (252, 269, "Main session: fixable errors, new tree member, candidates"),
          (270, 299, "Fork session: member rebuilds, submission switches, atlas")]
seen = {}
for eid, title, body, tn in entries:
    n = int(eid[1:4])
    if tn == LAST:
        for lo, hi, nm in GROUPS:
            if lo <= n <= hi: tn = nm
    seen[eid] = seen.get(eid, 0) + 1; key = eid if seen[eid] == 1 else f"{eid}·{seen[eid]}"
    body = body.strip(); sm = re.search(r"Status: ([^.]+)\.?\s*$", body)
    J["log"].append(dict(id=key, title=title.rstrip("."), body=re.sub(r"\s*Status: [^.]+\.?\s*$", "", body), status=sm.group(1) if sm else "", turn=tn))
# 13 files
def doc(p):
    s = open(p, encoding="utf-8", errors="ignore").read(); mm = re.search(r'"""(.*?)"""', s, re.S)
    return (mm.group(1).strip().splitlines()[0] if mm else "")
J["files"] = {"scripts": [dict(p=p.replace("\\", "/"), doc=doc(p)) for p in sorted(glob.glob("eda/*.py"))],
              "data": [dict(p=p.replace("\\", "/"), kb=round(os.path.getsize(p) / 1024)) for p in sorted(glob.glob("data/*"))]}
json.dump(J, open("data/project_map.json", "w", encoding="utf-8"), ensure_ascii=False)
print("ok", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in J.items()})
print("importance:", J["importance"][:8])
