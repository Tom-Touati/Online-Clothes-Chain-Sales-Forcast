"""Builds the submitted workbook, nx8 (D421; user, 18 Nov: "Combine them" = the forecast numbers of the campaign version nx7c with the
corrected text, explanations and "Method and decisions" sheet of the audit candidate nx7).

One consolidated builder: it replaces the chain eda/181k -> 181l -> 181m -> 181o (each a patched copy of the one before) and the generated
eda/181n builder. The numbers are those of eda/181o_final_nx7c_2026.py, step for step (checked cell for cell by eda/181p_check_nx8.py);
the words come from eda/181p_nx8_text.py.

Steps, for the 100 items of the exercise workbook, at the June 2026 origin:
  1  the six member forecasts of the July-October total (rules, LightGBM tree, neural network, TabPFN calibrated, TabPFN 32, recommender)
  2  their weighted mean (the recommender at a quarter weight) x the correction learnt from earlier forecast errors = the model forecast
  3  split by month, by the average monthly shape of the rules and the neural network
  4  the campaign plan: August and September raised for the two new site sales, by each item's edikted.com share
  5  hand rules: the out-of-stock item (forecast as if in stock) and the three World Cup items
  6  P10 / P25 / P75 / P90 from the per-segment range ratios; ranges for totals
  7  confidence 1-3;  8  buying quantity;  9  explanations
  10 the workbook (Forecast and Model explanation sheets filled; Method and decisions, Cascade and Ranges sheets added) and the cascade tables
Run from anywhere:  python eda/181p_final_nx8_2026.py"""
import importlib.util, math, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, openpyxl
from openpyxl.styles import Alignment, Font
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------- where things are
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEMPLATE = ROOT / "Edikted_Forecasting_Exercise.xlsx"                  # the exercise workbook as received
WORKBOOK = ROOT / "candidates/Edikted_Forecasting_Exercise_candidate_nx8.xlsx"    # written
CASCADE = DATA / "final_2026_nx8_cascade.csv"                           # written: every step, per item
HIERARCHY = DATA / "final_2026_nx8_cascade_hierarchy.csv"               # written: the steps summed by total / product type / style
TOTAL_RANGES = DATA / "final_2026_nx8_total_ranges.csv"                # written: ranges for totals
TEXT = Path(__file__).with_name("181p_nx8_text.py")
IN = dict(
    items=DATA / "items.parquet",                                  # product type per item
    monthly=DATA / "monthly_item.parquet",                         # units per item and month
    features=DATA / "features.parquet",                            # per origin and item: recent monthly rate and the type's seasonal ratio
    daily=DATA / "daily_item_chan.parquet",                        # units per item, day and channel (edikted.com share)
    rules=ROOT / "agent_eval" / "2026-06" / "predictions_verified.csv",   # member 1: the planning rules, by month
    tree=DATA / "q_backtest_ls_huber_routed_2026.parquet",         # member 2: LightGBM, launch-weighted version for items under six months
    nn=[DATA / "nn_backtest_preds_finalSec.parquet", DATA / "nn_backtest_preds_finalSec_s10.parquet"],   # member 3: two seeds, by month
    tabpfn_calibrated=DATA / "fk_2026_pf2c.parquet",               # member 4 (with quantiles)
    tabpfn32=DATA / "fk_2026_pf2d32.parquet",                      # member 5 (with quantiles)
    recommender=DATA / "rc_recsys_preds.parquet",                  # member 6
    correction=DATA / "geo_nx5r_x_2026.parquet",                   # weighted mean (F) and corrected forecast (geo) for every item
    ranges=DATA / "ranges_nx5r_2026.csv",                          # range ratios r10 / r25 / r75 / r90 per item
    level1=DATA / "fk_confidence_2026.csv",                        # the tested rule for confidence 1 (narrow range, rising, sold last year)
    buying_tree=DATA / "q_backtest_nx_w03ec_2026.parquet",         # the tree's 65th percentile, for the buying quantity
    buying_tree_fallback=DATA / "fk_2026_w03.parquet",
    out_of_stock=DATA / "cand_rwxs_final_forecast_q3.csv",         # the in-stock forecast of the item with no recent sales
    without_routing=DATA / "final_2026_nx5_d361_cascade.csv",      # the same system before the launch-weighted tree, to quote its effect
)
ORIGIN = "2026-06"
MONTHS = ["jul", "aug", "sep", "oct"]
MEMBERS = ["rules", "qnx", "nn", "pf2c", "pf2d32", "rc"]              # column names as in the cascade tables (qnx = the tree, rc = the recommender)
WEIGHTS = np.array([1, 1, 1, 1, 1, 0.25])
CAMPAIGN_LIFT = 0.22                                                  # half of log 1.55: a past campaign day sold 1.55x a normal day on edikted.com
NEW_SALE_DAYS = {"aug": 8 / 31, "sep": 6 / 30}                        # Clearance 12-19 August; the 24-29 September sale
WORLD_CUP = {"S24840_LIGHT-GRAY-MELANGE": [8554, 6416, 4277, 3849],   # July at the late-June pace to the 19 July final; Aug-Oct at 75 / 50 / 45% of July
             "S25039_YELLOW": [3366, 2524, 1683, 1515], "S24847_NAVY": [844, 633, 422, 380]}
spec = importlib.util.spec_from_file_location("workbook_text", TEXT); text = importlib.util.module_from_spec(spec); spec.loader.exec_module(text)

# ---------------------------------------------------------------- the 100 items
template = openpyxl.load_workbook(TEMPLATE)
ITEMS = [row[1].value for row in template["Forecast"].iter_rows(min_row=2) if row[1].value]
per_item = lambda path: pd.read_parquet(path).drop_duplicates("style_color").set_index("style_color").reindex(ITEMS)
monthly = pd.read_parquet(IN["monthly"]); monthly.columns = [str(c) for c in monthly.columns]; monthly = monthly.reindex(ITEMS)
C = pd.DataFrame(index=ITEMS)                                         # the cascade: one row per item, one column per step
C["R3"] = 4 * monthly[["2026-04", "2026-05", "2026-06"]].mean(axis=1)             # the recency baseline: last three months x 4
feat = pd.read_parquet(IN["features"]); feat = feat[feat.origin == ORIGIN]
C["season"] = feat.assign(st=feat.m3 * np.exp(feat.lfl_type.fillna(0))).groupby("style_color").st.sum().reindex(ITEMS).fillna(C.R3)   # x the product type's seasonal pattern

# ---------------------------------------------------------------- 1 the six members
rules_m = pd.read_csv(IN["rules"], dtype={"month": str}).pivot(index="style_color", columns="month", values="pred").reindex(ITEMS); rules_m.columns = MONTHS
nn_m = sum(pd.read_parquet(p).pivot(index="style_color", columns="h", values="q50") for p in IN["nn"]) / 2; nn_m = nn_m.reindex(ITEMS); nn_m.columns = MONTHS
tree, pf_cal, pf32 = per_item(IN["tree"]), per_item(IN["tabpfn_calibrated"]), per_item(IN["tabpfn32"])
C["rules"] = rules_m.sum(axis=1); C["qnx"] = tree.q50; C["nn"] = nn_m.sum(axis=1); C["pf2c"] = pf_cal.q50; C["pf2d32"] = pf32.q50
for m in ["rules", "nn"]: C[m] = C[m].fillna(C[["qnx", "pf2c", "pf2d32"]].mean(axis=1))
rc = pd.read_parquet(IN["recommender"]); rc = rc[rc.origin == ORIGIN].drop_duplicates("style_color").set_index("style_color").rc_deepfm
C["rc"] = rc.reindex(ITEMS).values; C["rc"] = C.rc.fillna(C[MEMBERS[:5]].mean(axis=1))

# ---------------------------------------------------------------- 2 weighted mean x correction = the model forecast
X = C[MEMBERS].clip(lower=0).values; ok = np.isfinite(X)
C["avg5"] = np.where(ok.any(axis=1), np.nansum(np.where(ok, X, 0) * WEIGHTS, axis=1) / np.maximum((ok * WEIGHTS).sum(axis=1), 1e-9), np.nan)
geo = pd.read_parquet(IN["correction"]).set_index("style_color").reindex(ITEMS)
assert np.allclose(geo.F.dropna(), C.avg5[geo.F.notna()], rtol=1e-6), "the members differ from those the correction was fitted on"
C["correction"] = (geo.geo / geo.F.clip(lower=1e-9)).fillna(1.0)
C["forecast"] = C.avg5 * C.correction
C["member_spread"] = C[MEMBERS].clip(lower=1).max(axis=1) / C[MEMBERS].clip(lower=1).min(axis=1)
model_p10 = C.forecast * pf_cal.q10 / pf_cal.q50.clip(lower=1); model_p90 = C.forecast * pf_cal.q90 / pf_cal.q50.clip(lower=1)   # the models' own range (quoted for the World Cup items)

# ---------------------------------------------------------------- 3 monthly split
share = lambda months: months.div(months.sum(axis=1), axis=0).clip(lower=1e-4)
profile = np.exp((np.log(share(rules_m)) + np.log(share(nn_m))) / 2); profile = profile.div(profile.sum(axis=1), axis=0).fillna(0.25)
for m in MONTHS: C[m] = (C.forecast * profile[m]).round().astype(int)

# ---------------------------------------------------------------- 4 the campaign plan: the two new site sales
daily = pd.read_parquet(IN["daily"], columns=["ny_date", "style_color", "chan", "qty", "cancelled"]); daily = daily[~daily.cancelled]; daily["ny_date"] = pd.to_datetime(daily.ny_date)
spring = daily[(daily.ny_date >= "2026-04-01") & (daily.ny_date <= "2026-06-30")]
units, web = spring.groupby("style_color").qty.sum(), spring[spring.chan == "edikted.com"].groupby("style_color").qty.sum()
C["web_share"] = (web.reindex(ITEMS).fillna(0) / units.reindex(ITEMS)).fillna(0.0).clip(0, 1)
for m, days in NEW_SALE_DAYS.items():
    C["uplift_" + m] = np.exp(CAMPAIGN_LIFT * days * C.web_share); C[m] = (C[m] * C["uplift_" + m]).round().astype(int)
C["campaign_add"] = C[MONTHS].sum(axis=1) - C.forecast * profile[MONTHS].sum(axis=1)

# ---------------------------------------------------------------- 5 hand rules (they keep their own months: no campaign uplift)
in_stock = pd.read_csv(IN["out_of_stock"], index_col=0); OUT_OF_STOCK = []
for s in ITEMS:                                                       # no model covers an item without recent sales; the brief says to assume stock
    if C.loc[s, ["qnx", "pf2c", "pf2d32"]].isna().all() and C.loc[s, "rules"] == 0 and s in in_stock.index:
        for m in MONTHS: C.loc[s, m] = int(in_stock.loc[s, "f_" + m])
        C.loc[s, ["avg5", "forecast"]] = in_stock.loc[s, ["f_" + m for m in MONTHS]].sum(); OUT_OF_STOCK.append(s)
for s, months in WORLD_CUP.items():                                   # the models extend the June ramp past the final; the model total stays in "forecast"
    for m, v in zip(MONTHS, months): C.loc[s, m] = int(v)
HAND_RULED = list(WORLD_CUP) + OUT_OF_STOCK
C.loc[HAND_RULED, "campaign_add"] = 0.0
C["total"] = C[MONTHS].sum(axis=1); C["final"] = C.total             # "final" is the name the pages read

# ---------------------------------------------------------------- 6 ranges
ratios = pd.read_csv(IN["ranges"], comment="#").drop_duplicates("style_color").set_index("style_color").reindex(ITEMS)
for q in ("10", "25", "75", "90"): C["P" + q] = ((C.total + 1) * ratios["r" + q] - 1).clip(lower=0).round()
C["P25"] = np.maximum(C.P25, C.P10); C["P75"] = np.minimum(C.P75, C.P90)
ranged = C.P10.notna(); assert ((C.P10 <= C.P25) & (C.P25 <= C.total) & (C.total <= C.P75) & (C.P75 <= C.P90))[ranged].all()
# ranges for totals: item half-ranges added in quadrature plus a shared term (item misses partly move together); never the sum of item ranges
half_lo = (C.total - ((C.total + 1) * ratios.r25 - 1).clip(lower=0)) / 0.6745; half_hi = (((C.total + 1) * ratios.r75 - 1).clip(lower=0) - C.total) / 0.6745
C["product_type"] = pd.read_parquet(IN["items"]).set_index("style_color").product_type.reindex(ITEMS).values; C["style"] = pd.Index(ITEMS).str.split("_").str[0]
SHARED, Z = 0.1278, {"P10": -1.801, "P25": -0.774, "P75": 0.921, "P90": 1.525}
parts = pd.DataFrame({"F": C.total, "lo": half_lo ** 2, "hi": half_hi ** 2, "product_type": C.product_type, "style": C["style"], "all": "all 100 targets"})
totals = []
for label, key in [("total", "all"), ("product type", "product_type"), ("style", "style")]:
    g = parts.groupby(key).agg(P50=("F", "sum"), items=("F", "size"), lo=("lo", "sum"), hi=("hi", "sum")).reset_index().rename(columns={key: "group"}); g = g[g["items"] >= 3]
    spread = np.sqrt(((np.sqrt(g.lo) + np.sqrt(g.hi)) / 2 / (g.P50 + 1)) ** 2 + SHARED ** 2)
    for name, z in Z.items(): g[name] = (np.exp(np.log(g.P50 + 1) + z * spread) - 1).round()
    g.insert(0, "level", label); totals.append(g[["level", "group", "items", "P10", "P25", "P50", "P75", "P90"]])
TOTALS = pd.concat(totals, ignore_index=True)

# ---------------------------------------------------------------- 7 confidence, by the scale's own definitions
knots = np.log1p(np.column_stack([C.P10, C.P25, C.total, C.P75, C.P90]).astype(float).clip(min=0)); knots = np.maximum.accumulate(knots, axis=1)
def chance_below(x):                                                  # the item's range read as a distribution: P(actual < x), interpolated in logs
    out = []
    for k, v in zip(knots, np.log1p(np.maximum(np.asarray(x, float), 0))):
        if not np.isfinite(k).all(): out.append(np.nan); continue
        out.append(np.interp(v, np.r_[k[0] - 2 * max(k[1] - k[0], .05), k, k[4] + 2 * max(k[4] - k[3], .05)], [0, .1, .25, .5, .75, .9, 1]))
    return np.array(out)
C["p_within20"] = (chance_below(1.2 * C.total) - chance_below(0.8 * C.total)).round(3)
C["p_within50"] = (chance_below(1.5 * C.total) - chance_below(0.5 * C.total)).round(3)
level1 = pd.read_csv(IN["level1"]).set_index("style_color").level.reindex(ITEMS)
C["confidence"] = np.where(level1.values == 1, 1, np.where(C.p_within50.fillna(1).values < 0.5, 3, 2)).astype(int)   # 3 = less than even odds of landing within +/-50%
C.loc[HAND_RULED, "confidence"] = 3

# ---------------------------------------------------------------- 8 buying quantity: the total x how far the members' 65th percentile sits above their median
buy_tree, fallback = per_item(IN["buying_tree"]), per_item(IN["buying_tree_fallback"])
upper = pd.concat([C.rules, buy_tree.q65.fillna(fallback.q65), C.nn, pf_cal.q65, pf32.q65], axis=1).clip(lower=0).mean(axis=1)
middle = pd.concat([C.rules, buy_tree.q50.fillna(fallback.q50), C.nn, pf_cal.q50, pf32.q50], axis=1).clip(lower=0).mean(axis=1)
C["buying"] = C.total * (upper / middle.clip(lower=1e-9)).clip(lower=1.0).fillna(1.0)

# ---------------------------------------------------------------- 9 explanations
n = lambda v: f"{float(v):,.0f}"
before_routing = pd.read_csv(IN["without_routing"], index_col=0).forecast.reindex(ITEMS)
def explain(s):
    r = C.loc[s]; mo = monthly.loc[s]
    adjusted = (text.ADJUSTED.format(pct=f"{r.campaign_add / max(r.forecast, 1):+.1%}", total=f"{int(r.total):,}") if r.campaign_add >= 0.005 * r.forecast else text.NOT_ADJUSTED)
    return text.EXPLANATIONS[s].format(
        apr=n(mo["2026-04"]), may=n(mo["2026-05"]), jun=n(mo["2026-06"]), spring=n(mo[["2026-04", "2026-05", "2026-06"]].mean()), spring_ly=n(mo[["2025-04", "2025-05", "2025-06"]].mean()),
        last_year=n(mo[["2025-07", "2025-08", "2025-09", "2025-10"]].sum()), lo=n(r[MEMBERS].min()), hi=n(r[MEMBERS].max()), raw=f"{round(float(r.forecast)):,}",
        jul=f"{int(r.jul):,}", oct=f"{int(r.oct):,}", lift=f"{r.forecast / before_routing[s] - 1:.0%}", adjusted=adjusted)
C["explanation"] = ""
for s in text.EXPLANATIONS: C.loc[s, "explanation"] = explain(s)
for s in OUT_OF_STOCK: C.loc[s, "explanation"] = text.OUT_OF_STOCK
for s in WORLD_CUP: C.loc[s, "explanation"] = text.WORLD_CUP.format(fact=text.WORLD_CUP_FACTS[s], model=n(C.forecast[s]), p10=n(model_p10[s]), p90=n(model_p90[s]), final=f"{int(C.total[s]):,}")
assert (C.explanation != "").sum() >= 15

# ---------------------------------------------------------------- 10 the tables and the workbook
C.to_csv(CASCADE)
STEPS = ["R3", "season"] + MEMBERS + ["avg5", "forecast", "campaign_add", "final", "buying"]
H = pd.concat([C[STEPS].sum().to_frame("total").T.assign(level="total", group="all 100 targets"),
               C.groupby("product_type")[STEPS].sum().assign(level="product type").reset_index().rename(columns={"product_type": "group"}),
               C.groupby("style")[STEPS].sum().assign(level="style").reset_index().rename(columns={"style": "group"})], ignore_index=True)
H.to_csv(HIERARCHY, index=False); TOTALS.to_csv(TOTAL_RANGES, index=False)
fill = dict(campaign_pct=f"{C.campaign_add.sum() / (C.total.sum() - C.campaign_add.sum()):+.1%}",   # against the total before the uplift
            buying_uplift=f"{round(float((C.buying / C.total.clip(lower=1)).median() - 1) * 100)}")

shutil.copyfile(TEMPLATE, WORKBOOK); wb = openpyxl.load_workbook(WORKBOOK)
ws = wb["Forecast"]; col = {c.value: c.column for c in ws[1]}
for row in ws.iter_rows(min_row=2):
    s = row[1].value
    if s not in C.index: continue
    r = C.loc[s]; put = lambda header, value: ws.cell(row=row[0].row, column=col[header], value=value)
    for m, header in zip(MONTHS, ["Jul-2026 units", "Aug-2026 units", "Sep-2026 units", "Oct-2026 units"]): put(header, int(r[m]))
    put("Total Jul-Oct", int(r.total)); put("Confidence (1-3)", int(r.confidence)); put("Explanation (at least 15 items of your choice)", r.explanation or None)
    put("Model raw total Jul-Oct (optional, before your adjustments)", round(float(r.forecast)))
wm = wb["Model explanation"]                                         # the three boxes, enlarged to their text (the template rows are 220 points; Excel's limit is 409)
for row in wm.iter_rows(min_row=2):
    if row[0].value in text.MODEL_EXPLANATION:
        t = text.MODEL_EXPLANATION[row[0].value].format(**fill); wm.cell(row=row[0].row, column=2, value=t)
        wm.row_dimensions[row[0].row].height = min(409, max(220, 15.5 * sum(max(1, math.ceil(len(p) / 118)) for p in t.split("\n")) + 8))
wd = wb.create_sheet("Method and decisions", wb.sheetnames.index("Model explanation") + 1)
wd.append(["Each major choice behind the forecast, and the evidence for it. Scores are the forecast's error relative to R3 (the last three months x 4) on the nine most recent backtest month-ends unless said otherwise; lower is better."])
wd.merge_cells("A1:C1"); wd["A1"].alignment = Alignment(wrap_text=True, vertical="top"); wd.row_dimensions[1].height = 34
wd.append(["Area", "Decision", "Why, and the evidence"])
for c in wd[2]: c.font = Font(bold=True)
for area, decision, why in text.DECISIONS:
    wd.append([area, decision, why.format(**fill)])
    for c in wd[wd.max_row]: c.alignment = Alignment(wrap_text=True, vertical="top")
    wd.row_dimensions[wd.max_row].height = 15.5 * max(math.ceil(len(decision) / 44), math.ceil(len(why) / 128)) + 6
wd.column_dimensions["A"].width = 14; wd.column_dimensions["B"].width = 42; wd.column_dimensions["C"].width = 120; wd.freeze_panes = "A3"
HEAD = {"R3": "Recent rate x4 (R3)", "season": "Seasonal baseline", "rules": "Model 1: planning rules", "qnx": "Model 2: LightGBM", "nn": "Model 3: neural network", "pf2c": "Model 4: TabPFN, calibrated",
        "pf2d32": "Model 5: TabPFN", "rc": "Model 6: recommender (1/4 weight)", "avg5": "Weighted average", "forecast": "After corrections (raw model total)", "web_share": "Share sold on edikted.com",
        "campaign_add": "Campaign uplift (units)", "final": "Final (Forecast sheet total)", "buying": "Buying quantity", "member_spread": "Highest / lowest model", "confidence": "Confidence"}
items_sheet = C.reset_index().rename(columns={"index": "style_color"})[["style_color", "product_type", "R3", "season"] + MEMBERS + ["avg5", "forecast", "web_share", "campaign_add", "final", "buying", "member_spread", "confidence", "P10", "P25", "P75", "P90"]]
for name, table in [("Cascade (items)", items_sheet.rename(columns=HEAD)), ("Cascade (hierarchy)", H[["level", "group"] + STEPS].rename(columns=HEAD)), ("Ranges (totals)", TOTALS)]:
    sheet = wb.create_sheet(name); sheet.append(list(table.columns))
    for values in table.itertuples(index=False): sheet.append([float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for v in values])
wb.save(WORKBOOK)

months_total = C[MONTHS + ["total"]].sum().astype(int).to_dict()
print("nx8:", months_total, "| model raw total", round(float(C.forecast.sum())), "| campaign", fill["campaign_pct"], round(float(C.campaign_add.sum())), "units, items >= 0.5%:", int((C.campaign_add >= 0.005 * C.forecast).sum()))
print("confidence", C.confidence.value_counts().sort_index().to_dict(), "| explanations", int((C.explanation != "").sum()), "| buying", round(float(C.buying.sum())), f"(+{fill['buying_uplift']}% median), below the total for", int((C.buying < C.total - 0.5).sum()), "items")
print("hand-ruled:", HAND_RULED, "| boxes:", {row[0].value: (len(row[1].value), wm.row_dimensions[row[0].row].height) for row in wm.iter_rows(min_row=2, max_row=4)})
