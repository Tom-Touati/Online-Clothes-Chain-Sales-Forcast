"""D343 (fork 2; user: "update the atlas for final inspection. put in all the features used of course"). Every input of every model in the
candidate system (Main D358: Huber tree member + rc_deepfm at a quarter weight, corrections re-fitted) and in the submitted system,
listed per model with its block and a plain description. Lists are taken from the code that builds each model, not retyped:
  tree member ec_ns / Huber   eda/180q_models.py sets("ec_ns"): fk_explain_meta FULL + co-purchase network (DY) - static descriptors + EV2 + CURVE
  TabPFN v2 members           eda/196b_tabpfn.py feat=dyn: fk_explain_meta FULL + DY (conformal and 32-estimator versions share the inputs)
  neural net S13ecx6          eda/57_embed_nn.py NUM + embeddings, + EV2 + CURVE (eda/197b_nn_ec.py, D272)
  rule agents                 agent_eval/<origin>/rules.py (written per origin; inputs = the recent monthly history and the item calendar)
  location recommender        eda/202_rc_model.py NUM + IF (fk_explain_table numeric inputs) + embeddings
  corrections                 eda/179x2_bias.py segments, eda/179b_breakout.py SIG (+ XSIG)
Writes data/feature_inventory.json ({models: [{id, name, role, weight, source, n, features: [{name, block, desc}]}]})."""
import json, re, sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
META = json.load(open("data/fk_explain_meta.json")); BLK = json.load(open("data/nf2_meta.json")); BLK["EV2"] = [c for c in BLK["EV2"] if c != "c_pay"]; GROUP = META["GROUP"]
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
STATIC = [c for c in META["FULL"] if c.startswith("a_") or c == "ptype"] + ["warmth", "coverage", "formality", "fit_tightness", "statement", "staple", "summer", "winter",
          "back_to_school", "party", "halloween", "festival", "layering", "versatility"]
GLOSS = [(r"^lagr(\d)$", "units {0} month(s) before the origin, relative to the recent level"), (r"^lvl$", "recent monthly level (91-day rate × 30.44), log"), (r"^level$", "recent monthly level"),
         (r"^f_r7$", "last 7 days' pace against the level"), (r"^f_r28$", "last 28 days' pace against the level"), (r"^trend91$", "91-day trend"), (r"^peak_r$", "best month so far against the level"),
         (r"^sh_ret$", "own-store share of units"), (r"^sh_drop$", "partner (drop-ship) share of units"), (r"^sh_tt$", "TikTok share of units"), (r"^disc_item$", "the item's web discount"),
         (r"^disc_rel$", "the item's discount against the site's"), (r"^lfl_(type|macro|occ)4?$", "like-for-like seasonal ratio of the item's {0} (last year, target months)"),
         (r"^own_sea4?$", "the item's own last-year ratio for the target months"), (r"^color_num$", "colour number within the style (1 = first colour)"), (r"^age_m$", "age in months"),
         (r"^days_live$", "days live"), (r"^cp_nb_(.*)$", "co-purchase neighbours: {0}"), (r"^dy_nb_(.*)$", "dynamic co-purchase network neighbours: {0}"), (r"^ev_(.*)_t$", "event window in the target months: {0}"),
         (r"^ev_(.*)_o$", "event window around the origin: {0}"), (r"^x_(.*)$", "item score × season or event interaction: {0}"), (r"^c_(.*)$", "calendar count in the target months: {0}"),
         (r"^curve_(.*)$", "launch curve: {0}"), (r"^tmonth$", "target month"), (r"^t_tgt_norm$", "normal temperature of the target months"), (r"^dT_norm$", "temperature change from the origin to the target months (normals)"),
         (r"^t_org$", "temperature at the origin"), (r"^ta_org$", "temperature anomaly at the origin"), (r"^beta_T$", "the item's sensitivity of sales to temperature"), (r"^bT_(.*)$", "temperature sensitivity × {0}"),
         (r"^occ_(.*)$", "occasion timing: {0}"), (r"^h$", "horizon (months ahead)"), (r"^l3$", "units at this location, last 3 months (log)"), (r"^l1$", "units at this location, last month (log)"),
         (r"^ly$", "units at this location a year before the target month (log)"), (r"^pair_age$", "months the item has sold at this location (log)"), (r"^loc_age$", "months the location has been open (log)"),
         (r"^loc3$", "the location's total units, last 3 months (log)"), (r"^it3$", "the item's units at all locations, last 3 months (log)"), (r"^share$", "this location's share of the item's units"),
         (r"^disc_in$", "discount input: the item's last discount at this location"), (r"^disc_last$", "the item's last discount at this location"), (r"^disc_d$", "planned minus last discount (0 when no plan)"),
         (r"^l_age$", "age in months (log)"), (r"^l_live$", "days live (log)"),
         (r"^e2_(.*)_t$", "event window in the target months: {0}"), (r"^e2_(.*)_o$", "event window around the origin: {0}"), (r"^acc7$", "last 7 days against the 28 before (acceleration)"),
         (r"^best_bucket$", "the item's best launch-curve bucket so far"), (r"^first30$", "units in the first 30 days after launch, against the level"), (r"^n_q4$", "number of October–December months in the window"),
         (r"^pe_(.*)$", "launch-curve fit: {0}"), (r"^pk_28$", "best 28 days in the last year against the level"), (r"^pk_365_days$", "days since the best 28 days of the last year"),
         (r"^pk_anniv_dist$", "months to the anniversary of the item's peak"), (r"^promo_lift$", "the item's sales lift on deep-discount days"), (r"^spread$", "disagreement between the five members (log range)"),
         (r"^wk_(.*)$", "weekly launch curve: {0}")]
def desc(c):
    for pat, txt in GLOSS:
        m = re.match(pat, c)
        if m: return txt.format(*[g.replace("_", " ") for g in m.groups()])
    return GROUP.get(c, "")
def blk(c):
    for k in ("EV2", "CURVE", "AGE", "TREND", "PEAK", "WDAY", "PLAN"):
        if c in BLK.get(k, []): return {"EV2": "events and calendar (D253)", "CURVE": "launch curve (D253)"}.get(k, k)
    if c in DY: return "dynamic co-purchase network (D198)"
    return GROUP.get(c, "history and item state")
def items(cols): return [dict(name=c, block=blk(c), desc=desc(c)) for c in cols]
tree = [c for c in META["FULL"] + DY if c not in STATIC] + BLK["EV2"] + BLK["CURVE"]; tree = list(dict.fromkeys(tree))
tab = list(dict.fromkeys(META["FULL"] + DY))
src = open("eda/57_embed_nn.py", encoding="utf-8").read(); NUM57 = [f"lagr{k}" for k in range(1, 7)] + ["lvl", "f_r7", "f_r28", "trend91", "peak_r", "sh_ret", "sh_drop", "sh_tt", "disc_item", "disc_rel", "lfl_type", "lfl_macro", "lfl_occ", "own_sea", "color_num", "l_age", "l_live"]
assert all(f"\"{c}\"" in src for c in NUM57[6:]), "eda/57 NUM changed"
nn_emb = [dict(name=n, block="embedding", desc=d) for n, d in [("macro group", "learned vector of the product group"), ("product type", "learned vector of the type"), ("style", "learned vector shared by the style's colours"),
          ("item", "learned vector of the item (falls back to style and type when new)"), ("occasion, collection, tagger attributes", "learned side vectors"), ("target month", "calendar month of each forecast month"),
          ("horizon", "months ahead"), ("item × target month", "the item's own seasonality")]]
LEAK = ("z", "ytot", "target", "dev_actual", "dT_real_from_org", "o_ord", "level", "R3")
T = pd.read_parquet("data/fk_explain_table.parquet")
IF = [c for c in T.columns if c not in LEAK and c not in ("origin", "style_color") and pd.api.types.is_numeric_dtype(T[c]) and not c.startswith("a_")]
RCN = ["h", "l3", "l1", "ly", "pair_age", "loc_age", "loc3", "it3", "share", "disc_in", "disc_last", "disc_d"]
rc_emb = [dict(name=n, block="embedding (factorisation machine)", desc=d) for n, d in [("item", "identity vector + a projection of its attributes (type, 7 tagged attributes, 14 fashion scores)"),
          ("location", "31 locations: edikted.com by US region (4) and other, TikTok, 18 own stores, 7 partners"), ("location kind", "web, TikTok, store, partner"), ("calendar month", "month of year of the target month"),
          ("interactions", "item × location, item × month, location × month: the compatibility terms")]]
SIG179 = ["age_m", "lagr1", "lagr2", "lagr3", "acc7", "trend91", "peak_r", "sh_ret", "sh_drop", "disc_item", "disc_rel", "sh_tt", "own_sea", "wk_mom", "wk_last", "wk_live", "wk_peak", "wk_recent", "pe_trend", "pe_mom",
          "pe_launch", "pe_resid", "spread", "n_q4", "pk_28", "pk_365_days", "pk_anniv_dist", "curve_fwd", "curve_dev", "promo_lift"]
assert all(c in open("eda/179b_breakout.py", encoding="utf-8").read() for c in SIG179)
M = [dict(id="rules", name="Rule agents", role="member, weight 1", source="agent_eval/<origin>/rules.py (D103–D105)", features=[dict(name=n, block="rules", desc=d) for n, d in [
        ("recent monthly units", "the last 3–6 months of the item's units"), ("launch age", "months since launch, for launch-curve rules"), ("type seasonality", "last year's like-for-like ratio of the product type"),
        ("stock and size gaps", "rules for items short of stock or sizes"), ("event and calendar windows", "dated events in the forecast months")]]),
     dict(id="tree", name="Quantile LightGBM, Huber loss on the log ratio (candidate; ec_ns in the submitted system)", role="member, weight 1", source="eda/180q_models.py sets('ec_ns'); D253, D353", features=items(tree)),
     dict(id="nn", name="Sales-curve neural net (S13ecx6)", role="member, weight 1", source="eda/57_embed_nn.py + eda/197b_nn_ec.py (D272)", features=items(NUM57 + BLK["EV2"] + BLK["CURVE"]) + nn_emb),
     dict(id="pf2c", name="TabPFN v2, conformal quantiles", role="member, weight 1", source="eda/196b_tabpfn.py feat=dyn; eda/199c_conformal.py (D207)", features=items(tab)),
     dict(id="pf2d32", name="TabPFN v2, 32 estimators", role="member, weight 1", source="eda/196b_tabpfn.py feat=dyn est=32 (D206)", features=items(tab)),
     dict(id="rc", name="Location recommender (DeepFM, log-error loss)", role="member, weight 1/4 in the candidate (D358); not in the submitted system", source="eda/202_rc_model.py itemfeat=1 loss=logl1 etaclip=3 mino=12 (D331–D339)",
          features=items(RCN) + [dict(name=c, block="item inputs (as the tree member, at the origin)", desc=desc(c)) for c in IF] + rc_emb),
     dict(id="seg", name="Segment correction", role="correction", source="eda/179x2_bias.py (D245)", features=[dict(name=n, block="segment", desc=d) for n, d in [("volume band", "fifth of the recent level"), ("age band", "under 3, 3–6, 6–12, 12+ months"), ("discount band", "the item's web discount")]]),
     dict(id="brk", name="Breakout correction", role="correction", source="eda/179b_breakout.py XSIG=1 (D247, D259)", features=items(SIG179))]
for m in M: m["n"] = len(m["features"])
json.dump(dict(models=M, note="Every input is known at the forecast origin. The recommender's discount input is the last discount at each location (the plan setting, with realised discounts, was a test only)."),
          open("data/feature_inventory.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print({m["id"]: m["n"] for m in M}); print("no description:", sorted({f["name"] for m in M for f in m["features"] if not f["desc"]})[:40])
