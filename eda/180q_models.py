"""D253: D198 tree member (fork's eda/193 'dyn': D159 features + dynamic co-occurrence neighbours) refitted at the 13 backtest origins,
median only (the BASE and FULL sets averaged, as the member), time-honest (training origins o' + 4 <= o; asserted).
Variants (argv; data/q_backtest_nx_<variant>.parquet, column q50 and, for 'base', in-sample WAPE / R3 per origin):
  base                 reproduces the D198 member's median
  +AGE +EV2 +TREND +CURVE +PEAK +WDAY +PLAN   one D252 block added (PLAN assumes the launch plan is known)
  all                  every block but PLAN; allplan = all + PLAN
  Memorisation / sanity:
  styleout             5 style-grouped folds: each test item is predicted by a model that never saw ANY row of its style
  nostatic             without the static item descriptors (LLM scores, tagger attributes, product type): nothing item-constant
  itemid               + the style as a categorical feature (explicit memory)
  shuffle              training target permuted within each training origin (should collapse to about the level)
  Volume / event models (on 'all' features):
  lowvol               separate model trained only on rows below the origin's 60th level percentile, used for those items
  bfsep                separate model trained only on rows whose window contains November (origins Jul-Oct), used at Jul-Oct origins
  bfw                  all, Black Friday rows weighted x3, + the item's and its type's last-year November uplift
  +PROMO, all2 (all + PROMO), lowvol2 / bfsep2 / bfw2: the same on all2 (D252b day-level promotion response)
  ec_ns_lyA / lyB / lyC / lyD = ec_ns with fewer last-year features (D351): none / the type's only / type's + item's own / all but the item's own
  ec_ns_aug = ec_ns trained on rows twice, as is and with the last-year features nulled (D298); q50 = normal view, q50_tta = average of the
              two views (log), q50_null = nulled view. Env LYDROP=1 drops training origins o - 12 +/- 1 (the fork's D315 gate), OSUF names the file.
  ec_ns_shr / ec_ns_shr25 = ec_ns with shrunk seasonal ratios (D297, #10b; run one per call, the shrink is applied to T)
  ec_ns_pl / w03ec_pl = + the D264 look-level launch plan (assumes the launch calendar is known)
  w03ec = ec_ns, volume-weighted rows ((4 x level + 1)^0.3), median + q65 (D260, newsvendor ladder member)
  bfwx = ec + last-year November uplift, Black Friday rows x3 (no PROMO); bfwx_ns without the static descriptors
  ec_ns = ec without the static descriptors (D254: they do not help); lowvol_ecns on it
  ec = EV2 + CURVE (the two blocks that helped alone), eca = + AGE, ecp = + PROMO; lowvol_ec / bfsep_ec / bfw_ec on ec
Writes data/q_backtest_nx_<variant>.parquet."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
T = pd.read_parquet("data/fk_explain_table.parquet"); META = json.load(open("data/fk_explain_meta.json"))
BASE0, FULL0, P = META["BASE"], META["FULL"], dict(META["PARAMS"]); P["n_jobs"] = int(__import__("os").environ.get("NJOBS", 8))   # D297: NJOBS caps threads (user: laptop must not lag)
DY = ["dy_nb_sea4", "dy_nb_lvl", "dy_nb_mom3", "dy_nb_yoy"]
T = T.merge(pd.read_parquet("data/fk_dyn_feats.parquet")[["origin", "style_color"] + DY], on=["origin", "style_color"], how="left")
NF = pd.read_parquet("data/nf2_feats.parquet"); BLK = json.load(open("data/nf2_meta.json")); BLK["EV2"] = [c for c in BLK["EV2"] if c != "c_pay"]
NF = NF.drop(columns=[c for c in ["age_m"] if c in T.columns]); T = T.merge(NF, on=["origin", "style_color"], how="left")
PL = pd.read_parquet("data/nf4_feats.parquet"); BLK["PLANLOOK"] = [c for c in PL.columns if c not in ("origin", "style_color")]; T = T.merge(PL, on=["origin", "style_color"], how="left")   # D264
PR = pd.read_parquet("data/nf3_feats.parquet"); BLK["PROMO"] = [c for c in PR.columns if c not in ("origin", "style_color")]; T = T.merge(PR, on=["origin", "style_color"], how="left")   # D252b
for c in META["CAT"]: T[c] = T[c].astype("category")
T["sty"] = T.style_color.str.split("_").str[0]; T["fold"] = T.sty.map(lambda s: sum(map(ord, s)) % 5)
T["bf"] = T.origin.str[-2:].isin(["07", "08", "09", "10"])
# last-year November uplift: own and type (Nov / mean Aug-Oct, a year before the target November); known at Jul-Oct origins
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
ptype = pd.read_parquet("data/items.parquet").set_index("style_color").product_type.reindex(M.index).fillna("na"); MT = M.groupby(ptype.values).sum()
def nov(o):
    y = int(o[:4]) - 1; nv, b = f"{y}-11", [f"{y}-08", f"{y}-09", f"{y}-10"]
    if nv not in M.columns or b[0] not in M.columns: return None, None
    own = np.log((M[nv] + 5) / (M[b].mean(1) + 5)).where(M[b].sum(1) > 30); ty = np.log((MT[nv] + 5) / (MT[b].mean(1) + 5)); return own, ty
T["ly_nov_own"] = np.nan; T["ly_nov_type"] = np.nan
for o in T.origin.unique():
    if o[-2:] not in ("07", "08", "09", "10"): continue
    own, ty = nov(o); m = (T.origin == o).values
    if own is None: continue
    T.loc[m, "ly_nov_own"] = own.reindex(T.style_color[m]).values; T.loc[m, "ly_nov_type"] = ty.reindex(T.ptype[m].astype(str)).values
STATIC = [c for c in FULL0 if c.startswith("a_") or c == "ptype"] + ["warmth", "coverage", "formality", "fit_tightness", "statement", "staple", "summer", "winter",
          "back_to_school", "party", "halloween", "festival", "layering", "versatility"]
ALL = sum([BLK[k] for k in ["AGE", "EV2", "TREND", "CURVE", "PEAK", "WDAY"]], [])
def sets(var):
    add, drop = [], []
    if var.startswith("+"): add = BLK[var[1:]]
    elif var in ("all", "lowvol", "bfsep"): add = ALL
    elif var == "allplan": add = ALL + BLK["PLAN"]
    elif var == "bfw": add = ALL + ["ly_nov_own", "ly_nov_type"]
    elif var in ("all2", "lowvol2", "bfsep2"): add = ALL + BLK["PROMO"]
    elif var in ("ec", "lowvol_ec", "bfsep_ec"): add = BLK["EV2"] + BLK["CURVE"]
    elif var == "eca": add = BLK["EV2"] + BLK["CURVE"] + BLK["AGE"]
    elif var in ("ec_ns", "lowvol_ecns", "w03ec"): add = BLK["EV2"] + BLK["CURVE"]; drop = STATIC
    elif var.startswith("ec_ns_ly"):   # D351 (user: "shrink down the number of features connecting to last year"): fewer last-year features
        LYALL = ["own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "cp_nb_sea4", "dy_nb_sea4", "dy_nb_yoy"]
        KEEP = {"ec_ns_lyA": [], "ec_ns_lyB": ["lfl_type4"], "ec_ns_lyC": ["lfl_type4", "own_sea4"], "ec_ns_lyD": [c for c in LYALL if c != "own_sea4"]}[var]
        add = BLK["EV2"] + BLK["CURVE"]; drop = STATIC + [c for c in LYALL if c not in KEEP]
    elif var in ("ec_ns_aug",): add = BLK["EV2"] + BLK["CURVE"]; drop = STATIC   # D298: last-year nulling augmentation, see the loop
    elif var in ("ec_ns_shr", "ec_ns_shr25"): add = BLK["EV2"] + BLK["CURVE"]; drop = STATIC   # D297: shrunk seasonality, see below
    elif var in ("ec_ns_pl", "w03ec_pl"): add = BLK["EV2"] + BLK["CURVE"] + BLK["PLANLOOK"]; drop = STATIC
    elif var == "ecp": add = BLK["EV2"] + BLK["CURVE"] + BLK["PROMO"]
    elif var == "bfw_ec": add = BLK["EV2"] + BLK["CURVE"] + BLK["PROMO"] + ["ly_nov_own", "ly_nov_type"]
    elif var in ("bfwx", "bfwx_ns"): add = BLK["EV2"] + BLK["CURVE"] + ["ly_nov_own", "ly_nov_type"]; drop = STATIC if var.endswith("_ns") else []
    elif var == "bfw2": add = ALL + BLK["PROMO"] + ["ly_nov_own", "ly_nov_type"]
    elif var == "nostatic": drop = STATIC
    elif var == "itemid": add = ["sty_c"]
    return [c for c in BASE0 + DY if c not in drop] + add, [c for c in FULL0 + DY if c not in drop] + add
T["sty_c"] = T.sty.astype("category")
# D297 (#10b): partial pooling of the seasonal ratios (a multi-year average is impossible in the backtest: the data start in Jan 2024, so
# every backtest origin has one prior year). ec_ns_shr: lfl_type4 -> (lfl_type4 + lfl_macro4) / 2 and own_sea4 -> (own_sea4 + lfl_type4) / 2;
# ec_ns_shr25: the same at weight 0.25 on the pooled value. Applied to train and test rows alike (features known at each origin).
import os as _os
_SHR = {"ec_ns_shr": 0.5, "ec_ns_shr25": 0.25}
for _v in sys.argv[1:]:
    if _v in _SHR:
        _w = _SHR[_v]; _lt = T.lfl_type4.copy()
        T["lfl_type4"] = (1 - _w) * _lt + _w * T.lfl_macro4.fillna(_lt)
        T["own_sea4"] = np.where(T.own_sea4.notna(), (1 - _w) * T.own_sea4 + _w * _lt.fillna(0), T.own_sea4)
def fit(tr, te, cols, w=None, y="z"):
    return lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(tr[cols], tr[y], sample_weight=w).predict(te[cols])
ORIG = [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")]
import os; ORIG = os.environ["ORIGS"].split(",") if os.environ.get("ORIGS") else ORIG; OSUF = os.environ.get("OSUF", "")   # D262: e.g. ORIGS=2026-06 OSUF=_2026 for the deliverable
for var in sys.argv[1:]:
    B_, F_ = sets(var); out, t0, ins = [], time.time(), []
    for o in ORIG:
        oo = pd.Period(o).ordinal; tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()].copy(); te = T[T.origin == o].copy()
        if os.environ.get("LYDROP") == "1": tr = tr[~tr.o_ord.between(oo - 13, oo - 11)]   # D298: the last-year-removed gate (fork D315: origins o - 12 +/- 1 dropped)
        assert (tr.o_ord + 4 <= oo).all() and tr.ytot.notna().all() and len(te) > 0
        z = np.zeros(len(te))
        if var == "styleout":
            for k in range(5):
                a, b = tr[tr.fold != k], te.fold.values == k
                if b.any(): z[b] = (fit(a, te[b], B_) + fit(a, te[b], F_)) / 2
        elif var == "shuffle":
            tr["zs"] = tr.groupby("origin").z.transform(lambda s: s.sample(frac=1, random_state=0).values); z = (fit(tr, te, B_, y="zs") + fit(tr, te, F_, y="zs")) / 2
        elif var in ("lowvol", "lowvol2", "lowvol_ec", "lowvol_ecns"):
            thr_tr = tr.groupby("origin")["level"].transform(lambda s: s.quantile(0.6)); thr_te = te.level.quantile(0.6)
            lo = tr[tr.level <= thr_tr]; z = (fit(tr, te, B_) + fit(tr, te, F_)) / 2; b = te.level.values <= thr_te
            z[b] = (fit(lo, te[b], B_) + fit(lo, te[b], F_)) / 2
        elif var in ("bfsep", "bfsep2", "bfsep_ec"):
            z = (fit(tr, te, B_) + fit(tr, te, F_)) / 2
            if o[-2:] in ("07", "08", "09", "10"):
                bt = tr[tr.bf]
                if bt.origin.nunique() >= 2: z = (fit(bt, te, B_) + fit(bt, te, F_)) / 2
        elif var == "ec_ns_aug":   # D298 (user: "augment by nulling last year's features, then average those 2 augmented views, or not")
            LYF = [c for c in ["own_sea4", "lfl_type4", "lfl_macro4", "lfl_occ4", "cp_nb_sea4", "dy_nb_sea4", "dy_nb_yoy"] if c in T.columns]
            trN = tr.copy(); trN[LYF] = np.nan; trA = pd.concat([tr, trN], ignore_index=True); teN = te.copy(); teN[LYF] = np.nan
            mB = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(trA[B_], trA.z); mF = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(trA[F_], trA.z)
            z = (mB.predict(te[B_]) + mF.predict(te[F_])) / 2; zN = (mB.predict(teN[B_]) + mF.predict(teN[F_])) / 2
            te = te.assign(_q50_tta=np.clip((4 * te.level.values + 1) * np.exp((z + zN) / 2) - 1, 0, None), _q50_null=np.clip((4 * te.level.values + 1) * np.exp(zN) - 1, 0, None))
        elif var == "w03ec_pl":
            w = (4 * tr.level.values + 1) ** 0.3; z = (fit(tr, te, B_, w) + fit(tr, te, F_, w)) / 2
        elif var == "w03ec":   # D260: volume-weighted (as the fork's w03: (4 x level + 1)^0.3), medians and q65 for the newsvendor ladder
            w = (4 * tr.level.values + 1) ** 0.3; z = (fit(tr, te, B_, w) + fit(tr, te, F_, w)) / 2
            P65 = dict(P); z65 = (lgb.LGBMRegressor(objective="quantile", alpha=0.65, **P65).fit(tr[B_], tr.z, sample_weight=w).predict(te[B_]) + lgb.LGBMRegressor(objective="quantile", alpha=0.65, **P65).fit(tr[F_], tr.z, sample_weight=w).predict(te[F_])) / 2
            te = te.assign(_q65=np.clip((4 * te.level.values + 1) * np.exp(np.maximum(z65, z)) - 1, 0, None))
        elif var in ("bfw", "bfw2", "bfw_ec", "bfwx", "bfwx_ns"):
            w = np.where(tr.bf, 3.0, 1.0); z = (fit(tr, te, B_, w) + fit(tr, te, F_, w)) / 2
        else:
            z = (fit(tr, te, B_) + fit(tr, te, F_)) / 2
            if var == "base":
                zi = (fit(tr, tr, B_) + fit(tr, tr, F_)) / 2; pi = (4 * tr.level + 1) * np.exp(zi) - 1
                ins.append(dict(origin=o, insample_wape_r3=(pi - tr.ytot).abs().sum() / (tr.R3 - tr.ytot).abs().sum(), test_wape_r3=(((4 * te.level + 1) * np.exp(z) - 1) - te.ytot).abs().sum() / (te.R3 - te.ytot).abs().sum()))
        R = te[["origin", "style_color", "level", "age_m", "ytot"]].copy(); R["q50"] = np.clip((4 * te.level.values + 1) * np.exp(z) - 1, 0, None)
        if "_q65" in te: R["q65"] = te._q65.values
        for _c in ("_q50_tta", "_q50_null"):
            if _c in te: R[_c[1:]] = te[_c].values
        R["seen_style"] = te.sty.isin(set(tr.sty)).values; out.append(R)
    pd.concat(out).to_parquet(f"data/q_backtest_nx_{var.replace('+', 'p_')}{OSUF}.parquet"); print(var, len(F_), "features (full set),", round(time.time() - t0), "s", flush=True)
    if ins: print(pd.DataFrame(ins).round(3).to_string(index=False))
