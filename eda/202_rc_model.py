"""D331 (fork 2; user: "a kind of recommender system model, that checks the compatibility of times and items" and "the ideal model should
be able to get future discounts and a location and predict the sales, then sum over regions and channels"; "use the top tier method").
A location-level recommender: it predicts the units of item i at location l in month t, then sums over locations and the four months
of the window to the item total that is scored.
Rows      item x location x target month, for every origin o and horizon h = 1..4 (t = o + h), built from data/rc_panel.parquet.
Settings  honest  the item's locations are those that sold it in the last three months before the origin; the discount input is the
                  item's last discount there. Everything is known at the origin.
          plan    the item's locations in each month are those that sold it that month (the roll-out plan, read from the sales),
                  and the discount is the realised one (the discount plan). Known only afterwards: an upper bound on what a plan
                  from the business would give.
Models    deepfm  item (identity + attributes), location and calendar-month embeddings with factorisation-machine interactions
                  (item x location, item x month, location x month: the compatibility terms) plus an MLP on the numeric inputs;
                  Poisson loss on units with the item's recent units at the location as offset.
          gbm     LightGBM with a Poisson objective on the same inputs and offset (the strong tabular reference).
          avg     the mean of the two item totals.
Training  rows of origins whose window had closed (o' + 4 <= o). Settings fixed in advance; 2 seeds for deepfm.
Writes data/rc_recsys_preds.parquet (origin, style_color, y, f_<setting>_<model>), data/rc_pair_preds_2026.parquet (2026-06 per item x
location x month), data/rc_factors_2026.parquet (the compatibility terms for the 2026 items). Run on the CPU server (~/edikted_rc).
Usage: python eda/202_rc_model.py [origins=...] [settings=honest,plan] [threads=8]"""
import sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); TH = int(KW.get("threads", 8)); torch.set_num_threads(TH)
SETS = KW.get("settings", "honest,plan").split(","); SEEDS = int(KW.get("seeds", 2)); EP = int(KW.get("epochs", 4)); TAG = KW.get("tag", "")
PSPLIT = int(KW.get("psplit", 0)); ECLIP = float(KW.get("etaclip", 0)); MINO = int(KW.get("mino", 0)); NOLY = int(KW.get("noly", 0)); RHL = float(KW.get("rhl", 0)); WIN = int(KW.get("win", 0)); HD = float(KW.get("hdelta", 0.5)); DIM = int(KW.get("dim", 16)); LR = float(KW.get("lr", 3e-3)); WD = float(KW.get("wd", 1e-4)); HID = int(KW.get("hid", 0)); DROP = float(KW.get("drop", 0.1)); IDROP = float(KW.get("idrop", 0.3)); BS = int(KW.get("bs", 8192)); GL = int(KW.get("gleaves", 63)); GLR = float(KW.get("glr", 0.05)); GR = int(KW.get("grounds", 400)); GMD = int(KW.get("gmind", 100)); GFF = float(KW.get("gff", 0.8));   # D345 tuning options
   # D342: recency half-life (months), training window (months), Huber delta, embedding size
   # D340: drop the training origins o - 12 +- 1 (the last-year-removed gate, fork D315)
   # D339: prediction kept within exp(+-ECLIP) of the recent level; deepfm falls back to gbm below MINO closed training origins
LOSS = KW.get("loss", "poisson"); WPOW = float(KW.get("wpow", 0)); DEV = torch.device(KW.get("device", "cpu")); IFEAT = int(KW.get("itemfeat", 0))   # D333 options: loss, item-volume weight power, device, the members' item inputs
ORIG = KW["origins"].split(",") if "origins" in KW else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")] + ["2026-06"]
T = pd.read_parquet("data/fk_explain_table.parquet"); T["origin"] = T.origin.astype(str); P = pd.read_parquet("data/rc_panel.parquet"); L = pd.read_parquet("data/rc_locations.parquet")
MON = [str(p) for p in pd.period_range("2024-01", "2026-06", freq="M")]; mix = {m: k for k, m in enumerate(MON)}; NM = len(MON)
items = sorted(T.style_color.unique()); iix = {s: k for k, s in enumerate(items)}; locs = sorted(L["loc"]); lix = {s: k for k, s in enumerate(locs)}
P = P[P.style_color.isin(iix) & P.m.isin(mix)]; pairs = P[["style_color", "loc"]].drop_duplicates().reset_index(drop=True); pix = {(a, b): k for k, (a, b) in enumerate(zip(pairs.style_color, pairs["loc"]))}
NP = len(pairs); U = np.zeros((NP, NM), np.float32); DS = np.full((NP, NM), np.nan, np.float32); pk = np.array([pix[(a, b)] for a, b in zip(P.style_color, P["loc"])]); mk = P.m.map(mix).values
U[pk, mk] = P.units.values; DS[pk, mk] = P.disc.values; pi_ = pairs.style_color.map(iix).values; pl_ = pairs["loc"].map(lix).values
LOCU = np.zeros((len(locs), NM), np.float32); np.add.at(LOCU, (pl_[:, None].repeat(NM, 1), np.arange(NM)[None, :].repeat(NP, 0)), U)
ITU = np.zeros((len(items), NM), np.float32); np.add.at(ITU, (pi_[:, None].repeat(NM, 1), np.arange(NM)[None, :].repeat(NP, 0)), U)
first_pair = np.where(U.sum(1) > 0, (U > 0).argmax(1), NM); first_loc = np.array([mix[f] if f in mix else 0 for f in L.set_index("loc").reindex(locs).first_m]); KIND = {"web": 0, "tiktok": 1, "store": 2, "partner": 3}
kind = np.array([KIND[l.split(":")[0]] for l in locs]); lastD = np.full((NP, NM), np.nan, np.float32); cur = np.full(NP, np.nan, np.float32)
for k in range(NM): cur = np.where(np.isfinite(DS[:, k]), DS[:, k], cur); lastD[:, k] = cur
CAT = ["ptype"] + [c for c in T.columns if c.startswith("a_")]; NUMA = ["summer", "winter", "party", "statement", "staple", "formality", "warmth", "layering", "versatility", "festival", "halloween", "back_to_school", "fit_tightness", "coverage"]
A = T.drop_duplicates("style_color").set_index("style_color").reindex(items); XA = np.hstack([pd.get_dummies(A[CAT].astype(str)).values.astype(np.float32), ((A[NUMA] - A[NUMA].mean()) / (A[NUMA].std() + 1e-6)).fillna(0).values.astype(np.float32)])
ptype_code = A.ptype.astype("category").cat.codes.values
print(f"panel: {len(P)} rows, {NP} item x location pairs, {len(items)} items, {len(locs)} locations; months {MON[0]}..{MON[-1]}", flush=True)


def rows(o, setting):
    """the window rows of origin o: arrays of pair index, horizon, target month index and the inputs"""
    k = mix[o]; it = T[T.origin == o].style_color.map(iix).values; ok_item = np.zeros(len(items), bool); ok_item[it] = True; pr = np.where(ok_item[pi_])[0]
    out = []
    for h in (1, 2, 3, 4):
        t = k + h
        # D334 settings: storeplan (proposal #9: the stores an item is in are the realised ones, every other location as at the origin)
        # and partnerplan (proposal #2 bound: the partners that carry it are the realised ones, i.e. the order book)
        pmask = {"honest": np.zeros(len(pr), bool), "plan": np.ones(len(pr), bool), "storeplan": kind[pl_[pr]] == 2, "partnerplan": kind[pl_[pr]] == 3, "discplan": np.zeros(len(pr), bool), "biz": np.isin(kind[pl_[pr]], (2, 3))}[setting]
        recent = U[pr, max(k - 2, 0):k + 1].sum(1) > 0; fut = (U[pr, t] > 0) if t < NM else np.zeros(len(pr), bool)
        sel = pr[(recent | (fut & pmask))] if setting == "biz" else pr[(recent & ~pmask) | (fut & pmask)]   # D344 biz: keep every current location, ADD the stores and partners the business sent it to; never drop one
        if not len(sel): continue
        l3 = U[sel, max(k - 2, 0):k + 1].mean(1); it3 = ITU[pi_[sel], max(k - 2, 0):k + 1].mean(1); dl = lastD[sel, k]
        din = DS[sel, t] if (setting in ("plan", "discplan", "biz") and t < NM) else dl   # beyond the data (2026 origin) the plan falls back to the last discount
        if setting == "discplan": din = np.where(np.isfinite(din), din, dl)
        out.append(pd.DataFrame({"pair": sel, "item": pi_[sel], "loc": pl_[sel], "kind": kind[pl_[sel]], "h": h, "t": t, "moy": (int(MON[min(t, NM - 1)][5:]) if t < NM else (int(o[5:]) + h - 1) % 12 + 1) - 1,
                                 "l3": np.log1p(l3), "l1": np.log1p(U[sel, k]), "ly": np.log1p(U[sel, t - 12]) if t - 12 >= 0 and t - 12 <= k else np.full(len(sel), np.nan, np.float32),
                                 "pair_age": np.log1p(np.maximum(k - first_pair[sel], 0)), "loc_age": np.log1p(np.maximum(t - first_loc[pl_[sel]], 0)), "loc3": np.log1p(LOCU[pl_[sel], max(k - 2, 0):k + 1].mean(1)),
                                 "it3": np.log1p(it3), "share": l3 / np.maximum(it3, 1e-3), "disc_in": din, "disc_last": dl, "disc_d": din - dl, "ptype": ptype_code[pi_[sel]],
                                 "off": np.log(l3 + 0.5).astype(np.float32), "y": U[sel, t] if t < NM else np.full(len(sel), np.nan, np.float32), "origin": o}))
    return pd.concat(out, ignore_index=True) if out else None
NUM = ["h", "l3", "l1", "ly", "pair_age", "loc_age", "loc3", "it3", "share", "disc_in", "disc_last", "disc_d"]
BIZ = int(KW.get("biz", 0))
if BIZ:   # D344: the business's launch plan in the window, known ahead: new colours of the item's style and new items of its product type
    _fi = np.where(ITU.sum(1) > 0, (ITU > 0).argmax(1), NM); _sty = np.array([s_.split("_")[0] for s_ in items]); _rows0b = rows
    def rows(o, setting):
        R_ = _rows0b(o, setting)
        if R_ is None: return None
        k = mix[o]; inwin = (_fi > k) & (_fi <= k + 4); ns = pd.Series(inwin).groupby(_sty).transform("sum").values; nt = pd.Series(inwin).groupby(ptype_code).transform("sum").values
        return R_.assign(biz_newcol=np.log1p(ns[R_.item.values] - inwin[R_.item.values]), biz_newtype=np.log1p(nt[R_.item.values]), biz_newloc=((R_.l3.values == 0) & (R_.kind.values >= 2)).astype(np.float32))
    NUM = NUM + ["biz_newcol", "biz_newtype", "biz_newloc"]
LEAK = ("z", "ytot", "target", "dev_actual", "dT_real_from_org", "o_ord", "level", "R3")
IF = [c for c in T.columns if c not in LEAK and c not in ("origin", "style_color") and pd.api.types.is_numeric_dtype(T[c]) and not c.startswith("a_")] if IFEAT else []
if IFEAT:   # the members' item inputs at the origin (fk_explain_table), joined to every location row of the item
    TI = T.set_index(["origin", "style_color"])[IF]; _rows0 = rows
    def rows(o, setting):
        R_ = _rows0(o, setting)
        if R_ is None: return None
        X_ = TI.reindex(pd.MultiIndex.from_arrays([[o] * len(R_), [items[i] for i in R_.item]])).values.astype(np.float32); return pd.concat([R_, pd.DataFrame(X_, columns=["if_" + c for c in IF])], axis=1)
    NUM = NUM + ["if_" + c for c in IF]
print("loss", LOSS, "| weight power", WPOW, "| device", DEV, "| item inputs", len(IF), flush=True)
def lossf(eta, y, w):
    if LOSS == "poisson": l = torch.exp(eta) - y * eta
    elif LOSS.startswith("tweedie"): p = float(LOSS.split(":")[1]); l = -y * torch.exp(eta * (1 - p)) / (1 - p) + torch.exp(eta * (2 - p)) / (2 - p)
    elif LOSS == "logl1": l = (torch.log(torch.exp(eta) + 0.5) - torch.log(y + 0.5)).abs()
    elif LOSS.startswith("pin:"):   # D346: pinball loss on the log scale, quantile a (user: "all the models should output 0.25 quantile and 0.75 quantile")
        a_ = float(LOSS.split(":")[1]); r = torch.log(y + 0.5) - torch.log(torch.exp(eta) + 0.5); l = torch.maximum(a_ * r, (a_ - 1) * r)
    elif LOSS == "loghuber": l = nn.functional.huber_loss(torch.log(torch.exp(eta) + 0.5), torch.log(y + 0.5), reduction="none", delta=HD)
    elif LOSS == "soft20": r = torch.log((torch.exp(eta) + 1) / (y + 1)); l = 1 - torch.exp(-r ** 2 / (2 * 0.18 ** 2)) + 0.05 * r.abs()
    return (l * w).sum() / w.sum()
def wts(R_):
    w_ = np.ones(len(R_), np.float32) if WPOW == 0 else np.exp(-WPOW * R_.it3.values).astype(np.float32)
    if RHL and "_age" in R_: w_ = w_ * np.power(0.5, R_["_age"].values / RHL).astype(np.float32)   # D342: recency weight, age in months of the training origin
    return w_   # (1 + item units a month)^-WPOW


class DeepFM(nn.Module):
    def __init__(s, d=DIM):
        super().__init__(); s.ie = nn.Embedding(len(items), d); s.ia = nn.Linear(XA.shape[1], d); s.le = nn.Embedding(len(locs), d); s.me = nn.Embedding(12, d); s.ke = nn.Embedding(4, d)
        for e in (s.ie, s.le, s.me, s.ke): nn.init.normal_(e.weight, std=0.05)
        H1 = HID or (128 if IFEAT else 96); s.mlp = nn.Sequential(nn.Linear(4 * d + len(NUM) + len(NUM), H1), nn.ReLU(), nn.Dropout(DROP), nn.Linear(H1, 96), nn.ReLU(), nn.Dropout(0.1), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1)); s.XA = torch.tensor(XA)
    def parts(s, item, loc, moy, kd, drop=0.0):
        m = (torch.rand(len(item), 1, device=item.device) > drop).float() if s.training and drop > 0 else 1.0
        vi = s.ie(item) * m + s.ia(s.XA[item]); vl = s.le(loc) + s.ke(kd); vm = s.me(moy); return vi, vl, vm
    def forward(s, item, loc, moy, kd, x, xm, off, drop=0.0):
        vi, vl, vm = s.parts(item, loc, moy, kd, drop); fm = (vi * vl).sum(1) + (vi * vm).sum(1) + (vl * vm).sum(1)
        return off + fm + s.mlp(torch.cat([vi, vl, vm, s.ke(kd), x, xm], 1)).squeeze(1)


def tens(R, med):
    x = R[NUM].values.astype(np.float32); xm = np.isnan(x).astype(np.float32); x = np.where(np.isnan(x), med, x)
    return [torch.tensor(R[c].values, dtype=torch.long) for c in ("item", "loc", "moy", "kind")] + [torch.tensor(x), torch.tensor(xm), torch.tensor(R.off.values, dtype=torch.float32)]
def fit_deepfm(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed); med = np.nanmedian(tr[NUM].values.astype(np.float32), 0); med = np.where(np.isfinite(med), med, 0); sd = np.nanstd(tr[NUM].values.astype(np.float32), 0) + 1e-3
    trn = tr.copy(); ten = te.copy(); trn[NUM] = (trn[NUM] - med) / sd; ten[NUM] = (ten[NUM] - med) / sd; Xtr = [a.to(DEV) for a in tens(trn, 0.0)]; Xte = [a.to(DEV) for a in tens(ten, 0.0)]; y = torch.tensor(tr.y.values, dtype=torch.float32, device=DEV); W = torch.tensor(wts(tr), device=DEV)
    m = DeepFM().to(DEV); m.XA = m.XA.to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD); n = len(y); bs = BS
    for ep in range(EP):
        m.train(); perm = torch.randperm(n, device=DEV)
        for b in range(0, n, bs):
            ix = perm[b:b + bs]; eta = m(*[a[ix] for a in Xtr], drop=IDROP).clamp(-10, 12); loss = lossf(eta, y[ix], W[ix]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): eta = torch.cat([m(*[a[b:b + 65536] for a in Xte]) for b in range(0, len(te), 65536)]).clamp(-10, 12).cpu().numpy()
    if ECLIP: off = te.off.values; eta = np.clip(eta, off - ECLIP, off + ECLIP)
    return np.exp(eta), m
def fit_gbm(tr, te):
    F = NUM + ["loc", "kind", "moy", "ptype"]
    if LOSS in ("logl1", "soft20", "loghuber") or LOSS.startswith("pin:"):   # log-error losses: an L1 / Huber tree on log(units + 0.5), relative to the offset
        d = lgb.Dataset(tr[F], np.log(tr.y + 0.5) - tr.off, weight=wts(tr), categorical_feature=["loc", "kind", "moy", "ptype"], free_raw_data=False)
        b = lgb.train(dict(objective="quantile" if LOSS.startswith("pin:") else ("l1" if LOSS == "logl1" else "huber"), alpha=float(LOSS.split(":")[1]) if LOSS.startswith("pin:") else (HD if LOSS == "loghuber" else 0.3), learning_rate=GLR, num_leaves=GL, min_data_in_leaf=GMD, feature_fraction=GFF, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, num_threads=TH), d, GR)
        return np.maximum(np.exp(b.predict(te[F]) + te.off.values) - 0.5, 0)
    obj = dict(objective="tweedie", tweedie_variance_power=float(LOSS.split(":")[1])) if LOSS.startswith("tweedie") else dict(objective="poisson")
    d = lgb.Dataset(tr[F], tr.y, init_score=tr.off, weight=wts(tr), categorical_feature=["loc", "kind", "moy", "ptype"], free_raw_data=False)
    b = lgb.train(dict(**obj, learning_rate=GLR, num_leaves=GL, min_data_in_leaf=GMD, feature_fraction=GFF, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, num_threads=TH), d, GR)
    return np.exp(b.predict(te[F], raw_score=True) + te.off.values)


t0 = time.time(); ALL = {s: {} for s in SETS}; OUT = []; PAIR26 = []
AUG = int(KW.get("aug", 0)); GBMONLY = int(KW.get("gbmonly", 0))
if AUG:   # Fork 2 (user: "augment the known future as nan so it can handle situations where its not available"): needs biz=1; every training
    # origin appears twice, once with the business plan (biz rows, plan_known 1) and once without it (the locations as at the origin, plan
    # inputs NaN, plan_known 0); tested both ways: aug_plan (plan given) and aug_none (plan missing)
    SETS = [s for s in SETS if s not in ("aug_plan", "aug_none")]; SETS = list(dict.fromkeys(SETS + ["biz", "honest"])) + ["aug_plan", "aug_none"]; NUM = NUM + ["plan_known"]
    PLANC = ["biz_newcol", "biz_newtype", "biz_newloc", "disc_in", "disc_d"]
IMP = bool({"bizimp", "bizimpaug"} & set(SETS))
if IMP: SETS = list(dict.fromkeys([s for s in SETS if not s.startswith("bizimp")] + ["biz", "honest"] + [s for s in SETS if s.startswith("bizimp")]))
ROWS = {s: {o: rows(o, s) for o in sorted(T.origin.unique())} for s in SETS if not s.startswith("aug_") and not s.startswith("bizimp")}
if IMP:   # Fork 2 (user: "try a variant where we predict (impute) each of those features from the past features and months since start", on top
    # of augmenting). Needs settings=biz,bizimp biz=1 itemfeat=1. The plan-known tree (trained on biz rows) is applied with every plan input
    # imputed from what is known at the origin, by models fitted on closed past windows only:
    #   discount     LightGBM on past realised discounts (last discount, location, kind, month, horizon, item volume, months since start)
    #   launches     biz_newcol / biz_newtype: LightGBM on the style's and type's launches of the previous 4 months and of the same window a
    #                year before, month, months since start
    #   placement    every store / partner open at the origin that did not sell the item in the last 3 months is a candidate row; a classifier
    #                gives P(it sells there in month t); the candidate's units = P x the plan-known tree's units for that row (biz_newloc 1)
    #   the item's current locations are the honest rows with the imputed discount and launches.
    assert BIZ and IFEAT and "biz" in ROWS and "honest" in ROWS
    TI2 = T.set_index(["origin", "style_color"])[IF]; _sty_code = pd.Series(_sty).astype("category").cat.codes.values
    def launch_feats(k):
        inwin = (_fi > k) & (_fi <= k + 4); prev = (_fi > k - 4) & (_fi <= k); ly_ = (_fi > k - 12) & (_fi <= k - 8)
        g = lambda a, c: pd.Series(a.astype(float)).groupby(c).transform("sum").values
        X = np.stack([np.log1p(g(prev, _sty_code)), np.log1p(g(prev, ptype_code)), np.log1p(g(ly_, ptype_code)), np.full(len(items), (k % 12)), np.full(len(items), k)], 1)
        return X, np.log1p(g(inwin, _sty_code) - inwin), np.log1p(g(inwin, ptype_code))
    def cand_rows(o):
        k = mix[o]; it = np.array(sorted(set(T[T.origin == o].style_color.map(iix).values))); lc = np.where((kind >= 2) & (first_loc <= k))[0]; out = []
        I, Lc = np.repeat(it, len(lc)), np.tile(lc, len(it)); pp = np.array([pix.get((items[a], locs[b]), -1) for a, b in zip(I, Lc)]); ex = pp >= 0; pq = np.where(ex, pp, 0)
        rec = np.where(ex, U[pq, max(k - 2, 0):k + 1].sum(1), 0) > 0; I, Lc, pp, ex, pq = I[~rec], Lc[~rec], pp[~rec], ex[~rec], pq[~rec]
        for h in (1, 2, 3, 4):
            t = k + h; lyv = np.where(ex, np.log1p(U[pq, t - 12]), 0.0) if 0 <= t - 12 <= k else np.full(len(I), np.nan)
            out.append(pd.DataFrame({"pair": pp, "item": I, "loc": Lc, "kind": kind[Lc], "h": h, "t": t, "moy": (int(MON[min(t, NM - 1)][5:]) if t < NM else (int(o[5:]) + h - 1) % 12 + 1) - 1,
                                     "l3": 0.0, "l1": 0.0, "ly": lyv, "pair_age": np.where(ex, np.log1p(np.maximum(k - first_pair[pq], 0)), 0.0), "loc_age": np.log1p(np.maximum(t - first_loc[Lc], 0)),
                                     "loc3": np.log1p(LOCU[Lc, max(k - 2, 0):k + 1].mean(1)), "it3": np.log1p(ITU[I, max(k - 2, 0):k + 1].mean(1)), "share": 0.0,
                                     "disc_last": np.where(ex, lastD[pq, k], np.nan), "ptype": ptype_code[I], "off": np.float32(np.log(0.5)),
                                     "y": np.where(ex & (t < NM), U[pq, min(t, NM - 1)], 0.0), "origin": o, "biz_newloc": 1.0}))
        R_ = pd.concat(out, ignore_index=True); X_ = TI2.reindex(pd.MultiIndex.from_arrays([[o] * len(R_), [items[i] for i in R_.item]])).values.astype(np.float32)
        return pd.concat([R_, pd.DataFrame(X_, columns=["if_" + c for c in IF])], axis=1)
    CAND = {o: cand_rows(o) for o in ORIG + [oo for oo in sorted(T.origin.unique()) if oo not in ORIG]}; print("candidate rows", {o: len(r) for o, r in list(CAND.items())[:3]}, flush=True)
    DF = ["disc_last", "kind", "loc", "h", "moy", "it3", "l3", "loc3", "ptype", "tsince"]; PF = ["kind", "loc", "h", "moy", "it3", "loc3", "loc_age", "ly", "pair_age", "disc_last", "ptype", "tsince", "n_st", "n_pa"] + ["if_" + c for c in IF]
    def gb(obj, X, y, cats): return lgb.train(dict(objective=obj, learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=TH), lgb.Dataset(X, y, categorical_feature=cats), 300)
    def item_counts(o):   # stores and partners the item sold in over the last 3 months (known at the origin)
        r = ROWS["honest"][o]; return (r[r.h == 1].groupby("item").kind.apply(lambda s: (s == 2).sum()), r[r.h == 1].groupby("item").kind.apply(lambda s: (s == 3).sum())) if r is not None else (pd.Series(dtype=float), pd.Series(dtype=float))
    def addc(r, o):
        st, pa = item_counts(o); return r.assign(tsince=mix[o], n_st=r.item.map(st).fillna(0).values, n_pa=r.item.map(pa).fillna(0).values)
    def bizimp_predict(o, k, gbm_tr):
        past = [oo for oo in ROWS["biz"] if ROWS["biz"][oo] is not None and mix[oo] + 4 <= k and not (NOLY and abs(mix[oo] - (k - 12)) <= 1)]
        # discount imputer (realised discounts on past biz rows)
        Dtr = pd.concat([ROWS["biz"][oo].assign(tsince=mix[oo]) for oo in past], ignore_index=True); Dtr = Dtr[np.isfinite(Dtr.disc_in)]
        md = gb("l2", Dtr[DF], Dtr.disc_in, ["kind", "loc", "moy", "ptype"])
        # launch imputer
        LX = [launch_feats(mix[oo]) for oo in past]; LXa = np.vstack([a for a, _, _ in LX]); mc = gb("l2", LXa, np.concatenate([b for _, b, _ in LX]), []); mt = gb("l2", LXa, np.concatenate([c for _, _, c in LX]), [])
        Xk = launch_feats(k)[0]; ncol, ntyp = mc.predict(Xk), mt.predict(Xk)
        # placement classifier on past candidate rows
        Ctr = pd.concat([addc(CAND[oo], oo) for oo in past if oo in CAND], ignore_index=True); mp = gb("binary", Ctr[PF], (Ctr.y > 0).astype(int), ["kind", "loc", "moy", "ptype"])
        def fill(r):
            r = r.assign(tsince=k); r["disc_in"] = md.predict(r[DF]); r["disc_d"] = r.disc_in - r.disc_last; r["biz_newcol"] = ncol[r.item.values]; r["biz_newtype"] = ntyp[r.item.values]
            if "plan_known" in NUM: r["plan_known"] = np.float32(1.0)
            return r
        base = fill(ROWS["honest"][o].copy()); base["biz_newloc"] = 0.0; cand = fill(addc(CAND[o], o)); p = mp.predict(cand[PF])
        u = gbm_tr(pd.concat([base, cand[base.columns.intersection(cand.columns)]], ignore_index=True)); ub, uc = u[:len(base)], u[len(base):]
        return pd.concat([pd.DataFrame({"item": base.item.values, "p": ub}), pd.DataFrame({"item": cand.item.values, "p": p * uc})]), float(p.sum())
if AUG:
    for s in ("biz", "honest"):
        for o, r in ROWS[s].items():
            if r is not None: r["plan_known"] = np.float32(1.0 if s == "biz" else 0.0)
    ROWS["aug_plan"] = ROWS["biz"]; ROWS["aug_none"] = {o: (None if r is None else r.assign(**{c: np.nan for c in PLANC})) for o, r in ROWS["honest"].items()}
print(f"rows built in {time.time() - t0:.0f} s", {s: sum(len(v) for v in d.values() if v is not None) for s, d in ROWS.items()}, flush=True)
for o in ORIG:
    k = mix[o]; res = T[T.origin == o][["origin", "style_color"]].copy()
    for s in SETS:
        if s in ("bizimp", "bizimpaug"):   # the plan-known tree (biz rows; bizimpaug: the augmented plan / no-plan training) with imputed plan inputs
            if ROWS["honest"].get(o) is None: continue
            src = ("aug_plan", "aug_none") if s == "bizimpaug" else ("biz",)
            tr = pd.concat([r.assign(_age=k - mix[oo]) for s2 in src for oo, r in ROWS[s2].items() if r is not None and mix[oo] + 4 <= k and not (NOLY and abs(mix[oo] - (k - 12)) <= 1)], ignore_index=True)
            P_, nexp = bizimp_predict(o, k, lambda X: fit_gbm(tr, X)); agg = P_.groupby("item").p.sum()
            v = pd.DataFrame({"style_color": [items[i] for i in agg.index], f"f_{s}_gbm": agg.values}); v[f"f_{s}_deepfm"] = v[f"f_{s}_gbm"]; v[f"f_{s}_avg"] = v[f"f_{s}_gbm"]
            res = res.merge(v, on="style_color", how="left"); print(f"{o} {s}: train {len(tr)} rows | expected new placements {nexp:.0f} | {time.time() - t0:.0f} s", flush=True); continue
        te = ROWS[s][o]
        if te is None: continue
        if not any(r is not None and mix[oo] + 4 <= k for oo, r in ROWS[s if not s.startswith("aug_") else "aug_plan"].items()): continue   # no closed window yet (walk-forward from the start)
        tr = pd.concat([r.assign(_age=k - mix[oo]) for s2 in (("aug_plan", "aug_none") if s.startswith("aug_") else (s,)) for oo, r in ROWS[s2].items() if r is not None and mix[oo] + 4 <= k and not (NOLY and abs(mix[oo] - (k - 12)) <= 1) and not (WIN and k - mix[oo] > WIN)], ignore_index=True)
        if PSPLIT:   # proposal #2: the partner channel modelled apart from direct demand (web, TikTok, own stores)
            pd_, pg = np.zeros(len(te)), np.zeros(len(te))
            for is_p in (True, False):
                part = (te.kind.values == 3) == is_p
                if not part.any(): continue
                trp = tr[(tr.kind.values == 3) == is_p].reset_index(drop=True); tep = te[part].reset_index(drop=True)
                pd_[part] = np.mean([fit_deepfm(trp, tep, sd)[0] for sd in range(SEEDS)], 0); pg[part] = fit_gbm(trp, tep)
        else:
            pg = fit_gbm(tr, te); pd_ = pg.copy() if GBMONLY or sum(1 for oo in ROWS[s] if mix[oo] + 4 <= k) < MINO else np.mean([fit_deepfm(tr, te, sd)[0] for sd in range(SEEDS)], 0)
        for nm, p in (("deepfm", pd_), ("gbm", pg)):
            res = res.merge(pd.DataFrame({"style_color": [items[i] for i in te.item], "p": p}).groupby("style_color").p.sum().rename(f"f_{s}_{nm}").reset_index(), on="style_color", how="left")
        res[f"f_{s}_avg"] = (res[f"f_{s}_deepfm"] + res[f"f_{s}_gbm"]) / 2
        if o == "2026-06": PAIR26.append(te.assign(style_color=[items[i] for i in te.item], location=[locs[l] for l in te["loc"]], p_deepfm=pd_, p_gbm=pg, setting=s)[["setting", "style_color", "location", "h", "moy", "disc_in", "p_deepfm", "p_gbm"]])
        print(f"{o} {s}: train {len(tr)} rows | test {len(te)} rows | {time.time() - t0:.0f} s", flush=True)
    OUT.append(res)
R = pd.concat(OUT); M = pd.read_parquet("data/fk_err_master.parquet")[["origin", "style_color", "y"]]; R = R.merge(M, on=["origin", "style_color"], how="left"); R.to_parquet(f"data/rc_recsys_preds{TAG}.parquet", index=False)
if PAIR26: pd.concat(PAIR26).to_parquet(f"data/rc_pair_preds_2026{TAG}.parquet", index=False)
# the compatibility terms of a deepfm trained on every closed window, for the 2026 items: item x month-of-year and item x location
if "2026-06" in ORIG and "honest" in SETS and DEV.type == "cpu":
    tr = pd.concat([r for oo, r in ROWS["honest"].items() if r is not None and mix[oo] + 4 <= mix["2026-06"]], ignore_index=True); _, m = fit_deepfm(tr, ROWS["honest"]["2026-06"].head(10), 0)
    with torch.no_grad():
        it26 = torch.tensor(sorted(set(ROWS["honest"]["2026-06"].item)), dtype=torch.long); vi = m.ie(it26) + m.ia(m.XA[it26]); VM = m.me.weight; VL = m.le.weight + m.ke(torch.tensor(kind, dtype=torch.long))
        F1 = pd.DataFrame((vi @ VM.T).numpy(), columns=[f"m{j + 1}" for j in range(12)]); F2 = pd.DataFrame((vi @ VL.T).numpy(), columns=locs)
    pd.concat([pd.Series([items[i] for i in it26.numpy()], name="style_color"), F1, F2], axis=1).to_parquet(f"data/rc_factors_2026{TAG}.parquet", index=False)
print("saved", R.shape, f"{time.time() - t0:.0f} s")
