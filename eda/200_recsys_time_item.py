"""D330 (fork 2; user: "i would really like to see a kind of recommender system model, that checks the compatibility of times and
items"). A two-tower recommender for item x time compatibility, in the members' own target space.

What is predicted   z = log((4-month total + 1) / (4 x level + 1)), the item's lift over its recent level in the window after the
                    origin (the members' target, data/fk_explain_table.parquet).
The "time"          the calendar window: the origin's month (12 positions) plus what is known about that window ahead of time
                    (16 event flags, month as sine / cosine, the normal temperature of the window and its change from the origin).
The "item"          its identity (a learned vector, as a user or item id in a recommender) plus its attributes (product type, seven
                    tagged attributes, 14 scores such as summer / winter / party / warmth), so a new item gets a vector from what it is.
Score               z_hat = mu + item bias + window bias + context(age, level) + <item vector, time vector>.
                    The dot product is the compatibility of the item with the time; everything else does not depend on the pairing.
Training            at each origin only rows whose window had closed (o' + 4 <= o); Huber loss on z, every item counted equally;
                    identity vectors are dropped at random during training and penalised, so the model leans on attributes when an
                    item has little history; 3 seeds averaged. Settings fixed in advance (K = 8), not tuned on the scored origins.
Variants            bias     no item x time term at all (window, item and context biases only)
                    content  attributes x time (no identity vectors): a content-based recommender
                    collab   identity x time (no attributes): pure collaborative filtering
                    hybrid   both (the recommender)
Writes data/rc_recsys_preds.parquet (origin, style_color, z_<variant>, c_<variant> = the compatibility term, f_<variant> = forecast)
for the 13 backtest origins and 2026-06, and data/rc_recsys_map.parquet (2026 targets x 12 window positions, hybrid, trained on
every closed window). stdout -> eda/recsys_time_item_output.txt. Usage: python eda/200_recsys_time_item.py [origins=2025-02,...] [seeds=3]"""
import sys, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, warnings, openpyxl
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); SEEDS = int(KW.get("seeds", 3)); K = int(KW.get("k", 8)); STEPS = int(KW.get("steps", 400)); TAG = KW.get("tag", "")
T = pd.read_parquet("data/fk_explain_table.parquet").reset_index(drop=True); T["origin"] = T.origin.astype(str); ALLO = sorted(T.origin.unique())
ORIG = KW["origins"].split(",") if "origins" in KW else [str(p) for p in pd.period_range("2025-02", "2026-02", freq="M")] + ["2026-06"]
items = sorted(T.style_color.unique()); iix = {s: k for k, s in enumerate(items)}; T["ii"] = T.style_color.map(iix); NI = len(items)
CAT = ["ptype"] + [c for c in T.columns if c.startswith("a_")]; NUM = ["summer", "winter", "party", "statement", "staple", "formality", "warmth", "layering", "versatility", "festival", "halloween", "back_to_school", "fit_tightness", "coverage"]
A = T.drop_duplicates("style_color").set_index("style_color").reindex(items); XA = pd.get_dummies(A[CAT].astype(str)).values.astype(np.float32); XN = A[NUM].values.astype(np.float32); XN = (XN - XN.mean(0)) / (XN.std(0) + 1e-6); XI = np.hstack([XA, XN])
EV = [c for c in T.columns if c.startswith("ev_") and c.endswith("_t")]; O = T.groupby("origin").agg({**{c: "first" for c in EV}, "tmonth": "first", "t_tgt_norm": "mean", "dT_norm": "mean", "o_ord": "first"}).reindex(ALLO)
for c in ("t_tgt_norm", "dT_norm"):   # the last origins have no closed window yet: the normal temperature of a calendar window repeats, so take it from a year earlier
    for o in ALLO:
        if not np.isfinite(O.loc[o, c]): O.loc[o, c] = O.loc[str(pd.Period(o, "M") - 12), c]
O["msin"] = np.sin(2 * np.pi * O.tmonth / 12); O["mcos"] = np.cos(2 * np.pi * O.tmonth / 12); TC = EV + ["msin", "mcos", "t_tgt_norm", "dT_norm"]
CT = O[TC].values.astype(np.float32); CT = (CT - CT.mean(0)) / (CT.std(0) + 1e-6); oix = {o: k for k, o in enumerate(ALLO)}; T["oi"] = T.origin.map(oix); T["w"] = (T.tmonth.astype(int) - 1) % 12
T["cx1"] = np.log(T.age_m.clip(lower=0.5)); T["cx2"] = np.log1p(T.level); CX = T[["cx1", "cx2"]].values.astype(np.float32); CX = (CX - CX.mean(0)) / CX.std(0); Z = T.z.clip(-4, 4).values.astype(np.float32)
tt = torch.tensor; XI_t, CT_t = tt(XI), tt(CT)


class TwoTower(nn.Module):
    def __init__(s, use_id, use_attr, use_inter):
        super().__init__(); s.use_id, s.use_attr, s.use_inter = use_id, use_attr, use_inter
        s.ei = nn.Embedding(NI, K); s.bi = nn.Embedding(NI, 1); nn.init.zeros_(s.ei.weight); nn.init.zeros_(s.bi.weight)
        s.ax = nn.Sequential(nn.Linear(XI.shape[1], 32), nn.ReLU(), nn.Linear(32, K)); s.ew = nn.Embedding(12, K); s.bw = nn.Embedding(12, 1); nn.init.normal_(s.ew.weight, std=0.1); nn.init.zeros_(s.bw.weight)
        s.ct = nn.Sequential(nn.Linear(CT.shape[1], 16), nn.ReLU(), nn.Linear(16, K)); s.cx = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)); s.mu = nn.Parameter(torch.zeros(1))
    def item_vec(s, i, known, drop=0.0):
        m = known.float().unsqueeze(1)
        if s.training and drop > 0: m = m * (torch.rand(len(i), 1) > drop).float()
        u = torch.zeros(len(i), K)
        if s.use_id: u = u + s.ei(i) * m
        if s.use_attr: u = u + s.ax(XI_t[i])
        return u, m
    def time_vec(s, w, oi): return s.ew(w) + s.ct(CT_t[oi])
    def forward(s, i, w, oi, cx, known, drop=0.0):
        u, m = s.item_vec(i, known, drop); c = (u * s.time_vec(w, oi)).sum(1) if s.use_inter else torch.zeros(len(i))
        return s.mu + (s.bi(i) * m).squeeze(1) + s.bw(w).squeeze(1) + s.cx(cx).squeeze(1) + c, c


VAR = {"bias": (True, False, False), "content": (False, True, True), "collab": (True, False, True), "hybrid": (True, True, True)}
def fit_predict(tr, te, var, seed, extra=None):
    torch.manual_seed(seed); np.random.seed(seed); m = TwoTower(*VAR[var]); opt = torch.optim.Adam(m.parameters(), lr=0.01)
    i, w, oi, cx, z = tt(tr.ii.values), tt(tr.w.values), tt(tr.oi.values), tt(CX[tr.index.values]), tt(Z[tr.index.values]); known_tr = torch.ones(len(tr), dtype=torch.bool); seen = np.zeros(NI, bool); seen[tr.ii.unique()] = True
    for step in range(STEPS):
        m.train(); opt.zero_grad(); p, _ = m(i, w, oi, cx, known_tr, drop=0.3); loss = nn.functional.huber_loss(p, z, delta=1.0) + 0.02 * (m.ei(i) ** 2).sum(1).mean() + 0.05 * (m.bi(i) ** 2).mean(); loss.backward(); opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for q in [te] + (extra or []):
            p, c = m(tt(q.ii.values), tt(q.w.values), tt(q.oi.values), tt(CX[q.cxi.values] if "cxi" in q else CX[q.index.values]), tt(seen[q.ii.values])); out.append((p.numpy(), c.numpy()))
    return out, float(loss)


res = []; t0 = time.time(); MAP = None
for o in ORIG:
    oo = int(T.o_ord[T.origin == o].iloc[0]); tr = T[(T.o_ord + 4 <= oo) & T.z.notna()]; te = T[T.origin == o]; R = te[["origin", "style_color", "level", "ytot"]].copy(); extra = None
    if o == "2026-06":   # the compatibility map: every 2026 target at each of the 12 window positions, with its own context as of June 2026
        ws = openpyxl.load_workbook("Edikted_Forecasting_Exercise_filled.xlsx", data_only=True)["Forecast"]; tg = [r[1] for r in ws.iter_rows(min_row=2, values_only=True) if r[1]]; base = te[te.style_color.isin(tg)]
        lastw = {int(O.tmonth[x] - 1) % 12: x for x in ALLO}; extra = []
        for wpos in range(12):
            q = base[["style_color", "ii"]].copy(); q["w"] = wpos; q["oi"] = oix[lastw[wpos]]; q["cxi"] = base.index.values; extra.append(q.reset_index(drop=True))
    for var in VAR:
        zs, cs, ex = [], [], []
        for sd in range(SEEDS):
            out, ls = fit_predict(tr, te, var, sd, extra if var == "hybrid" else None); zs.append(out[0][0]); cs.append(out[0][1])
            if extra and var == "hybrid": ex.append(np.stack([np.stack(x) for x in out[1:]]))   # 12 x 2 x n
        R["z_" + var] = np.mean(zs, 0); R["c_" + var] = np.mean(cs, 0); R["f_" + var] = np.maximum((4 * R.level.values + 1) * np.exp(R["z_" + var].values) - 1, 0)
        if ex:
            e = np.mean(ex, 0); MAP = pd.concat([pd.DataFrame({"style_color": extra[k].style_color, "wpos": k, "start_month": (k + 1) % 12 + 1, "z": e[k, 0], "c": e[k, 1]}) for k in range(12)])
    R["seen"] = te.ii.isin(tr.ii.unique()).values; res.append(R); print(f"{o}: train {len(tr)} rows / {tr.ii.nunique()} items / {tr.origin.nunique()} origins | test {len(te)} | seen {R.seen.mean():.2f} | {time.time() - t0:.0f} s", flush=True)
P = pd.concat(res); P.to_parquet(f"data/rc_recsys_preds{TAG}.parquet", index=False)
if MAP is not None: MAP.to_parquet(f"data/rc_recsys_map{TAG}.parquet", index=False)
print("saved", P.shape, None if MAP is None else MAP.shape)
