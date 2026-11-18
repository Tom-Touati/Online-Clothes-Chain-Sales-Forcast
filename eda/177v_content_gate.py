"""D180 (overnight M5 goal, direction 6): product photos and text on the cart co-occurrence gate (user rule: embeddings first
predict cart co-occurrence, against naive raw-feature baselines). Same gate as D164/D166 (eda/170, eda/172: held-out styles,
3 origins x 5 folds, past and future cart PPMI, style bootstrap).
Content (data/content_emb.npz, eda/177u on the GPU server): CLIP image embedding (mean of up to 3 product photos), mpnet text
embedding of title + description + static tags, CLIP text embedding; 1,408 of 1,500 items (the rest zero).
Models: v4 (D166), v6_img = v4 + image, v6_txt = v4 + text, v6_all = v4 + image + both texts, content = image + texts alone.
Naive baselines: raw product features, raw image cosine, raw text cosine.
Writes eda/content_gate_output.txt (stdout) and data/content_gate_results.csv."""
import os, sys
src = open("eda/172_style_emb_v4.py", encoding="utf-8").read(); _k = sys.argv; sys.argv = sys.argv[:1]
exec(src[:src.index("CFG4 = {")]); sys.argv = _k
CE = np.load("data/content_emb.npz", allow_pickle=True); ci_ = pd.Series(range(len(CE["items"])), index=CE["items"].astype(str)).reindex(items).values
IMG = CE["img"][ci_]; TXT = CE["txt"][ci_]; CTX = CE["cliptxt"][ci_]
CFG6 = {"v4": ("nce", ["prod", "reg", "temp4", "evt4"]), "v6_img": ("nce", ["prod", "reg", "temp4", "evt4", "img"]), "v6_txt": ("nce", ["prod", "reg", "temp4", "evt4", "txt"]),
        "v6_all": ("nce", ["prod", "reg", "temp4", "evt4", "img", "txt", "ctx"]), "content": ("nce", ["img", "txt", "ctx"])}
res = []; t0 = time.time()
for o in ORIG:
    REGION, TEMP, EVENTS = inputs(o); TEMP2, EVT2 = inputs2(o); TR = truth(o)
    B = {"prod": np.hstack([ATT, COLL, NAME, LLM]), "reg": REGION, "temp4": np.hstack([TEMP, TEMP2]), "evt4": np.hstack([EVENTS, EVT2]), "img": IMG, "txt": TXT, "ctx": CTX}
    Hs0 = {"raw product": rown(zs(PROD_RAW)), "raw image": rown(IMG + 1e-6), "raw text": rown(TXT + 1e-6)}
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    styles = np.unique(sty); fold = dict(zip(styles, np.random.default_rng(0).permutation(len(styles)) % 5)); f = np.array([fold[s] for s in sty])
    for k in range(NF):
        tr = np.where(alive & (f != k))[0]; te = np.where(alive & (f == k))[0]; Hs = dict(Hs0)
        for v, (obj, blk) in CFG6.items(): Hs[v] = train([B[b] for b in blk], tr, *TR["past"], obj)
        for tgt, (P_, ok_) in TR.items():
            for nm_, H in Hs.items():
                d = score(H, te, tr, P_, ok_); d["model"] = nm_; d["target"] = tgt; d["origin"] = o; d["fold"] = k; res.append(d)
        print(o, "fold", k, f"{time.time() - t0:.0f}s", flush=True)
R = pd.concat(res); R.to_csv("data/content_gate_results.csv", index=False)
print("\n== mean over held-out items: Spearman with cart PPMI | top-20 precision | chance")
print(R.groupby(["target", "model"])[["rho", "prec", "chance"]].mean().round(4).to_string())
rng = np.random.default_rng(0)
for ref in ["raw product", "raw image", "v4"]:
    print(f"\n== paired bootstrap over held-out styles, model minus {ref} (95% interval)")
    for tgt in ["past", "future"]:
        base = R[(R.target == tgt) & (R.model == ref)].groupby(["origin", "style"])[["rho", "prec"]].mean()
        for m in [x for x in R.model.unique() if x != ref]:
            x = R[(R.target == tgt) & (R.model == m)].groupby(["origin", "style"])[["rho", "prec"]].mean(); dlt = (x - base.reindex(x.index)).dropna()
            bs = np.array([dlt.values[rng.integers(0, len(dlt), len(dlt))].mean(0) for _ in range(1000)]); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            print(f"  {tgt:6s} {m:12s} rho {dlt.rho.mean():+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | prec {dlt.prec.mean():+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]")
