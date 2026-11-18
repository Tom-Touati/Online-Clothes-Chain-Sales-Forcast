"""D168 (fork request, for its 5-nearest-neighbour features): export the v4 content style embedding (D166, eda/172) at every origin
2024-03 .. 2026-06. At each origin the model is trained on all items alive in the last 12 months, using carts and sales up to that
origin only (no held-out folds: the gate D164/D166 already measured generalisation), so features built from it are time-honest.
Inputs: product (attributes, collection, name words, LLM scores), region, temperature (TEMP + TEMP2), events (EVENTS + EVT2).
Objective: InfoNCE on cart pairs + MSE to the PPMI-SVD cosine (v3/v4). 32 dims, unit length; every one of the 1,500 items gets a
vector (items without carts or sales sit in the space through their inputs).
Writes data/style_emb_v4.npz: items (style_color, the row order) and one float16 array (1500 x 32) per origin key 'YYYY-MM'."""
import os, sys
src = open("eda/172_style_emb_v4.py", encoding="utf-8").read()
_keep = sys.argv; sys.argv = sys.argv[:1]
exec(src[:src.index("CFG4 = {")])
OUTV = {}; t0 = time.time()
for o in [str(p) for p in pd.period_range("2024-03", "2026-06", freq="M")]:
    REGION, TEMP, EVENTS = inputs(o); TEMP2, EVT2 = inputs2(o)
    past, rp = ppmi(sum(CM[m] for m in CM if Pm(o, -11) <= m <= o)); okp = rp >= 20
    B = [np.hstack([ATT, COLL, NAME, LLM]), REGION, np.hstack([TEMP, TEMP2]), np.hstack([EVENTS, EVT2])]
    alive = M.reindex(columns=[Pm(o, -k) for k in range(12)]).fillna(0).values.sum(1) > 0
    tr = np.where(alive)[0] if okp[alive].sum() >= 50 else np.arange(n)
    if okp[tr].sum() < 50: print(o, "too few cart items, skipped", flush=True); continue
    OUTV[o] = train(B, tr, past, okp, "nce").astype(np.float16)
    print(o, f"items with carts {int(okp.sum())} | {time.time() - t0:.0f}s", flush=True)
np.savez_compressed("data/style_emb_v4.npz", items=items, **OUTV); print("saved data/style_emb_v4.npz with", len(OUTV), "origins")
