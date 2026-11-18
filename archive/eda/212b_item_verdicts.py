"""Fork 2: a verdict per item for the explorer (eda/212). Fork 2's reading, rule-based, stated as such on the page.
Benchmark for the projection / R3 (Jul-Oct 2026 against four times the recent monthly rate), from last year's same step (Jul-Oct 2025 against
4 x the Apr-Jun 2025 mean) of
  the product type, times the life-cycle factor measured last year (an item's step against its type's, volume-weighted, 2025-05/06/07
  origins: under 3 months old 1.15, 3-5 months 0.65, 6-11 0.78, 12+ 0.74; launches climb, existing items fade as the type grows by new
  launches), the item itself (if it sold through Apr-Jun 2025), and its 10 nearest neighbours in the multi-view embedding (median);
each times 0.94 for this year's slower growth (site plan: YoY easing from 1.20 in June to 1.12-1.15 in Jul-Oct; rules.py SITE PLAN).
Verdict: projection / benchmark below 0.80 = looks low, above 1.25 = looks high (momentum of the last 4 weeks against the 8 before can
strengthen or soften it); suggested total = halfway to the benchmark in logs, capped at +-25%. Adds 'verdict' to data/item_explorer.json."""
import json, numpy as np
J = json.load(open("data/item_explorer.json", encoding="utf-8")); GROW = 0.94
def life(a):
    if a is None: return 0.75
    return 1.15 if a < 3 else 0.65 if a < 6 else 0.78 if a < 12 else 0.74
fmt = lambda x: f"{x:,.0f}"; pct = lambda x: f"{(x - 1) * 100:+.0f}%"
n_low = n_high = 0
for it in J["items"]:
    c, p = it["checks"], it["proj"]; R3, F = p["R3"] or 0, p["forecast"]; age = c["age"]; L = life(age)
    parts = []
    if c["type"]: parts.append(("type", c["type"] * L * GROW, f"{it['type']} last year ×{c['type']:.2f}, × life-cycle {L:.2f} at {(age or 0):.0f} months, × 0.94 slower growth"))
    if c["own"] and age is not None and age >= 13: parts.append(("own", c["own"] * GROW, f"its own step last year ×{c['own']:.2f}, × 0.94"))
    if c["nb"]: parts.append(("nb", c["nb"] * GROW, f"its 10 nearest neighbours' median step last year ×{c['nb']:.2f}, × 0.94"))
    bench = float(np.exp(np.median([np.log(v) for _, v, _ in parts]))) if parts else None
    imp = F / R3 if R3 > 0 else None; mom = c["mom4"]
    launch = (age is not None and age < 4) or R3 < 0.15 * F
    v = dict(bench=round(bench, 2) if bench else None, parts=[dict(k=k, v=round(x, 2), how=h) for k, x, h in parts], launch=bool(launch))
    if imp is None or bench is None or launch:
        v.update(call="no call", tone="neutral", adj=None,
                 text=("A launch or near-launch: the recent rate (R3) covers too little selling to anchor a seasonal check, so the projection rests on the models' launch curves and neighbours. "
                       "I would not override it; I would watch the first four weeks of July and re-forecast (the month-1 update cut the error from 0.395 to 0.369 in the backtest)."))
    else:
        g = imp / bench; v["gap"] = round(g, 2)
        if g < 0.80:
            strong = mom is not None and mom >= 1.0
            half = 0.25 if (mom is not None and mom < 0.7) else 0.5   # cooling into June: go a quarter of the way
            adj = float(np.exp(np.clip(half * np.log(bench / imp), -np.log(1.25), np.log(1.25)))); v.update(call="looks low", tone="up", adj=round(F * adj))
            v["text"] = (f"The projection is ×{imp:.2f} the recent rate; comparable evidence points to about ×{bench:.2f}, so it sits {pct(g)} below. "
                         + (f"Sales were still rising into June (last 4 weeks ×{mom:.2f} the 8 before), which makes the gap more credible. " if strong else
                            (f"Sales were cooling into June (last 4 weeks ×{mom:.2f} the 8 before), which explains part of the caution. " if mom is not None else ""))
                         + f"I would lift it to about {fmt(F * adj)} (+{(adj - 1) * 100:.0f}%), {'halfway' if half == 0.5 else 'a quarter of the way'} to the benchmark.")
            n_low += 1
        elif g > 1.25:
            strong = mom is not None and mom < 0.85
            half = 0.25 if (mom is not None and mom > 1.3) else 0.5   # still accelerating into June: a quarter of the way
            adj = float(np.exp(np.clip(half * np.log(bench / imp), -np.log(1.25), np.log(1.25)))); v.update(call="looks high", tone="down", adj=round(F * adj))
            v["text"] = (f"The projection is ×{imp:.2f} the recent rate; comparable evidence points to about ×{bench:.2f}, so it sits {pct(g)} above. "
                         + (f"Sales were already slowing into June (last 4 weeks ×{mom:.2f} the 8 before), which supports trimming. " if strong else
                            (f"Recent weeks held up (×{mom:.2f}), so the models may be reading momentum the benchmark does not see. " if mom is not None else ""))
                         + f"I would trim it to about {fmt(F * adj)} ({(adj - 1) * 100:.0f}%), {'halfway' if half == 0.5 else 'a quarter of the way'} to the benchmark.")
            n_high += 1
        else:
            v.update(call="plausible", tone="ok", adj=None,
                     text=f"The projection is ×{imp:.2f} the recent rate and comparable evidence points to about ×{bench:.2f} ({pct(g)}). That is within the band I would accept, so I would keep it."
                          + (f" The last 4 weeks ran ×{mom:.2f} the 8 before." if mom is not None else ""))
    it["verdict"] = v
J["verdict_note"] = __doc__
json.dump(J, open("data/item_explorer.json", "w", encoding="utf-8"), ensure_ascii=False)
calls = [it["verdict"]["call"] for it in J["items"]]; print({k: calls.count(k) for k in set(calls)})
tot = sum(it["proj"]["total"] for it in J["items"]); adj = sum((it["verdict"]["adj"] or it["proj"]["total"]) for it in J["items"]); print("total", tot, "adjusted", adj, round(adj / tot - 1, 3))
for it in sorted(J["items"], key=lambda x: -x["proj"]["total"])[:25]:
    v = it["verdict"]; print(f"{it['id']:22s} {it['type']:12s} F {it['proj']['total']:6d} imp {it['checks']['implied']} bench {v['bench']} mom {it['checks']['mom4']} age {it['checks']['age']} -> {v['call']} {v.get('adj')}")
