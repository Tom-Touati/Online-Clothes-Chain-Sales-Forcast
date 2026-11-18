"""D170 (overnight M5 goal, direction 6): Edikted's public Shopify product feed (https://edikted.com/products.json, paged, 250 per
page): title, description (body_html), product type, tags, variant SKUs and image URLs. Matched to our 1,500 items by handle
(= style_color in lower case), falling back to variant SKUs from Dataset.parquet. Only the public catalogue is read; nothing is sent.
Writes data/product_feed.parquet (one row per matched item: style_color, title, description text, tags, n_images, image_urls json)
and prints coverage."""
import sys, re, json, time, html, requests, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = {"User-Agent": "Mozilla/5.0 (forecasting research)"}
prods, page = [], 1
while True:
    for a in range(5):
        try:
            r = requests.get("https://edikted.com/products.json", params={"limit": 250, "page": page}, headers=H, timeout=60)
            if r.status_code == 200: break
        except requests.exceptions.RequestException: pass
        time.sleep(5 * (a + 1))
    js = r.json().get("products", [])
    if not js: break
    prods += js; print("page", page, len(prods), flush=True); page += 1; time.sleep(1.0)
items = pd.read_parquet("data/items.parquet").style_color.values; low = {s.lower(): s for s in items}
sk = pd.read_parquet("Dataset.parquet", columns=["style_color", "sku"]).drop_duplicates()
sku2 = dict(zip(sk.sku.astype(str), sk.style_color))
rows = []
for p in prods:
    sc = low.get(p["handle"].lower())
    if sc is None:
        hits = {sku2.get(str(v.get("sku"))) for v in p["variants"]} - {None}
        sc = next(iter(hits)) if len(hits) == 1 else None
    if sc is None: continue
    txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p.get("body_html") or ""))).strip()
    rows.append({"style_color": sc, "title": p["title"], "description": txt, "shop_type": p["product_type"], "tags": ",".join(p["tags"]) if isinstance(p["tags"], list) else p["tags"],
                 "published_at": p.get("published_at"), "n_images": len(p["images"]), "image_urls": json.dumps([i["src"] for i in p["images"]][:6])})
F = pd.DataFrame(rows).drop_duplicates("style_color"); F.to_parquet("data/product_feed.parquet", index=False)
print("feed products", len(prods), "| matched items", len(F), "of", len(items), "| with images", int((F.n_images > 0).sum()), "| with description", int((F.description.str.len() > 20).sum()))
print("example description:", F.description.iloc[0][:300]); print("example tags:", F.tags.iloc[0][:300])
