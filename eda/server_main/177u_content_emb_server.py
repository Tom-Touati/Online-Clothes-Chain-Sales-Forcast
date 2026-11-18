"""D171 (overnight M5 goal, direction 6), runs on the GPU server (~/edikted_main): product-photo and text embeddings per item.
Images: up to 3 product photos per item from the public feed (data/product_feed.parquet), resized to 512 px; OpenCLIP ViT-B-32
(laion2b_s34b_b79k) image embedding, mean over photos, unit length (512 dims).
Text: title + description + static tags (sale / clearance / discount / retailer / garage / size tags dropped: they describe today's
state and would leak into the backtest); sentence-transformers all-mpnet-base-v2 (768 dims) and the CLIP text tower (512 dims).
Writes data/content_emb.npz: items, img (n x 512), txt (n x 768), cliptxt (n x 512), has_img, has_txt."""
import os, io, json, re, time, numpy as np, pandas as pd, requests, torch
torch.backends.cudnn.enabled = False   # cuDNN sub-library fails to load on the server build; plain CUDA kernels are used
from PIL import Image
os.chdir(os.path.expanduser("~/edikted_main"))
F = pd.read_parquet("data/product_feed.parquet"); items = np.sort(pd.read_parquet("data/items.parquet").style_color.values)
F = F.set_index("style_color").reindex(items); n = len(items); dev = "cuda" if torch.cuda.is_available() else "cpu"
LEAK = re.compile(r"(discount|percent|sale|clearance|garage|size_|nordstrom|tillys|pacsun|macys|^__|bf|cyber|final|markdown|new_?in|restock|price)", re.I)
def clean_tags(t):
    if not isinstance(t, str): return ""
    keep = [x for x in t.split(",") if x and not LEAK.search(x)]
    return ", ".join(x.replace("_", " ").replace(".", " ") for x in keep)
texts = [" | ".join([str(F.title.iloc[i] or ""), str(F.description.iloc[i] or ""), clean_tags(F.tags.iloc[i])]) if isinstance(F.title.iloc[i], str) else "" for i in range(n)]
import open_clip
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev); tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
img = np.zeros((n, 512), np.float32); has_img = np.zeros(n, bool); S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (forecasting research)"
t0 = time.time()
for i in range(n):
    urls = json.loads(F.image_urls.iloc[i]) if isinstance(F.image_urls.iloc[i], str) else []
    ims = []
    for u in urls[:3]:
        try:
            r = S.get(u + ("&" if "?" in u else "?") + "width=512", timeout=30)
            if r.status_code == 200: ims.append(pre(Image.open(io.BytesIO(r.content)).convert("RGB")))
        except Exception: pass
    if ims:
        with torch.no_grad():
            e = model.encode_image(torch.stack(ims).to(dev)).float(); e = e / e.norm(dim=-1, keepdim=True)
        img[i] = e.mean(0).cpu().numpy(); img[i] /= max(np.linalg.norm(img[i]), 1e-9); has_img[i] = True
    if i % 100 == 0: print(i, "items", int(has_img.sum()), "with images", f"{time.time() - t0:.0f}s", flush=True)
with torch.no_grad():
    ct = []
    for b in range(0, n, 64):
        e = model.encode_text(tok([t[:300] for t in texts[b:b + 64]]).to(dev)).float(); ct.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
cliptxt = np.vstack(ct)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-mpnet-base-v2", device=dev); txt = st.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
has_txt = np.array([len(t) > 5 for t in texts])
np.savez_compressed("data/content_emb.npz", items=items, img=img, txt=txt.astype(np.float32), cliptxt=cliptxt, has_img=has_img, has_txt=has_txt)
print("saved: images", int(has_img.sum()), "texts", int(has_txt.sum()), f"{time.time() - t0:.0f}s")
