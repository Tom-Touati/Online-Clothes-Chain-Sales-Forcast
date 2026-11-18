"""Is the 'same-mix benchmark' itself reliable? Apply its method to June 2025: prior-year (2024) July/June ratios by group
(item age bucket; and age x macro x peak position), weighted by the June 2025 mix, then compare with the actual July 2025 / June 2025."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
src = open("eda/44_mix_matched_july.py", encoding="utf-8").read(); src = src[:src.index("age25 =")]; ns = {}; exec(src, ns)
M, cols, start, it, MAC = ns["M"], ns["cols"], ns["start"], ns["it"], ns["MAC"]
def frame(y):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(M.index)).dt.days / 30.44
    ids = M.index[(M[f"{y}-06"] >= 50).values & (age >= 1).values]; a = age[ids]
    k = cols.index(f"{y}-06"); pos = M.loc[ids, f"{y}-06"] / M.loc[ids, cols[:k + 1]].max(axis=1)
    d = pd.DataFrame({"jun": M.loc[ids, f"{y}-06"], "jul": M.loc[ids, f"{y}-07"]}, index=ids)
    d["age"] = np.where(a < 3, "<3m", np.where(a < 12, "3-12m", "12m+"))
    d["cell"] = d.age + "|" + it.product_type.reindex(ids).map(MAC).fillna("other").values + "|" + pd.cut(pos, [0, .35, .7, 10], labels=["<35", "35-70", ">=70"]).astype(str).values
    return d
d24, d25 = frame(2024), frame(2025)
for key in ["age", "cell"]:
    r = d24.groupby(key).jul.sum() / d24.groupby(key).jun.sum(); w = d25.groupby(key).jun.sum()
    fb = d24.jul.sum() / d24.jun.sum(); bench = (w * r.reindex(w.index).fillna(fb)).sum() / w.sum()
    print(f"benchmark by {key:4s}: predicted July/June for June 2025 = {bench:.3f} | actual = {d25.jul.sum() / d25.jun.sum():.3f} | benchmark error {bench / (d25.jul.sum() / d25.jun.sum()) - 1:+.1%}")
for g in ["<3m", "3-12m", "12m+"]:
    a24 = d24[d24.age == g]; a25 = d25[d25.age == g]
    print(f"  {g:6s} 2024 ratio {a24.jul.sum() / a24.jun.sum():.3f} -> 2025 actual {a25.jul.sum() / a25.jun.sum():.3f}")
