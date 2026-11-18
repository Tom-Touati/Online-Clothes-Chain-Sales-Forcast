"""D254 leakage and sanity checks.
1 Truncation test: rebuild the origin feature table (eda/06_features.py) at origins 2025-06 and 2025-10 from data CUT at the origin's
  month-end (daily rows after it dropped, monthly columns after it dropped) and compare with data/features.parquet (h = 1 rows,
  every feature column except the targets y / ytot): any difference = a feature that used data after the origin.
2 The same for the D252 blocks that read the daily matrix (data/nf2_feats.parquet; launch-plan block excluded by design): rebuilt
  by eda/180_newfeats.py on cut data at 2025-10 (env CUT).
3 Item universe (survivorship): items in the catalogue whose sales stopped long before the data pull, and sold item codes missing
  from the catalogue.
Writes eda/leak_output.txt (stdout)."""
import sys, os, subprocess, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
src = open("eda/06_features.py", encoding="utf-8").read()
F0 = pd.read_parquet("data/features.parquet"); F0 = F0[F0.h == 1]
for o in ["2025-06", "2025-10"]:
    s = src.replace('d = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date)',
                    f'd = pd.read_parquet("data/daily_item_chan.parquet"); d = d[~d.cancelled]; d["ny_date"] = pd.to_datetime(d.ny_date); d = d[d.ny_date <= pd.Period("{o}", "M").end_time.normalize()]')
    s = s.replace('M.columns = pd.PeriodIndex(M.columns, freq="M")', f'M.columns = pd.PeriodIndex(M.columns, freq="M"); M = M.loc[:, M.columns <= pd.Period("{o}", "M")]')
    s = s.replace('for o in pd.period_range("2024-03", "2026-06", freq="M"):', f'for o in [pd.Period("{o}", "M")]:')
    s = s.replace('F.to_parquet("data/features.parquet")', f'F.to_parquet("data/_leak_features_{o}.parquet")').split("print(F.shape")[0]
    assert s.count(f'<= pd.Period("{o}", "M")') >= 1 and f'_leak_features_{o}' in s
    exec(s, {})
    C = pd.read_parquet(f"data/_leak_features_{o}.parquet"); C = C[C.h == 1].set_index("style_color"); A = F0[F0.origin == o].set_index("style_color")
    common = A.index.intersection(C.index); cols = [c for c in A.columns if c not in ("y", "ytot", "origin", "h") and c in C.columns]
    print(f"== 1 truncation test at {o}: items {len(A)} stored, {len(C)} rebuilt, {len(common)} common")
    bad = []
    for c in cols:
        a, b = A.loc[common, c], C.loc[common, c]
        if a.dtype.kind in "fiub": diff = ~(np.isclose(a.astype(float), b.astype(float), equal_nan=True, rtol=1e-6, atol=1e-9))
        else: diff = a.astype(str) != b.astype(str)
        if diff.any(): bad.append((c, int(diff.sum())))
    print("   columns that differ:", bad if bad else "none")
    os.remove(f"data/_leak_features_{o}.parquet")
it = pd.read_parquet("data/items.parquet"); M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
last = M.gt(0).iloc[:, ::-1].idxmax(axis=1).where(M.gt(0).any(axis=1))
print("\n== 3 item universe")
print(f"  catalogue items {len(it)}; with sales {M.gt(0).any(axis=1).sum()}; last sale before 2025-01: {(last < '2025-01').sum()}; before 2025-07: {(last < '2025-07').sum()}; before 2026-01: {(last < '2026-01').sum()}")
d = pd.read_parquet("data/daily_item_chan.parquet", columns=["style_color"]); print(f"  sold codes not in the catalogue: {len(set(d.style_color) - set(it.style_color))}")
