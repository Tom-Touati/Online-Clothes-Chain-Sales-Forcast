from bt import run, R, it, di
import pandas as pd
for n, o in [("COH off (v13)", {"COH_POW": 0}), ("COH 0.5", {}), ("COH 0.75", {"COH_POW": 0.75}), ("COH 1.0", {"COH_POW": 1.0}),
             ("COH 0.75 age 4-15", {"COH_POW": 0.75, "COH_AGE": (4.0, 15.0)}), ("COH 0.75 age 2-12", {"COH_POW": 0.75, "COH_AGE": (2.0, 12.0)})]:
    df = run(n, o)
    print("   per-origin wape", " ".join(f"{w:.3f}" for w in df.wape))
R.P["COH_POW"] = 0.5; R.P["COH_AGE"] = (3.0, 15.0)
d = di; B = pd.Period("2026-06", "M"); T = [B + h for h in range(1, 5)]
st = pd.read_parquet("data/daily_store.parquet")
fac = R.seasonal_factors(it, d, st, B, T)
print({str(k): {a: round(b, 3) for a, b in v.items()} for k, v in R.cohort_factors(it, d, B, T, fac).items()})
