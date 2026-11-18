from bt import run
df0 = run("tags v14, COH off", {"COH_POW": 0}); print("   ", " ".join(f"{w:.3f}" for w in df0.wape))
df1 = run("tags v14, COH 0.5", {}); print("   ", " ".join(f"{w:.3f}" for w in df1.wape))
