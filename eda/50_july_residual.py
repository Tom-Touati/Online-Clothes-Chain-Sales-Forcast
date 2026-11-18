"""After the D061 July calibration, is July still short in the June 2025 replay? (eda/49_rules_in_sim.py objects)"""
import importlib.util, sys, numpy as np, pandas as pd, io, contextlib
sys.stdout.reconfigure(encoding="utf-8")
spec = importlib.util.spec_from_file_location("r49", "eda/49_rules_in_sim.py"); r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r); print("-"*40)
FACTOR = {"12m+": 0.295 / 0.282, "3-12 months": 0.271 / 0.273, "launch <3m": 0.335 / 0.297}
age = (pd.Timestamp("2025-06-30") - r.start.reindex(r.I)).dt.days.values / 30.44
grp = np.where(age < 3, "launch <3m", np.where(age < 12, "3-12 months", "12m+"))
segshape = np.isin(r.sg, ["swimwear", "Homecoming", "Halloween"]) & ~r.mature
wgt = np.where(segshape, 0.0, np.where(r.mature, 0.5, 1.0))
F = r.V["all rules"].copy(); tot = F.sum(1)
F[:, 0] *= np.array([1 + wi * (FACTOR[g] - 1) for wi, g in zip(wgt, grp)])
F[:, 1:] *= ((tot - F[:, 0]) / F[:, 1:].sum(1))[:, None]
jun, act = r.jun, r.act
print(f"July/June 2025 replay: rules {r.V['all rules'][:,0].sum()/jun.sum():.3f} | rules + D061 calibration {F[:,0].sum()/jun.sum():.3f} | benchmark {r.V['all rules + July at benchmark'][:,0].sum()/jun.sum():.3f} | actual {act[:,0].sum()/jun.sum():.3f}")
for name, m in [("segment-shaped (weight 0)", segshape), ("mature (weight 0.5)", r.mature & ~segshape), ("other (weight 1)", ~r.mature & ~segshape)]:
    print(f"  {name:27s} n={m.sum():3d} June {jun[m].sum():7.0f} | July/June forecast {F[m,0].sum()/jun[m].sum():.3f} actual {act[m,0].sum()/jun[m].sum():.3f} | July share forecast {F[m,0].sum()/F[m].sum():.3f} actual {act[m,0].sum()/act[m].sum():.3f}")
for a in (0.0, 0.5):
    p_c = r.run(F, a)[0].sum(); p_r = r.run(r.V["all rules"], a)[0].sum(); p_b = r.run(r.V["all rules + July at benchmark"], a)[0].sum(); ip = r.ip.sum()
    print(f"alpha {a}: loss rules {(ip-p_r)/ip:.4f} | rules+D061 {(ip-p_c)/ip:.4f} | rules+benchmark July {(ip-p_b)/ip:.4f}")
