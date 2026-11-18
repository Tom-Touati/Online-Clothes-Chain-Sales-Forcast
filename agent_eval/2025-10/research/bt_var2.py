import sys; sys.path.insert(0,"research"); sys.path.insert(0,".")
from bt import run
V={"v9":{},"v9 no spike cap":{"SPIKE_X":1e9},"v9 cap3":{"SPIKE_CAP":3.0},"v9 no tier":{"TIER_IDX":{}},"v9 HAL_SOFT1":{"HAL_SOFT":1.0},"v9 drift1.01":{"DRIFT_ADJ":1.01}}
for k,v in V.items():
    R=run(v); m=R.mean()
    print(k,round(m.wape,4),round(m.bias,3),round(m.h20,3),round(m.h50,3),round(m.w1,4)," ".join(f"{x:.3f}" for x in R.wape)," ".join(f"{x:+.2f}" for x in R.bias),flush=True)
