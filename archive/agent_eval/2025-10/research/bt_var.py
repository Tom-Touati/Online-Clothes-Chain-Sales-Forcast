import sys; sys.path.insert(0,"research"); sys.path.insert(0,".")
from bt import run
V={"v8":{},"drift1.0":{"DRIFT_ADJ":1.0},"drift0.99":{"DRIFT_ADJ":0.99},
   "amp_win2":{"AMP_WIN":2},"amp_win1":{"AMP_WIN":1},"amp_floor.6":{"AMP_CLIP":(0.6,1.0)},
   "ret.6":{"RET_POW":0.6},"ret1.0":{"RET_POW":1.0},"anc.25":{"SEAS_ANCHOR":0.25},
   "w14.3":{"W14":0.3},"wlong.3":{"WLONG":0.3},"pow.03":{"POW":0.03}}
rows=[]
for k,v in V.items():
    R=run(v); m=R.mean()
    rows.append((k,round(m.wape,4),round(m.bias,3),round(m.h20,3),round(m.h50,3),round(m.w1,4),
                 " ".join(f"{x:.3f}" for x in R.wape)," ".join(f"{x:+.2f}" for x in R.bias)," ".join(f"{x:.2f}" for x in R.win)))
    print(rows[-1],flush=True)
