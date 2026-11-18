import sys; sys.path.insert(0,"research"); sys.argv=["x"]
exec(open("research/bt1.py").read().split("variants = {")[0])
base = dict(W_DEFL=1.0, OTHER_MONTHLY=0.9)
variants = {
 "defl0.75+oth0.9": dict(W_DEFL=0.75, OTHER_MONTHLY=0.9),
 "defl1+oth0.85": dict(W_DEFL=1.0, OTHER_MONTHLY=0.85),
 "defl1+oth0.95": dict(W_DEFL=1.0, OTHER_MONTHLY=0.95),
 "B site0.95": dict(base, SITE_WEB=0.95),
 "B ramp 1.3/1.25/1.1": dict(base, RAMP0=1.3, RAMP1=1.25, RAMP2=1.1),
 "B ramp 1.8/1.5/1.2": dict(base, RAMP0=1.8, RAMP1=1.5, RAMP2=1.2),
 "B K4": dict(base, SHRINK_K=4), "B K16": dict(base, SHRINK_K=16),
 "B W14 0.6": dict(base, W14=0.6), "B WLONG 0.1": dict(base, WLONG=0.1), "B WLONG 0.3": dict(base, WLONG=0.3),
 "B POW0.05": dict(base, POW=0.05),
 "B CH_K 5": dict(base, CH_K=5),
}
for k,v in variants.items():
    r = run(v)
    print(f"{k:22s} mean WAPE {r.wape.mean():.3f} ±50 {r.in50.mean():.3f}| " + " | ".join(f"{x.o}: w{x.wape:.3f} b{x.bias:+.3f} [{x.mb}]" for x in r.itertuples()))
o="2025-04-30"; B=pd.Period("2025-04"); T=[B+h for h in range(1,5)]
old=dict(R.P); R.P.update(base); f=R.seasonal_factors(it,di,st,B,T); R.P.update(old)
print("defl web now", f["defl_w"].round(3).to_dict()); print("ret now", f["ret"].round(3).to_dict())
