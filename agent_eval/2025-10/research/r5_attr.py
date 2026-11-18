# attribute / tag profiles from an Oct-24 base (tagged vs same-group untagged), TY Sep->Oct check
import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
at=pd.read_csv("data/item_attributes.csv").set_index("style_color")
pass
mi=pd.read_parquet("data/monthly_item.parquet")
M=mi.pivot_table(index="style_color",columns="month",values="units",aggfunc="sum").fillna(0)
def prof(mask_all, base="2024-10", ms=("2024-10","2024-11","2024-12","2025-01","2025-02"), live="2024-09-01"):
    ok=M.index[(M[base]>0)&(it.launch.reindex(M.index)<live)]
    X=M.loc[ok,list(ms)]; grp=it.grp.reindex(ok); has=mask_all.reindex(ok).fillna(False).astype(bool)
    if has.sum()<5: return None
    w=X[has].groupby(grp[has])[base].sum()
    u=sum(w[g]*(X[(~has)&(grp==g)].sum()/X[(~has)&(grp==g)][base].sum()) for g in w.index)/w.sum()
    t=X[has].sum()/X[has][base].sum()
    return has.sum(), (t/u).round(2).tolist()
occ=it.occasion.fillna("")
print("NYE excl Homecoming", prof(occ.str.contains("New Year")&~occ.str.contains("Homecoming")))
print("NYE TY Sep->Oct", prof(occ.str.contains("New Year")&~occ.str.contains("Homecoming"),"2025-09",("2025-09","2025-10"),"2025-08-01"))
print("Halloween TY Sep->Oct", prof(occ.str.contains("Halloween"),"2025-09",("2025-09","2025-10"),"2025-08-01"))
print("Halloween LY Sep->Oct", prof(occ.str.contains("Halloween"),"2024-09",("2024-09","2024-10"),"2024-08-01"))
txt=at.fillna("").astype(str).agg(" ".join,axis=1).str.lower()
for kw in ["sequin","velvet","satin","metallic","faux fur","fur","lace","sheer","leather","denim","fleece","knit","wool","plaid","red","party","holiday","christmas","festive","cozy","going out","lounge","y2k","western","boho","office"]:
    r=prof(txt.str.contains(kw,regex=False))
    if r: print(f"{kw:10s} n={r[0]:4d} Oct..Feb {r[1]}")
# colour
col=it.color.fillna("").str.upper()
for c in ["RED","BURGUNDY","BLACK","WHITE","GRAY","BROWN","PINK","SILVER","GOLD","GREEN"]:
    r=prof(col.str.contains(c,regex=False)); 
    if r: print(f"col {c:9s} n={r[0]:4d} {r[1]}")
# Halloween: blended base vs Oct month rate
o=pd.Timestamp("2025-10-31"); d=di[di.date>o-pd.Timedelta(days=31)]
D=d.pivot_table(index="style_color",columns="date",values="units",aggfunc="sum").fillna(0)
r31=D.sum(axis=1)/31; r14=D.iloc[:,-14:].sum(axis=1)/14
h=occ.reindex(D.index).fillna("").str.contains("Halloween")
print("Halloween r14/r31", (r14[h].sum()/r31[h].sum()).round(3), "others", (r14[~h].sum()/r31[~h].sum()).round(3))
# last-7-days collapse of Halloween
r7=D.iloc[:,-7:].sum(axis=1)/7; print("Halloween r7/r31",(r7[h].sum()/r31[h].sum()).round(3),"others",(r7[~h].sum()/r31[~h].sum()).round(3))
dd=di[(di.date>="2025-10-20")].copy(); dd["h"]=dd.style_color.map(occ).fillna("").str.contains("Halloween")
print(dd.groupby(["date","h"]).units.sum().unstack().to_string())
