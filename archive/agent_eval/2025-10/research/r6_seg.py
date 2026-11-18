# attribute profiles vs SAME-SEGMENT untagged items (Oct-24 base), and scoreboard rel by attribute (h=1)
import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules
it,di,st=rules.load()
at=pd.read_csv("data/item_attributes.csv").set_index("style_color")
txt=at.fillna("").astype(str).agg(" ".join,axis=1).str.lower()
occ=it.occasion.fillna("")
mi=pd.read_parquet("data/monthly_item.parquet")
M=mi.pivot_table(index="style_color",columns="month",values="units",aggfunc="sum").fillna(0)
def prof(mask,base="2024-10",ms=("2024-10","2024-11","2024-12","2025-01","2025-02"),live="2024-09-01"):
    ok=M.index[(M[base]>0)&(it.launch.reindex(M.index)<live)]
    X=M.loc[ok,list(ms)]; seg=it.seg.reindex(ok); has=mask.reindex(ok).fillna(False).astype(bool)
    w=X[has].groupby(seg[has])[base].sum(); num=0; den=0
    for s in w.index:
        u=X[(~has)&(seg==s)]
        if u[base].sum()<200: continue
        num+=X[has&(seg==s)].sum(); den+=w[s]*u.sum()/u[base].sum()
    return int(has.sum()), (num/den).round(2).tolist() if isinstance(den,pd.Series) else None
masks={"Halloween":occ.str.contains("Halloween"),"NYE-noHC":occ.str.contains("New Year")&~occ.str.contains("Homecoming"),
       "Homecoming":occ.str.contains("Homecoming"),"Valentine":occ.str.contains("Valentine"),
       "party girl":txt.str.contains("party girl"),"going out":txt.str.contains("going out"),
       "party-noHal":txt.str.contains("party girl")&~occ.str.contains("Halloween"),
       "sheer":txt.str.contains("sheer"),"lounge":txt.str.contains("lounge"),"denim":txt.str.contains("denim"),
       "evening tier":at.occasion_tier.reindex(it.index).fillna("").str.contains("Evening|Party"),
       "everyday tier":at.occasion_tier.reindex(it.index).fillna("").str.contains("Everyday|Casual")}
print(at.occasion_tier.value_counts().head(8).to_string())
for k,m in masks.items(): print(f"{k:14s}", prof(m))
sb=pd.read_csv("feedback/scoreboard.csv")
for o,mo in [("2025-09","2025-10"),("2025-08","2025-09"),("2025-07","2025-08"),("2025-06","2025-07")]:
    x=sb[(sb.origin==o)&(sb.month==mo)]; tot=x.actual.sum()/x.pred.sum(); out=[]
    for k,m in masks.items():
        y=x[x.style_color.map(m).fillna(False).astype(bool)]
        out.append(f"{k}:{(y.actual.sum()/y.pred.sum()/tot):.2f}(n{len(y)})")
    print(mo," ".join(out))
print("---- ex-Halloween / overlap")
noH=~occ.str.contains("Halloween")
tier=at.occasion_tier.reindex(it.index).fillna("")
for k,m in {"evening-noHal":tier.str.contains("Evening")&noH,"party tier-noHal":tier.str.contains("Party")&noH,
            "sheer-noHal":txt.str.contains("sheer")&noH,"Hal&evening":occ.str.contains("Halloween")&tier.str.contains("Evening|Party"),
            "Hal&other tier":occ.str.contains("Halloween")&~tier.str.contains("Evening|Party")}.items():
    print(f"{k:16s}",prof(m), "LYSep->Oct",prof(m,"2024-09",("2024-09","2024-10"),"2024-08-01"), "TY Sep->Oct",prof(m,"2025-09",("2025-09","2025-10"),"2025-08-01"))
