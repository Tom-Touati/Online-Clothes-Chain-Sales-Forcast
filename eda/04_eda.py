import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 120)
C = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
plt.rcParams.update({"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#c9c8c2","axes.grid":True,"grid.color":"#e6e5df","grid.linewidth":0.6,"axes.axisbelow":True,
  "text.color":"#0b0b0b","axes.labelcolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","font.size":9,
  "axes.titlesize":10.5,"axes.titleweight":"bold","axes.titlelocation":"left","lines.linewidth":2,"legend.frameon":False,"figure.dpi":130})
d = pd.read_parquet("data/daily_item_chan.parquet"); d["ny_date"]=pd.to_datetime(d["ny_date"])
it = pd.read_parquet("data/items.parquet")
it["published_date"]=pd.to_datetime(it["published_date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
it["first_sale"]=pd.to_datetime(it["first_sale"]); it["last_sale"]=pd.to_datetime(it["last_sale"])
tg = pd.read_csv("data/targets.csv"); T = set(tg.style_color)
it["target"]=it.style_color.isin(T)
ok = d[~d.cancelled].copy(); ok["m"]=ok.ny_date.dt.to_period("M")
P = lambda y,ms: [pd.Period(f"{y}-{m:02d}") for m in ms]

print("== ITEM MASTER consistency"); print(it[["n_names","n_types","n_coll","n_pub","n_colnum","n_occ"]].max().to_dict())
print("items first sale before published:", (it.first_sale<it.published_date).sum(), "| published before 2024:", (it.published_date<"2024-01-01").sum(),
      "| published 2026:", (it.published_date>="2026-01-01").sum())
x=it[it.published_date>="2024-02-01"]; print("lag publish->first sale (days), items published >=2024-02:", ((x.first_sale-x.published_date).dt.days).describe(percentiles=[.05,.25,.5,.75,.95]).round(1).to_dict())
print("published year x target:\n", it.groupby([it.published_date.dt.year,"target"]).size().unstack(fill_value=0))
print("last sale quarter x target:\n", it.groupby([it.last_sale.dt.to_period("Q"),"target"]).size().unstack(fill_value=0).tail(6))
print("product types (all / target):\n", pd.concat([it.product_type.value_counts(), it[it.target].product_type.value_counts()],axis=1).fillna(0).astype(int).head(30))
print("occasion among targets:\n", it[it.target].occasion.fillna("-").value_counts())

mc = ok.groupby(["m","class"]).qty.sum().unstack(fill_value=0); mc["TOTAL"]=mc.sum(axis=1)
print("\n== MONTHLY UNITS BY CLASS"); print(mc.round(0).astype(int))
canc = d.assign(m=d.ny_date.dt.to_period("M")).groupby(["m","cancelled"]).qty.sum().unstack(fill_value=0); print("cancel rate by month:", (canc[True]/canc.sum(axis=1)).round(3).tolist())
fig,ax=plt.subplots(figsize=(9,3.6))
for i,c in enumerate(["ONLINE","RETAIL","DROP"]): ax.plot(mc.index.to_timestamp(), mc[c]/1e3, color=C[i], label={"DROP":"DROP (dropship marketplaces)"}.get(c,c.title()), marker="o", ms=3)
ax.set_title("Monthly units by sales class (non-cancelled, 1,500 items)"); ax.set_ylabel("units (thousands, scaled)"); ax.legend(ncol=3, loc="upper left"); ax.set_ylim(0)
fig.tight_layout(); fig.savefig("eda/figs/01_monthly_by_class.png"); plt.close()

ch = ok.groupby(["m","chan"]).qty.sum().unstack(fill_value=0); print("\n== MONTHLY BY CHANNEL (k)"); print((ch/1e3).round(1).to_string())
st = pd.read_parquet("data/daily_otype.parquet"); st["ny_date"]=pd.to_datetime(st.ny_date); st=st[~st.cancelled]
fs = st.groupby(["class","otype"]).agg(first=("ny_date","min"), last=("ny_date","max"), qty=("qty","sum")).sort_values(["class","first"]); print("\n== CHANNEL / STORE FIRST-LAST DATES"); print(fs)

on = ok[ok["class"]=="ONLINE"].groupby("ny_date").agg(qty=("qty","sum"), dw=("disc_w","sum")); on["disc"]=on.dw/on.qty
alld = ok.groupby("ny_date").qty.sum().reindex(pd.date_range("2024-01-01","2026-06-30"), fill_value=0)
print("\n== DAILY: zero days:", (alld==0).sum()); print("lowest 8 days:\n", alld.nsmallest(8).to_string()); print("top 8 days:\n", alld.nlargest(8).to_string())
la=np.log(alld.clip(lower=1)); z=la-la.rolling(29,center=True,min_periods=10).median(); print("days with <40% of local median:\n", alld[z<np.log(0.4)].to_string())
fig,axs=plt.subplots(2,1,figsize=(10,5.2),sharex=True)
axs[0].plot(alld.index, alld/1e3, color=C[0], lw=1); axs[0].set_title("Daily units, all channels"); axs[0].set_ylabel("units (k)"); axs[0].set_ylim(0)
axs[1].plot(on.index, on.disc*100, color=C[1], lw=1); axs[1].set_title("Online: unit-weighted discount depth (the past campaign calendar is visible here)"); axs[1].set_ylabel("% off")
fig.tight_layout(); fig.savefig("eda/figs/02_daily_units_discount.png"); plt.close()
onm = ok[ok["class"]=="ONLINE"].groupby("m").agg(qty=("qty","sum"), dw=("disc_w","sum")); print("online monthly disc:", (onm.dw/onm.qty).round(3).tolist())
on["rel"]=on.qty/on.qty.rolling(29,center=True,min_periods=10).median(); on["dd"]=on.disc-on.disc.rolling(29,center=True,min_periods=10).median()
print("corr(daily rel units, disc deviation) online:", round(on[["rel","dd"]].corr().iloc[0,1],3)); print(on.assign(b=pd.cut(on.disc,[0,.45,.5,.55,.6,.65,.7,1])).groupby("b",observed=True).agg(days=("rel","size"), rel_units=("rel","mean")).round(2))
dow = alld.groupby(alld.index.dayofweek).mean(); print("day-of-week index (Mon..Sun):", (dow/dow.mean()).round(2).tolist())

M = ok.groupby(["style_color","m"]).qty.sum().unstack(fill_value=0); M = M.reindex(columns=pd.period_range("2024-01","2026-06",freq="M"), fill_value=0)
Ms = M.copy(); Ms.columns = Ms.columns.astype(str); Ms.to_parquet("data/monthly_item.parquet")
it=it.set_index("style_color")
am = M.where(M.columns.to_timestamp(how="end").values[None,:] >= it.loc[M.index,"published_date"].values[:,None]); v=am.stack()
print("\n== ITEM-MONTH distribution (months after publish):", v.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(1).to_dict()); print("share zero item-months after publish:", round((v==0).mean(),3), "| <10 units:", round((v<10).mean(),3))
tot=M.sum(axis=1).sort_values(ascending=False); print("concentration: top 10% items share:", round(tot.head(150).sum()/tot.sum(),3), "| target items share of all units:", round(tot[tot.index.isin(T)].sum()/tot.sum(),3))

pt = it.product_type
ty = M.groupby(pt.reindex(M.index).values).sum()
big = ty.sum(axis=1).sort_values(ascending=False).head(22).index
yr = lambda y,ms: ty[P(y,ms)].sum(axis=1)
sea = pd.DataFrame({"JulOct24/MarJun24": yr(2024,[7,8,9,10])/yr(2024,[3,4,5,6]), "JulOct25/MarJun25": yr(2025,[7,8,9,10])/yr(2025,[3,4,5,6]), "MarJun26/MarJun25": yr(2026,[3,4,5,6])/yr(2025,[3,4,5,6]), "units(k)": ty.sum(axis=1)/1e3}).loc[big].round(2)
print("\n== TYPE-LEVEL SEASON RATIOS (all items, mix effects included)"); print(sea)
tot_ratio = {y: round(M[P(y,[7,8,9,10])].sum().sum()/M[P(y,[3,4,5,6])].sum().sum(),3) for y in (2024,2025)}; print("TOTAL JulOct/MarJun:", tot_ratio)
prof = ty.loc[big, P(2025,range(1,13))]; prof = prof.div(prof.mean(axis=1),axis=0)
fig,ax=plt.subplots(figsize=(9,5.5)); im=ax.imshow(prof.values, aspect="auto", cmap="Blues", vmin=0, vmax=2.5)
ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_yticks(range(len(big))); ax.set_yticklabels(big); ax.grid(False)
for i in range(prof.shape[0]):
    for j in range(12): ax.text(j,i,f"{prof.values[i,j]:.1f}",ha="center",va="center",fontsize=7,color="#0b0b0b" if prof.values[i,j]<1.6 else "white")
ax.set_title("2025 monthly seasonal index by product type (1.0 = that type's monthly average)"); fig.colorbar(im,ax=ax,shrink=.6); fig.tight_layout(); fig.savefig("eda/figs/03_type_seasonality.png"); plt.close()

occ = it.occasion.fillna("")
for o in ["Halloween","4th of July","Homecoming","Valentine","Graduation","New Year","St. Patrick","World Cup"]:
    idx = occ[occ.str.contains(o)].index; s = M.loc[M.index.isin(idx), P(2025,range(1,13))].sum(); print(f"occasion {o:12s} n={len(idx):4d} n_target={len(set(idx)&T):3d} 2025 idx:", (s/s.mean()).round(2).tolist())
s = M.loc[M.index.isin(occ[occ==""].index), P(2025,range(1,13))].sum(); print(f"occasion none         n={(occ=='').sum():4d} 2025 idx:", (s/s.mean()).round(2).tolist())

lc = []
for sc,pubd in it.published_date.items():
    if pd.Timestamp("2024-02-01")<=pubd<=pd.Timestamp("2025-06-30") and sc in M.index:
        p0 = pubd.to_period("M"); s = M.loc[sc]; s = s[s.index>=p0]; base = s.iloc[1:4].mean()
        if base>=30 and len(s)>=13: lc.append((s.values[:13]/base))
L = pd.DataFrame(lc); print("\n== LIFECYCLE (n=%d items; units rel. to mean of months 1-3 after launch month)"%len(L)); print(pd.DataFrame({"p25":L.quantile(.25),"med":L.median(),"p75":L.quantile(.75),"mean":L.mean()}).round(2).T.to_string())
fig,ax=plt.subplots(figsize=(8,3.6)); xx=range(13); ax.fill_between(xx, L.quantile(.25), L.quantile(.75), color=C[0], alpha=.18, label="25-75th pct"); ax.plot(xx, L.median(), color=C[0], marker="o", ms=4, label="median")
ax.set_ylim(0,1.5); ax.set_title(f"Launch curve: units by month since launch ({len(L)} items)"); ax.set_xlabel("months since launch month (0 = partial launch month)"); ax.set_ylabel("units / mean of months 1-3"); ax.legend(); fig.tight_layout(); fig.savefig("eda/figs/04_launch_curve.png"); plt.close()
mat = it.index[(it.published_date<"2024-03-01")]; a=M.loc[M.index.isin(mat)]; r=(a[P(2025,(7,8,9,10))].sum(axis=1)+1)/(a[P(2024,(7,8,9,10))].sum(axis=1)+1)
print("mature items (n=%d) JulOct25/JulOct24 ratio quantiles p10..p90:"%len(r), r.quantile([.1,.25,.5,.75,.9]).round(2).tolist(), "| aggregate:", round(a[P(2025,(7,8,9,10))].sum().sum()/a[P(2024,(7,8,9,10))].sum().sum(),2))
# persistence: JulOct vs preceding MarJun at item level, 2025, items published before 2025-01
a=M.loc[M.index.isin(it.index[it.published_date<"2025-01-01"])]; b=a[P(2025,(3,4,5,6))].sum(axis=1); f=a[P(2025,(7,8,9,10))].sum(axis=1); k=b>=100
print("items pub<2025 with MarJun25>=100 (n=%d): JulOct25/MarJun25 quantiles p10..p90:"%k.sum(), (f[k]/b[k]).quantile([.1,.25,.5,.75,.9]).round(2).tolist(), "| corr log:", round(np.corrcoef(np.log1p(b[k]),np.log1p(f[k]))[0,1],3))

t = it[it.target].copy(); Mt = M.loc[t.index]
t["age_m_at_Jul26"]=((pd.Timestamp("2026-07-01")-t.published_date).dt.days/30.4).round(1)
t["L3"]=Mt.iloc[:,-3:].sum(axis=1); t["L12"]=Mt.iloc[:,-12:].sum(axis=1); t["JulOct25"]=Mt[P(2025,(7,8,9,10))].sum(axis=1); t["JulOct24"]=Mt[P(2024,(7,8,9,10))].sum(axis=1)
t["zero_m_last6"]=(Mt.iloc[:,-6:]==0).sum(axis=1); t["jun26"]=Mt.iloc[:,-1]
print("\n== TARGETS"); print(t[["age_m_at_Jul26","L3","L12","JulOct25","JulOct24","jun26","zero_m_last6"]].describe(percentiles=[.1,.25,.5,.75,.9]).round(1))
print("age buckets (months at Jul-26):", pd.cut(t.age_m_at_Jul26,[0,6,12,24,36,100]).value_counts().sort_index().to_dict())
print("targets published after Jul 2025 (no Jul-Oct history):", (t.published_date>="2025-07-01").sum(), "| published 2026:", (t.published_date>="2026-01-01").sum())
print("targets with 0 units in last 3 months:", (t.L3==0).sum(), "| with <20 units L3:", (t.L3<20).sum(), "| in attributes file: n/a here")
cols=["product_name","product_type","published_date","L3","L12","JulOct25","jun26","zero_m_last6","occasion"]
print(t.sort_values("L3")[cols].head(15).to_string()); print(t.sort_values("L3")[cols].tail(10).to_string())
t.to_csv("data/targets_overview.csv")
for nm,sel in [("targets",ok.style_color.isin(T)),("all",ok.style_color.notna())]:
    cm = ok[sel&(ok.m>=pd.Period("2026-01"))].groupby("class").qty.sum(); print(nm, "class mix 2026H1:", (cm/cm.sum()).round(3).to_dict())

dd = ok.groupby(["style_color","ny_date"]).qty.sum().unstack(fill_value=0).reindex(columns=pd.date_range("2024-01-01","2026-06-30"), fill_value=0)
gaps=[]
for sc,row in dd.iterrows():
    v=row.values; nz=np.flatnonzero(v>0)
    if len(nz)<30: continue
    g=np.diff(nz); j=g.argmax(); rate=v[max(0,nz[j]-28):nz[j]+1].mean()
    if g[j]>=14 and rate>=3: gaps.append((sc, dd.columns[nz[j]], dd.columns[nz[j+1]], g[j], round(rate,1), sc in T))
G=pd.DataFrame(gaps,columns=["style_color","gap_start","gap_end","days","rate_before","target"]); print("\n== SUSPECTED STOCKOUT GAPS (>=14 zero days after >=3 units/day): items:", len(G), "| targets:", G.target.sum()); print(G.sort_values("rate_before",ascending=False).head(10).to_string()); print(G[G.target].sort_values("rate_before",ascending=False).head(15).to_string())
G.to_csv("data/suspected_stockouts.csv",index=False)
# trailing silence: items with last sale well before end
end=pd.Timestamp("2026-06-30"); t["days_since_last"]=(end-t.last_sale).dt.days; print("targets with no sale in last 30d:", (t.days_since_last>30).sum()); print(t[t.days_since_last>14][["product_type","last_sale","L12"]].to_string())
ex = list(t.sort_values("L12",ascending=False).index[:3])+list(G[G.target].sort_values("rate_before",ascending=False).style_color[:3])+list(t[t.published_date>="2026-01-01"].sort_values("L3",ascending=False).index[:3])
fig,axs=plt.subplots(3,3,figsize=(11,6.2))
for a_,sc in zip(axs.ravel(),ex):
    w=dd.loc[sc].resample("W").sum(); a_.plot(w.index,w.values,color=C[0],lw=1.3); a_.set_title(f"{sc} | {it.loc[sc,'product_type']}",fontsize=8); a_.set_ylim(0); a_.tick_params(labelsize=7)
    a_.axvline(it.loc[sc,"published_date"],color="#52514e",lw=.8,ls=":")
fig.suptitle("Weekly units - top: biggest targets | middle: suspected stock-out gaps | bottom: 2026 launches",x=0.01,ha="left",fontsize=10,fontweight="bold"); fig.tight_layout(); fig.savefig("eda/figs/05_example_items.png"); plt.close()
