import pandas as pd, numpy as np, sys
sys.path.insert(0,"."); import rules as R
it, di, st = R.load()
occ = it.occasion.fillna("")
M = di.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").fillna(0)
months = [pd.Period(x) for x in ["2024-03","2024-04","2024-05","2024-06","2024-07","2024-08","2024-09","2025-03","2025-04"]]
for tag in ["4th of July","Graduation","Homecoming","Halloween"]:
    has = occ.str.contains(tag, regex=False)
    for base in [pd.Period("2024-04"), pd.Period("2025-03")]:
        live = M.index[(it.launch.reindex(M.index) < base.start_time - pd.Timedelta(days=30)) & (M[base] > 0)]
        X = M.loc[live]
        tg = has.reindex(X.index).fillna(False)
        grp = it.grp.reindex(X.index)
        out = {}
        for t in months:
            if t <= base: continue
            # tagged vs untagged items of the same season group, LFL ratio t/base
            rel = []
            for gname in ["SUMMER","CORE","WINTER"]:
                a = X[tg & (grp == gname)]; b = X[~tg & (grp == gname)]
                if len(a) >= 3 and a[base].sum() > 0:
                    rel.append((a[t].sum() / a[base].sum()) / (b[t].sum() / b[base].sum()) * a[base].sum())
            n = X[tg][base].sum()
            out[str(t)] = round(sum(rel) / n, 2) if n and rel else None
        print(tag, "base", base, "n", int(tg.sum()), out)
