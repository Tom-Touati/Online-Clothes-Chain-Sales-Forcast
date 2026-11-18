"""Dump level factors for the June-25 origin; check the new store; v4 June channel split."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import rules
pd.set_option("display.width", 200)
it, di, st = rules.load()
o = di.date.max(); B = o.to_period("M"); T = [B + h for h in range(1, 5)]
f = rules.seasonal_factors(it, di, st, B, T)
print("web", f["web"].round(3).to_dict())
print("defl_w", f["defl_w"].round(3).to_dict())
print("a_feb (LY LFL, regime-fixed)", {str(k): round(v, 3) for k, v in f["a_feb"].items()})
print("ret", f["ret"].round(3).to_dict(), "ret_adj", round(f["ret_adj"], 3), "cur_pd", round(f["cur_pd"], 3), "cur_ret", round(f["cur_ret"], 3))
print("grp_rel\n", f["grp_rel"].round(2).to_string())
seg = f["seg_rel"].copy(); seg["n"] = f["seg_n"]
print(seg[seg.n >= 8].round(2).to_string())
print("NYR daily:", st[st.store == "NYR"].set_index("date").units.to_dict())
