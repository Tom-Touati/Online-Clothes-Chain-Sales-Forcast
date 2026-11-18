"""Review of v9 (origin 2025-10 -> Nov-25, h=1) and other origins; drift / amplitude Oct->Nov TY vs LY."""
import sys
import pandas as pd, numpy as np
sys.path.insert(0, ".")
import rules as R

it, di, st = R.load()
sb = pd.read_csv("feedback/scoreboard.csv")
print("WAPE / bias by origin-month (h):")
g = sb.groupby(["h"]).apply(lambda x: pd.Series({"wape": (x.pred - x.actual).abs().sum() / x.actual.sum(),
                                                  "bias": x.pred.sum() / x.actual.sum() - 1}))
print(g.round(3).to_string())

s = sb[(sb.origin == "2025-10")].copy()
s = s.join(it[["grp", "seg", "product_type", "occasion", "tier", "launch"]], on="style_color")
lm = di[di.m == pd.Period("2025-10")].groupby("style_color")[["units", "web_units", "retail", "other"]].sum()
s = s.join(lm, on="style_color")
tot = s.actual.sum() / s.pred.sum()
print(f"\nNov h=1 total pred {s.pred.sum():.0f} actual {s.actual.sum():.0f} ratio {tot:.3f}")
def rel(col, x=s):
    t = x.groupby(col).agg(n=("pred", "size"), pred=("pred", "sum"), act=("actual", "sum"))
    t["rel"] = (t.act / t.pred / tot).round(3)
    t["wape"] = x.groupby(col).apply(lambda y: (y.pred - y.actual).abs().sum() / max(y.actual.sum(), 1)).round(3)
    return t[t.n >= 8].sort_values("pred", ascending=False)
print(rel("grp").to_string())
print(rel("product_type").head(25).to_string())
s["age"] = pd.cut((pd.Timestamp("2025-10-01") - s.launch).dt.days / 30.44, [-9, 0, 1, 2, 4, 8, 14, 99])
print(rel("age").to_string())
s["ret_sh"] = pd.cut(s.retail / s.units.clip(lower=1), [-0.01, 0.0001, 0.1, 0.25, 0.5, 1.01])
s["web_sh"] = pd.cut(s.web_units / s.units.clip(lower=1), [-0.01, 0.5, 0.75, 0.9, 1.01])
s["oth_sh"] = pd.cut(s.other / s.units.clip(lower=1), [-0.01, 0.05, 0.15, 0.3, 1.01])
print(rel("ret_sh").to_string()); print(rel("web_sh").to_string()); print(rel("oth_sh").to_string())
s["size"] = pd.qcut(s.pred.rank(method="first"), 5, labels=False)
print(rel("size").to_string())
for tag in ["Halloween", "New Year's Eve", "Valentine's", "Homecoming", "Graduation", "4th of July", "Christmas",
            "Holiday", "Party", "Date Night", "Vacation", "Festival", "St. Patrick"]:
    has = s.occasion.fillna("").str.contains(tag, regex=False)
    if has.sum() >= 5:
        x = s[has]
        print(f"tag {tag:16s} n={has.sum():4d} pred {x.pred.sum():8.0f} rel {x.actual.sum()/x.pred.sum()/tot:.3f}")
s["party"] = s.tier.str.contains("Party") & ~s.occasion.fillna("").str.contains("Halloween")
print(rel("tier").to_string())

# drift TY Oct->Nov vs LY Oct->Nov / Nov->Dec ... (LFL per-day move / site web per-day move)
def lfl(m0, m1, col="web_units"):
    X = di[di.m.isin([m0, m1])].pivot_table(index="style_color", columns="m", values=col, aggfunc="sum").fillna(0)
    ok = (it.launch.reindex(X.index) < m0.start_time - pd.Timedelta(days=30)) & (X[m0] > 0)
    r = X[ok][m1].sum() / X[ok][m0].sum() * m0.days_in_month / m1.days_in_month
    site = di[di.m.isin([m0, m1])].groupby("m")[col].sum()
    sr = site[m1] / site[m0] * m0.days_in_month / m1.days_in_month
    return r, sr, r / sr
P_ = pd.Period
for a, b in [("2024-10", "2024-11"), ("2024-11", "2024-12"), ("2024-12", "2025-01"), ("2025-01", "2025-02"),
             ("2025-02", "2025-03"), ("2025-09", "2025-10"), ("2025-10", "2025-11")]:
    print(a, b, "web lfl / site / drift %.3f %.3f %.3f" % lfl(P_(a), P_(b)), "| other drift %.3f" % lfl(P_(a), P_(b), "other")[2])
# group moves rel panel, TY vs LY, Oct->Nov
def grp_move(m0, m1):
    X = di[di.m.isin([m0, m1])].pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").fillna(0)
    ok = (it.launch.reindex(X.index) < m0.start_time - pd.Timedelta(days=30)) & (X[m0] > 0)
    X = X[ok]; gg = X.groupby(it.grp.reindex(X.index))
    r = gg[m1].sum() / gg[m0].sum()
    return (r / (X[m1].sum() / X[m0].sum())).round(3)
print("TY Oct->Nov group rel", grp_move(P_("2025-10"), P_("2025-11")).to_dict())
print("LY Oct->Nov group rel", grp_move(P_("2024-10"), P_("2024-11")).to_dict())
for a, b in [("2024-11", "2024-12"), ("2024-12", "2025-01"), ("2025-01", "2025-02"), ("2025-02", "2025-03")]:
    print("LY", a, b, grp_move(P_(a), P_(b)).to_dict())
# LY daily around BF / CM, Christmas
s2 = pd.read_parquet("data/daily_site.parquet").set_index("date")
print(s2.loc["2024-11-24":"2024-12-08", ["units", "web_units", "retail"]].to_string())
print("TY Nov 1-22 web/day %.0f vs LY Nov 1-23 %.0f" % (s2.loc["2025-11-01":"2025-11-22", "web_units"].mean(),
                                                        s2.loc["2024-11-01":"2024-11-23", "web_units"].mean()))
