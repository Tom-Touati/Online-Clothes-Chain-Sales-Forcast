"""Error map (D114) of the best system (rules x quantile x NN, 'r_q_nn') and its members on 4-month item totals, same items as eda/86.
(1) Where the error lives: units-weighted decomposition of the log error log((pred+1)/(y+1)) into origin (site level), origin x macro,
    origin x type, origin x style (siblings) and item residual, plus oracle rescaling: WAPE after scaling forecasts so that totals match
    the actuals at each level (an upper bound on what fixing that level is worth).
(2) Segments: WAPE / R3, bias and the best member per segment (age, volume, retail share, trend, occasion, macro, colour number,
    stock-out exposure, member disagreement, R3 track record).
Writes data/error_map_items.parquet for the later probes."""
import numpy as np, pandas as pd, contextlib, io, sys
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
_a = sys.argv; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()): exec(open("eda/86_metrics_suite.py", encoding="utf-8").read().split("REC_ = ")[0])
sys.argv = _a
it = pd.read_parquet("data/items.parquet").set_index("style_color")
F1 = pd.read_parquet("data/features.parquet"); F1 = F1[F1.h == 1][["origin", "style_color", "level", "sh_ret", "sh_drop", "trend91", "disc_item", "f_r28", "lagr1", "color_num", "ptype"]]
T = T.merge(F1, on=["origin", "style_color"], how="left", suffixes=("", "_f"))
X85 = pd.read_parquet("data/relative_eval.parquet")[["origin", "style_color", "ante"]]; T = T.merge(X85, on=["origin", "style_color"], how="left")
IB = pd.read_parquet("data/instock_backtest.parquet")[["origin", "style_color", "u_own"]]; T = T.merge(IB, on=["origin", "style_color"], how="left")
T["style"] = T.style_color.str.split("_").str[0]
M = ["rules", "qmodel", "nn", "lgb"]
L = np.log(T[M].clip(lower=0.5)); T["disagree"] = L.std(axis=1); T["rules_vs_q"] = L.rules - L.qmodel; T["nn_vs_q"] = L.nn - L.qmodel
T["vol"] = pd.qcut(T.level.rank(method="first"), 5, labels=["v1 smallest", "v2", "v3", "v4", "v5 largest"]).astype(str)
T["retail"] = pd.cut(T.sh_ret.fillna(0), [-1, 0.05, 0.2, 0.4, 1.1], labels=["<5%", "5-20%", "20-40%", ">40%"]).astype(str)
T["trend"] = pd.cut(T.trend91, [-9, -0.3, -0.1, 0.1, 0.3, 9], labels=["falling fast", "falling", "flat", "rising", "rising fast"]).astype(str).replace("nan", "no 6m history")
T["dis_q"] = pd.qcut(T.disagree.rank(method="first"), 5, labels=["d1 agree", "d2", "d3", "d4", "d5 disagree"]).astype(str)
T["oos"] = pd.cut(T.u_own.fillna(0), [-1, 0, 0.05, 1], labels=["none", "<5%", ">=5%"]).astype(str)
T["launch_m"] = pd.to_datetime(it.published_date.reindex(T.style_color).values, utc=True).month
T["launch_season"] = pd.cut(T.launch_m, [0, 2, 5, 8, 11, 12], labels=["winter", "spring", "summer", "autumn", "winter"], ordered=False).astype(str)
T["cnum"] = T.color_num.clip(upper=3).astype("Int64").astype(str)
REC_ = [o for o in ORIGINS if o >= "2025-06"]; R = T[T.origin.isin(REC_)].copy()
S = "r_q_nn"
def oracle(F, c, keys):
    s = F.groupby(keys)[[c, "y"]].transform("sum"); p = F[c] * (s.y / s[c].replace(0, np.nan)).fillna(1)
    return (p - F.y).abs().sum() / F.y.sum()
print("######## (1) where the error lives (nine recent origins, three-way)")
w0 = (R[S] - R.y).abs().sum() / R.y.sum(); print(f"WAPE as is {w0:.3f}")
for keys, nm in [(["origin"], "site level known"), (["origin", "macro"], "+ product group totals known"), (["origin", "ptype"], "+ product type totals known"),
                 (["origin", "style"], "+ style totals known (siblings)"), (["origin", "occ"], "occasion-tag totals known"), (["origin", "age"], "age-band totals known"), (["origin", "retail"], "retail-share band totals known")]:
    print(f"  oracle: {nm:34s} WAPE {oracle(R, S, keys):.3f}")
e = np.log((R[S] + 1) / (R.y + 1)); wt = R.y + 1
def wvar(x): m = np.average(x, weights=wt); return np.average((x - m) ** 2, weights=wt)
tot = wvar(e); prev = np.zeros(len(e)); rem = e.copy(); print(f"units-weighted variance of log error {tot:.3f}; share explained by group means, added level by level:")
for keys, nm in [(["origin"], "origin (site level)"), (["origin", "macro"], "origin x macro"), (["origin", "ptype"], "origin x type"), (["origin", "style"], "origin x style")]:
    g = (e * wt).groupby([R[k] for k in keys]).transform("sum") / wt.groupby([R[k] for k in keys]).transform("sum")
    print(f"  {nm:22s} cumulative share {1 - wvar(e - g) / tot:.3f}")
print("  item residual = the rest")
print("\n######## (2) segments (nine recent origins): WAPE / R3 for each member, the three-way's bias, best member")
def seg(col):
    rows = {}
    for v, x in R.groupby(col):
        e3 = (x.R3 - x.y).abs().sum(); r = {"items": len(x), "units %": x.y.sum() / R.y.sum() * 100}
        for c in ["rules", "qmodel", "nn", "lgb", "r_q_nn"]: r[c] = (x[c] - x.y).abs().sum() / e3
        r["bias 3way"] = x[S].sum() / x.y.sum() - 1; r["best"] = min(["rules", "qmodel", "nn", "lgb"], key=lambda c: r[c]); rows[v] = r
    return pd.DataFrame(rows).T
for col in ["age", "vol", "macro", "retail", "trend", "occ", "launch_season", "cnum", "oos", "dis_q", "ante", "target"]:
    print(f"\n-- {col}"); print(seg(col).round(3).to_string())
# which way is the truth when members disagree? share of the gap rules-vs-quantile that the actual closes
print("\n######## (3) when rules and quantile disagree by more than 30%: where does the actual land?")
d = R[R.rules_vs_q.abs() > np.log(1.3)].copy(); pos = (np.log(d.y.clip(lower=0.5)) - np.log(d.qmodel.clip(lower=0.5))) / d.rules_vs_q
print(f"items {len(d)} ({len(d) / len(R):.0%}); position of the actual on the quantile(0) -> rules(1) line: median {pos.median():.2f}, IQR {pos.quantile(.25):.2f}-{pos.quantile(.75):.2f}")
for col in ["age", "macro", "occ"]: print(f"  by {col}:", pos.groupby(d[col]).median().round(2).to_dict())
T.to_parquet("data/error_map_items.parquet", index=False)
