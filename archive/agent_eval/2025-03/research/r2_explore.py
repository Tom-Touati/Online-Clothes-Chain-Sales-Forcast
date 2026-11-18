"""March-25 exploration: stores, channels, new items, factors for the Mar-anchored origin."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 20)
from rules_v1 import load, seasonal_factors, lfl_chain
it, di, st = load()
tf = pd.read_csv("data/to_forecast.csv")
print("to_forecast", len(tf), "launch months:"); print(it.launch.reindex(tf.style_color).dt.to_period("M").value_counts().sort_index().tail(6))
print("not in scoreboard:", len(set(tf.style_color) - set(pd.read_csv("feedback/scoreboard.csv").style_color)))
# stores daily in March
s = st[st.date >= "2025-02-15"].pivot_table(index="date", columns="store", values="units", aggfunc="sum")
print(s.resample("W").mean().round(0))
print("FSM first day", st[st.store == "FSM"].date.min())
dc = pd.read_parquet("data/daily_channel.parquet")
w = dc[dc.date >= "2025-01-01"].pivot_table(index="date", columns="channel", values="units", aggfunc="sum").fillna(0)
print(w.resample("W").mean().round(0))
# factors for this origin
B = pd.Period("2025-03"); T = [B + h for h in range(1, 5)]
for ap in [0.0, 0.4]:
    f = seasonal_factors(it, di, st, B, T, anchor_pre=ap)
    print("anchor_pre", ap, "web", f["web"].round(3).to_dict(), "\n ret", f["ret"].round(3).to_dict(),
          "\n a_feb", {str(k): round(v, 3) for k, v in f["a_feb"].items()},
          "\n a_pre", {str(k): round(v, 3) for k, v in f["a_pre"].items()}, "cur_pd", round(f["cur_pd"], 3))
print("grp_rel\n", f["grp_rel"].round(2))
sr = f["seg_rel"].copy(); sr["n"] = f["seg_n"]
print(sr[sr.n >= 3].round(2).sort_values("n", ascending=False))
# all-units LFL chain Mar-24 -> Jul-24 per day and web vs other
ly = [pd.Period(x) for x in ["2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08"]]
d = di[di.m.isin(ly)]
for col in ["units", "web_units", "retail", "other"]:
    X = d.pivot_table(index="style_color", columns="m", values=col, aggfunc="sum").reindex(columns=ly).fillna(0)
    U = d.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").reindex(columns=ly).fillna(0)
    # LFL on items selling in earlier month (any channel), live >30d
    out = []
    for a, b in zip(ly[:-1], ly[1:]):
        ok = (it.launch.reindex(X.index) < a.start_time - pd.Timedelta(days=30)) & (U[a] > 0)
        out.append(round(X.loc[ok, b].sum() / max(X.loc[ok, a].sum(), 1) * a.days_in_month / b.days_in_month, 3))
    print(col, "LFL per-day mom 24:", dict(zip([str(x) for x in ly[1:]], out)))
# same for 2025 Jan->Feb->Mar
ly2 = [pd.Period(x) for x in ["2024-12", "2025-01", "2025-02", "2025-03"]]
d = di[di.m.isin(ly2)]
for col in ["units", "web_units", "retail", "other"]:
    X = d.pivot_table(index="style_color", columns="m", values=col, aggfunc="sum").reindex(columns=ly2).fillna(0)
    U = d.pivot_table(index="style_color", columns="m", values="units", aggfunc="sum").reindex(columns=ly2).fillna(0)
    out = []
    for a, b in zip(ly2[:-1], ly2[1:]):
        ok = (it.launch.reindex(X.index) < a.start_time - pd.Timedelta(days=30)) & (U[a] > 0)
        out.append(round(X.loc[ok, b].sum() / max(X.loc[ok, a].sum(), 1) * a.days_in_month / b.days_in_month, 3))
    print(col, "LFL per-day mom 25:", dict(zip([str(x) for x in ly2[1:]], out)))
