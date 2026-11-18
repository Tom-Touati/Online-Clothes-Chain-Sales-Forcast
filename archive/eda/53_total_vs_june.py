"""(a) Decompose the 2026 July/June gap: July/June = (Jul-Oct / June) x July share.
(b) Test a same-mix benchmark for the 4-month total: prior-year Jul-Oct/June ratio by age group, weighted by the current June mix.
    Backtest: 2024 ratios on the 2025 mix against actual 2025. Then the 2026 forecast against the 2025-ratio benchmark,
    corrected by the benchmark's own 2025 error, per group. Hand-rule items (World Cup, S16079) excluded."""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding="utf-8")
O = pd.read_csv("data/final_forecast_q4.csv", index_col=0); M = pd.read_parquet("data/monthly_item.parquet")
it = pd.read_parquet("data/items.parquet").set_index("style_color")
start = pd.concat([pd.to_datetime(it.published_date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None), pd.to_datetime(it.first_sale)], axis=1).min(axis=1)
grp = lambda a: np.where(a < 3, "<3m", np.where(a < 12, "3-12m", "12m+"))
mo = lambda y: [f"{y}-{m:02d}" for m in (7, 8, 9, 10)]
def frame(y):
    age = (pd.Timestamp(f"{y}-06-30") - start.reindex(M.index)).dt.days / 30.44
    ids = M.index[(M[f"{y}-06"] >= 50).values & (age >= 1).values]
    return pd.DataFrame({"g": grp(age[ids].values), "jun": M.loc[ids, f"{y}-06"].values, "jul": M.loc[ids, f"{y}-07"].values, "tot": M.loc[ids, mo(y)].sum(axis=1).values}, index=ids)
d24, d25 = frame(2024), frame(2025)
for y, d in [(2024, d24), (2025, d25)]:
    print(f"{y} actual, all items live in June: Jul-Oct/June {d.tot.sum()/d.jun.sum():.3f} x July share {d.jul.sum()/d.tot.sum():.3f} = July/June {d.jul.sum()/d.jun.sum():.3f}")
jun = M.loc[O.index, "2026-06"]
print(f"2026 submitted, 100 targets:       Jul-Oct/June {O.final_total.sum()/jun.sum():.3f} x July share {O.f_jul.sum()/O.final_total.sum():.3f} = July/June {O.f_jul.sum()/jun.sum():.3f}")
RULE = ["S24840_LIGHT-GRAY-MELANGE", "S24847_NAVY", "S25039_YELLOW", "S16079_DARK-GRAY"]
f = O.loc[~O.index.isin(RULE)]; fj = jun[f.index]; fg = grp(f.age_m.values)
print(f"2026 submitted, targets without hand rule (n={len(f)}): Jul-Oct/June {f.final_total.sum()/fj.sum():.3f}, July share {f.f_jul.sum()/f.final_total.sum():.3f}")
r24 = d24.groupby("g").tot.sum() / d24.groupby("g").jun.sum(); r25 = d25.groupby("g").tot.sum() / d25.groupby("g").jun.sum()
print("\n== benchmark backtest on 2025 (2024 Jul-Oct/June by age group on the 2025 June mix)")
for g in ["<3m", "3-12m", "12m+"]:
    e = r24[g] / r25[g] - 1; print(f"  {g:6s} 2024 ratio {r24[g]:.3f} | 2025 actual {r25[g]:.3f} | benchmark error {e:+.1%}")
w25 = d25.groupby("g").jun.sum(); b = (w25 * r24).sum() / w25.sum(); a = d25.tot.sum() / d25.jun.sum()
print(f"  all: benchmark {b:.3f} vs actual {a:.3f} -> error {b/a-1:+.1%}")
print("\n== 2026 targets without hand rule, by age group: forecast Jul-Oct/June against 2025 ratios")
rows = []
for g in ["<3m", "3-12m", "12m+"]:
    m = fg == g
    if m.sum() == 0: continue
    fr = f.final_total[m].sum() / fj[m].sum(); err = r24[g] / r25[g] - 1
    rows.append(dict(group=g, n=int(m.sum()), june=int(fj[m].sum()), forecast=round(fr, 3), r2025=round(r25[g], 3), r2024=round(r24[g], 3),
                     fc_vs_2025=f"{fr/r25[g]-1:+.1%}", fc_vs_2024=f"{fr/r24[g]-1:+.1%}", yoy_change_of_ratio_2024_to_2025=f"{-err/(1+err):+.1%}"))
print(pd.DataFrame(rows).to_string(index=False))
w26 = pd.Series(fj.values, index=fg).groupby(level=0).sum()
for nm, r in [("2025 ratios", r25), ("2024 ratios", r24)]:
    print(f"benchmark on 2026 mix with {nm}: {(w26 * r.reindex(w26.index)).sum() / w26.sum():.3f} | submitted {f.final_total.sum()/fj.sum():.3f}")
