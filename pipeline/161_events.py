"""D156: US special-events calendar, known in advance. For each month 2023-01 .. 2026-12: the number of days of each event's
shopping window that fall in the month. Windows (shopping happens before the day):
  valentine   Feb 1-14            spring_break  Mar 1-31 (peak weeks)   easter   10 days to Easter Sunday
  festival    Apr 1-25 (Coachella weekends and the festival look)        prom     Apr 1 - May 20
  mothers     10 days to Mother's Day      memorial  Memorial Day weekend (Fri-Mon, sales)
  graduation  May 1 - Jun 15      july4      Jun 25 - Jul 4      prime_day  Amazon Prime Day sale days (competing site event)
  bts         back-to-school Jul 20 - Sep 7      labor_day  Labor Day weekend (Fri-Mon)
  halloween   Oct 1-31            bfcm       Thanksgiving + 1 .. + 4 (Black Friday to Cyber Monday)
  holiday     Dec 1-24 (holiday parties, gifting)   nye   Dec 26-31
Writes data/events_month.parquet (m, one column per event)."""
import sys, numpy as np, pandas as pd
from datetime import date, timedelta
sys.stdout.reconfigure(encoding="utf-8")
EASTER = {2023: date(2023, 4, 9), 2024: date(2024, 3, 31), 2025: date(2025, 4, 20), 2026: date(2026, 4, 5)}
PRIME = {2023: [date(2023, 7, 11), date(2023, 7, 12)], 2024: [date(2024, 7, 16), date(2024, 7, 17)], 2025: [date(2025, 7, 8) + timedelta(i) for i in range(4)],
         2026: [date(2026, 7, 14) + timedelta(i) for i in range(4)]}   # 2026 dates: assumed mid-July, as in past years
def nth_wd(y, m, wd, n):   # n-th weekday (0 = Monday) of month; n = -1 for the last
    d = date(y, m, 1)
    if n > 0: d += timedelta((wd - d.weekday()) % 7 + 7 * (n - 1)); return d
    d = date(y, m + 1, 1) - timedelta(1); return d - timedelta((d.weekday() - wd) % 7)
def rng(a, b): return [a + timedelta(i) for i in range((b - a).days + 1)]
days = {}
for y in range(2023, 2027):
    thanks = nth_wd(y, 11, 3, 4); mem = nth_wd(y, 5, 0, -1); lab = nth_wd(y, 9, 0, 1); mom = nth_wd(y, 5, 6, 2)
    ev = {"valentine": rng(date(y, 2, 1), date(y, 2, 14)), "spring_break": rng(date(y, 3, 1), date(y, 3, 31)), "easter": rng(EASTER[y] - timedelta(9), EASTER[y]),
          "festival": rng(date(y, 4, 1), date(y, 4, 25)), "prom": rng(date(y, 4, 1), date(y, 5, 20)), "mothers": rng(mom - timedelta(9), mom),
          "memorial": rng(mem - timedelta(3), mem), "graduation": rng(date(y, 5, 1), date(y, 6, 15)), "july4": rng(date(y, 6, 25), date(y, 7, 4)), "prime_day": PRIME[y],
          "bts": rng(date(y, 7, 20), date(y, 9, 7)), "labor_day": rng(lab - timedelta(3), lab), "halloween": rng(date(y, 10, 1), date(y, 10, 31)),
          "bfcm": rng(thanks + timedelta(1), thanks + timedelta(4)), "holiday": rng(date(y, 12, 1), date(y, 12, 24)), "nye": rng(date(y, 12, 26), date(y, 12, 31))}
    for k, ds in ev.items(): days.setdefault(k, []).extend(ds)
months = pd.period_range("2023-01", "2026-12", freq="M")
E = pd.DataFrame({"m": months.astype(str)})
for k, ds in days.items():
    c = pd.Series(pd.PeriodIndex([pd.Timestamp(d) for d in ds], freq="M").astype(str)).value_counts()
    E[k] = E.m.map(c).fillna(0).astype(int)
E.to_parquet("data/events_month.parquet", index=False); print(E[E.m.str.startswith("2025")].to_string(index=False))
