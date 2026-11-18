"""D155: monthly state temperature from NOAA NCEI Climate at a Glance (statewide average and maximum temperature, area-weighted,
deg F converted to deg C), 2014-01 .. 2026-06, for the 48 contiguous states (codes 1-48) and Alaska (50). Replaces the Open-Meteo
city fetch (eda/160), which hit that service's hourly limit; Hawaii and DC come from the Open-Meteo daily data already cached
(data/weather_raw/, Honolulu and Washington DC), averaged by month. Normals = the 2014-2023 mean of each calendar month.
Writes data/weather_state_month.parquet: m, state, tmean, tmax, cm, tmean_norm, tmax_norm, tanom. Nothing but the URL is sent."""
import sys, glob, io, re, time, requests, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
BASE = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{code}/{var}/1/0/2014-2026/data.csv"
rows = []
for code in list(range(1, 49)) + [50]:
    rec = {}
    for var in ["tavg", "tmax"]:
        for attempt in range(5):
            try:
                r = requests.get(BASE.format(code=code, var=var), timeout=60)
                if r.status_code == 200: break
            except requests.exceptions.RequestException: pass
            time.sleep(5 * (attempt + 1))
        name = re.search(r"#\s*(?:&quot;|\")?(.+?) (?:Average|Maximum) Temperature", r.text).group(1).strip()
        d = pd.read_csv(io.StringIO(r.text), comment="#"); d = d[d.Value.notna()]
        rec[var] = pd.Series(((d.Value.astype(float) - 32) / 1.8).values, index=pd.PeriodIndex(d.Date.astype(str), freq="M").astype(str))
    rows.append(pd.DataFrame({"m": rec["tavg"].index, "state": name, "tmean": rec["tavg"].values, "tmax": rec["tmax"].reindex(rec["tavg"].index).values}))
    print(code, name, len(rec["tavg"]), flush=True); time.sleep(0.3)
N = pd.concat(rows)
OM = pd.concat([pd.read_parquet(p) for p in glob.glob("data/weather_raw/batch_*.parquet")])
OM = OM[OM.state.isin(["Hawaii", "District of Columbia"])]
OM["m"] = OM.date.dt.to_period("M").astype(str)
OM = OM.assign(tw=OM.tmean * OM.w, xw=OM.tmax * OM.w).groupby(["m", "state"]).agg(tw=("tw", "sum"), xw=("xw", "sum"), w=("w", "sum")).reset_index()
OM["tmean"] = OM.tw / OM.w; OM["tmax"] = OM.xw / OM.w
M = pd.concat([N, OM[["m", "state", "tmean", "tmax"]]], ignore_index=True)
M = M[(M.m >= "2014-01") & (M.m <= "2026-06")]; M["cm"] = M.m.str[-2:].astype(int)
NR = M[M.m <= "2023-12"].groupby(["state", "cm"])[["tmean", "tmax"]].mean().add_suffix("_norm").reset_index()
M = M.merge(NR, on=["state", "cm"]); M["tanom"] = M.tmean - M.tmean_norm
M.to_parquet("data/weather_state_month.parquet", index=False)
print(M.state.nunique(), "states |", M.shape); print(M[M.m >= "2025-07"].groupby("m")[["tmean", "tmean_norm", "tanom"]].mean().round(2).to_string())
