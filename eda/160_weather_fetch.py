"""D155: daily weather for every US state (user request: "import temperature information in each region").
Source: Open-Meteo historical archive API (ERA5 reanalysis; free, no key), https://archive-api.open-meteo.com/v1/archive.
Each state is its largest metro (two or three metros, population-weighted, for CA, TX, FL, NY, PA, IL, OH, GA, NC, WA); DC and PR too.
Daily mean and max temperature (deg C) and precipitation (mm), 2014-01-01 .. 2026-06-30, local time America/New_York days.
Writes data/weather_state_daily.parquet (date, state, tmean, tmax, precip) and data/weather_state_month.parquet with the month's
mean, the 2014-2023 normal for that calendar month, and the anomaly. Only coordinates are sent to the service."""
import sys, time, requests, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
PTS = [  # state, lat, lon, weight within state
    ("Alabama", 33.52, -86.80, 1), ("Alaska", 61.22, -149.90, 1), ("Arizona", 33.45, -112.07, 1), ("Arkansas", 34.75, -92.29, 1),
    ("California", 34.05, -118.24, 0.45), ("California", 37.77, -122.42, 0.25), ("California", 32.72, -117.16, 0.10), ("California", 38.58, -121.49, 0.10), ("California", 36.74, -119.79, 0.10),
    ("Colorado", 39.74, -104.99, 1), ("Connecticut", 41.76, -72.67, 1), ("Delaware", 39.74, -75.55, 1), ("District of Columbia", 38.90, -77.04, 1),
    ("Florida", 25.76, -80.19, 0.45), ("Florida", 28.54, -81.38, 0.25), ("Florida", 27.95, -82.46, 0.20), ("Florida", 30.33, -81.66, 0.10),
    ("Georgia", 33.75, -84.39, 0.8), ("Georgia", 32.08, -81.09, 0.2), ("Hawaii", 21.31, -157.86, 1), ("Idaho", 43.62, -116.20, 1),
    ("Illinois", 41.88, -87.63, 0.8), ("Illinois", 39.78, -89.65, 0.2), ("Indiana", 39.77, -86.16, 1), ("Iowa", 41.59, -93.62, 1), ("Kansas", 37.69, -97.34, 1),
    ("Kentucky", 38.25, -85.76, 1), ("Louisiana", 29.95, -90.07, 1), ("Maine", 43.66, -70.26, 1), ("Maryland", 39.29, -76.61, 1), ("Massachusetts", 42.36, -71.06, 1),
    ("Michigan", 42.33, -83.05, 0.7), ("Michigan", 42.96, -85.67, 0.3), ("Minnesota", 44.98, -93.27, 1), ("Mississippi", 32.30, -90.18, 1), ("Missouri", 38.63, -90.20, 0.5), ("Missouri", 39.10, -94.58, 0.5),
    ("Montana", 45.78, -108.50, 1), ("Nebraska", 41.26, -95.93, 1), ("Nevada", 36.17, -115.14, 1), ("New Hampshire", 42.99, -71.46, 1), ("New Jersey", 40.74, -74.17, 1),
    ("New Mexico", 35.08, -106.65, 1), ("New York", 40.71, -74.01, 0.75), ("New York", 42.89, -78.88, 0.10), ("New York", 43.05, -76.15, 0.15),
    ("North Carolina", 35.23, -80.84, 0.6), ("North Carolina", 35.78, -78.64, 0.4), ("North Dakota", 46.88, -96.79, 1),
    ("Ohio", 39.96, -83.00, 0.35), ("Ohio", 41.50, -81.69, 0.35), ("Ohio", 39.10, -84.51, 0.30), ("Oklahoma", 35.47, -97.52, 1), ("Oregon", 45.52, -122.68, 1),
    ("Pennsylvania", 39.95, -75.17, 0.65), ("Pennsylvania", 40.44, -80.00, 0.35), ("Rhode Island", 41.82, -71.41, 1), ("South Carolina", 34.85, -82.40, 0.5), ("South Carolina", 32.78, -79.93, 0.5),
    ("South Dakota", 43.55, -96.73, 1), ("Tennessee", 36.16, -86.78, 0.6), ("Tennessee", 35.15, -90.05, 0.4),
    ("Texas", 29.76, -95.37, 0.35), ("Texas", 32.78, -96.80, 0.35), ("Texas", 29.42, -98.49, 0.15), ("Texas", 30.27, -97.74, 0.15),
    ("Utah", 40.76, -111.89, 1), ("Vermont", 44.48, -73.21, 1), ("Virginia", 36.85, -76.29, 0.5), ("Virginia", 38.88, -77.10, 0.5),
    ("Washington", 47.61, -122.33, 0.85), ("Washington", 47.66, -117.43, 0.15), ("West Virginia", 38.35, -81.63, 1), ("Wisconsin", 43.04, -87.91, 1), ("Wyoming", 41.14, -104.82, 1),
    ("Puerto Rico", 18.47, -66.11, 1),
]
URL = "https://archive-api.open-meteo.com/v1/archive"; out = []
import os; os.makedirs("data/weather_raw", exist_ok=True); BS = 5          # resumable: each batch of 5 points is cached
for b in range(0, len(PTS), BS):
    ch = PTS[b:b + BS]; cache = f"data/weather_raw/batch_{b:03d}.parquet"
    if os.path.exists(cache): out.append(pd.read_parquet(cache)); continue
    for attempt in range(12):
        try:
            r = requests.get(URL, params=dict(latitude=",".join(str(p[1]) for p in ch), longitude=",".join(str(p[2]) for p in ch), start_date="2014-01-01", end_date="2026-06-30",
                                              daily="temperature_2m_mean,temperature_2m_max,precipitation_sum", timezone="America/New_York"), timeout=180)
            if r.status_code == 200: break
            print("retry", b, r.status_code, r.text[:120], flush=True)
        except requests.exceptions.RequestException as e: print("retry", b, type(e).__name__, flush=True)
        time.sleep(30 + 15 * attempt)
    else: raise SystemExit(f"batch {b} failed; rerun to resume")
    js = r.json(); js = js if isinstance(js, list) else [js]; part = []
    for (st, la, lo, w), j in zip(ch, js):
        d = j["daily"]; part.append(pd.DataFrame({"date": pd.to_datetime(d["time"]), "state": st, "w": w, "tmean": d["temperature_2m_mean"], "tmax": d["temperature_2m_max"], "precip": d["precipitation_sum"]}))
    part = pd.concat(part); part.to_parquet(cache, index=False); out.append(part)
    print("fetched", b + len(ch), "of", len(PTS), flush=True); time.sleep(35)
W = pd.concat(out)
for c in ["tmean", "tmax", "precip"]: W[c + "_w"] = W[c] * W.w
D = W.groupby(["date", "state"]).agg(tmean_w=("tmean_w", "sum"), tmax_w=("tmax_w", "sum"), precip_w=("precip_w", "sum"), w=("w", "sum")).reset_index()
for c in ["tmean", "tmax", "precip"]: D[c] = D[c + "_w"] / D.w
D = D[["date", "state", "tmean", "tmax", "precip"]]; D.to_parquet("data/weather_state_daily.parquet", index=False)
D["m"] = D.date.dt.to_period("M").astype(str); D["cm"] = D.date.dt.month
M = D.groupby(["m", "state"]).agg(tmean=("tmean", "mean"), tmax=("tmax", "mean"), precip=("precip", "sum"), cm=("cm", "first")).reset_index()
N = M[(M.m >= "2014-01") & (M.m <= "2023-12")].groupby(["state", "cm"])[["tmean", "tmax", "precip"]].mean().add_suffix("_norm").reset_index()
M = M.merge(N, on=["state", "cm"]); M["tanom"] = M.tmean - M.tmean_norm; M.to_parquet("data/weather_state_month.parquet", index=False)
print(D.shape, M.shape); print(M[M.m >= "2025-09"].groupby("m")[["tmean", "tmean_norm", "tanom"]].mean().round(1).to_string())
