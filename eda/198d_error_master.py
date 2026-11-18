"""D276 (fork): master table for the error analysis of the SUBMITTED system (candidate_nx2 = E:ecnsnn_x_geo, D273/D274).
One row per backtest item-origin (13 origins, 11,727 rows) and one per item at the June 2026 origin, with
  forecasts   F (submitted system = geo-corrected stack), F0 (stack before corrections), the five members, R3, y
  cohort axes known at the origin (age, volume, type, look, colour, channels, discount, size run, stores, partners, siblings,
              momentum, region skew, history length, model spread, forecast direction, interval width, stock flags ...)
  hindsight   what happened in the target window that the business could have planned or read early: web discount change, own-store
              and partner distribution change, size-run and stock-flag share, launches of the same style / look / type, month-1 ratio,
              the largest-day share (a spike), channel shifts
Writes data/fk_err_master.parquet (backtest) and data/fk_err_master_2026.parquet (June 2026, origin-known axes only)."""
import sys, glob, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
P = lambda o, k: str(pd.Period(o, "M") + k)
A = pd.read_parquet("data/fk_axes_month.parquet"); A = A.set_index(["style_color", "m"])
it = pd.read_parquet("data/items.parquet").set_index("style_color")
M = pd.read_parquet("data/monthly_item.parquet"); M.columns = [str(c) for c in M.columns]
att = pd.read_csv("item_attributes.csv").set_index("style_color")
L = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("data/llm/scores_part*.csv"))]).drop_duplicates("style_color").set_index("style_color")
FE = pd.read_parquet("data/features.parquet"); FE = FE[FE.h == 1].set_index(["origin", "style_color"])
RG = pd.read_parquet("data/region_feats.parquet").set_index(["origin", "style_color"])
OOS = pd.read_parquet("data/oos_flags_daily.parquet"); OOS.columns = pd.to_datetime(OOS.columns)
first = pd.to_datetime(it.first_sale); pub = pd.to_datetime(it.published_date, utc=True).dt.tz_localize(None)
sty = pd.Series(it.index.str.split("_").str[0], index=it.index)
site_disc = (A.disc_web * A.u_edikted_com).groupby(level=1).sum() / A.u_edikted_com.groupby(level=1).sum()


def win(col, items, o, ks, how="sum"):
    X = np.column_stack([A[col].reindex(pd.MultiIndex.from_arrays([items, [P(o, k)] * len(items)])).values for k in ks]).astype(float)
    if how == "sum": return np.nansum(X, 1)
    if how == "mean": return np.nanmean(np.where(np.isfinite(X), X, np.nan), 1)
    if how == "max": return np.nanmax(np.where(np.isfinite(X), X, -np.inf), 1)


def axes(D, o, hindsight=True):
    s = D.style_color.values; n = len(s); out = {}
    B3, T4 = [-2, -1, 0], [1, 2, 3, 4]
    with np.errstate(all="ignore"):
        u3 = win("units", s, o, B3); out["cancel_rate"] = win("u_cancel", s, o, B3) / np.maximum(u3 + win("u_cancel", s, o, B3), 1)
        for c, nm in [("u_tiktok", "sh_tiktok"), ("uc_retail", "sh_retail"), ("uc_drop", "sh_partner"), ("u_edikted_com", "sh_web")]:
            out[nm] = win(c, s, o, B3) / np.maximum(u3, 1)
        out["n_stores"] = np.nan_to_num(win("n_stores", s, o, [0], "max"), neginf=0); out["n_partners"] = np.nan_to_num(win("n_partners", s, o, [0], "max"), neginf=0)
        out["stores_mom"] = out["n_stores"] - np.nan_to_num(win("n_stores", s, o, [-3], "max"), neginf=0)
        life_sz = A.n_sizes.groupby(level=0).max().reindex(s).values
        out["n_sizes_life"] = life_sz; out["size_run"] = np.nan_to_num(win("n_sizes", s, o, [0], "max"), neginf=0) / np.maximum(life_sz, 1)
        out["size_hhi"] = win("size_hhi", s, o, [0], "mean"); out["state_hhi"] = win("state_hhi", s, o, B3, "mean")
        out["top_day_share"] = win("top_day_share", s, o, [0], "mean"); out["top_hour_share"] = win("top_hour_share", s, o, [0], "mean")
        out["multi_share"] = win("multi_share", s, o, B3, "mean"); out["wkend_share"] = win("wkend_share", s, o, B3, "mean")
        d0 = win("disc_web", s, o, [0], "mean"); d12 = win("disc_web", s, o, [-2, -1], "mean"); out["disc_web"] = d0; out["disc_mom"] = d0 - d12
        out["disc_vs_site"] = d0 - site_disc.reindex([o]).values[0]
        tk0 = win("u_tiktok", s, o, [0]) / np.maximum(win("units", s, o, [0]), 1); tk1 = win("u_tiktok", s, o, [-2, -1]) / np.maximum(win("units", s, o, [-2, -1]), 1); out["tiktok_mom"] = tk0 - tk1
        e = pd.Period(o, "M").end_time.normalize(); d28 = [c for c in OOS.columns if e - pd.Timedelta(days=27) <= c <= e]
        out["oos_28"] = OOS.reindex(s)[d28].mean(1).values if d28 else np.nan
        # siblings in the style, alive in the last 3 months
        live = (M.reindex(columns=[P(o, k) for k in B3]).fillna(0).sum(1) > 0); lv = live[live].index
        cnt = sty.reindex(lv).value_counts(); out["n_siblings"] = cnt.reindex(sty.reindex(s).values).fillna(1).values
        su = M.reindex(columns=[P(o, k) for k in B3]).fillna(0).sum(1).groupby(sty.reindex(M.index).values).sum()
        out["share_of_style"] = u3 / np.maximum(su.reindex(sty.reindex(s).values).values, 1)
        newest = first.groupby(sty.reindex(first.index).values).transform("max"); out["newest_colour"] = (first.reindex(s).values == newest.reindex(s).values).astype(float)
        out["hist_months"] = np.minimum((pd.Period(o, "M") - pd.Period("2024-01", "M")).n + 1, [((pd.Period(o, "M") - x.to_period("M")).n + 1) for x in first.reindex(s)])
        out["pre2024"] = (pub.reindex(s).values < np.datetime64("2024-01-01")).astype(float)
        out["has_ly"] = (M.reindex(s).reindex(columns=[P(o, k) for k in [-11, -10, -9, -8]]).fillna(0).sum(1).values > 0).astype(float)
        out["launch_month"] = first.reindex(s).dt.month.values
        if hindsight:
            dT = win("disc_web", s, o, T4, "mean"); d3 = win("disc_web", s, o, B3, "mean"); sT = site_disc.reindex([P(o, k) for k in T4]).mean(); s3 = site_disc.reindex([P(o, k) for k in B3]).mean()
            out["h_disc_chg"] = dT - d3; out["h_disc_chg_rel"] = (dT - d3) - (sT - s3)
            out["h_stores_chg"] = np.nan_to_num(win("n_stores", s, o, T4, "max"), neginf=0) - np.nan_to_num(win("n_stores", s, o, B3, "max"), neginf=0)
            out["h_partners_chg"] = np.nan_to_num(win("n_partners", s, o, T4, "max"), neginf=0) - np.nan_to_num(win("n_partners", s, o, B3, "max"), neginf=0)
            out["h_size_run"] = np.nan_to_num(win("n_sizes", s, o, T4, "mean")) / np.maximum(life_sz, 1)
            dw = [c for c in OOS.columns if e < c <= (pd.Period(o, "M") + 4).end_time.normalize()]; out["h_oos"] = OOS.reindex(s)[dw].mean(1).values if dw else np.nan
            out["h_m1"] = M.reindex(s).reindex(columns=[P(o, 1)]).fillna(0).values[:, 0]
            out["h_top_day"] = win("top_day_share", s, o, T4, "max")
            uT = np.maximum(win("units", s, o, T4), 1)
            for c, nm in [("u_tiktok", "h_tiktok_chg"), ("uc_retail", "h_retail_chg"), ("uc_drop", "h_partner_chg")]: out[nm] = win(c, s, o, T4) / uT - win(c, s, o, B3) / np.maximum(u3, 1)
            fm = first.dt.to_period("M").astype(str); new = fm[(fm > o) & (fm <= P(o, 4))].index
            lk = D.look.values; look_of = pd.Series(D.look.values, index=s)
            out["h_new_style"] = sty.reindex(new).value_counts().reindex(sty.reindex(s).values).fillna(0).values
            ty = it.product_type.reindex(new).value_counts(); out["h_new_type"] = ty.reindex(it.product_type.reindex(s).values).fillna(0).values
            out["h_dead"] = (M.reindex(s).reindex(columns=[P(o, 4)]).fillna(0).values[:, 0] == 0).astype(float)
    return pd.DataFrame(out, index=D.index)


def build(base, geo, hindsight):
    B = pd.read_parquet(base); G = pd.read_parquet(geo).set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([B.origin, B.style_color]))
    B = B.rename(columns={"F": "F0"}); B["F"] = G.geo.values; B["m_seg"] = G["vol_age_disc_0.5n"].values / G.F.values; B["m_brk"] = G["res_all_0.5n"].values / G.F.values
    B = B[B.F.notna()].reset_index(drop=True)
    X = pd.concat([axes(g, o, hindsight) for o, g in B.groupby("origin")]).reindex(B.index); B = pd.concat([B, X], axis=1)
    k = pd.MultiIndex.from_arrays([B.origin, B.style_color])
    for c in ["sh_ret", "sh_drop", "disc_rel", "peak_r", "lagr1", "lagr3", "color_num", "days_live", "a_occasion_tier", "a_weight", "a_silhouette"]: B[c] = FE[c].reindex(k).values if c in FE else np.nan
    for c in ["clim_warm_rel", "clim_cold_rel"]: B[c] = RG[c].reindex(k).values
    B["colour"] = it.color.reindex(B.style_color).values; B["collection"] = it.product_collection.reindex(B.style_color).values; B["occasion"] = it.occasion.reindex(B.style_color).fillna("none").values
    B["colour_family"] = att.color_primary.reindex(B.style_color).fillna("na").values; B["aesthetic"] = att.aesthetics.reindex(B.style_color).fillna("na").astype(str).str.split("|").str[0].str.strip().values
    for c in ["summer", "winter", "party", "statement", "staple", "formality", "warmth"]: B[c] = L[c].reindex(B.style_color).values
    return B


BT = build("data/dec_base_ecnsnn.parquet", "data/geo_ecnsnn_x_preds.parquet", True)
q = pd.read_parquet("data/q_backtest_fkconf_pfn2dyn_seg_tails.parquet").set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([BT.origin, BT.style_color]))
BT["p10"] = q.q10.values; BT["p90"] = q.q90.values
for nm, f, c in [("s_rq", "data/fixes_items.parquet", "r_q_nn"), ("s_wx3bf", "data/fk_stack_preds.parquet", "wx3_bf"), ("s_dyn3bf", "data/fk_stack_preds.parquet", "dyn3_bf"), ("s_p2", "data/fk_stack_arith_preds.parquet", "p2"),
                  ("s_p2c32a", "data/fk_stack_arith_preds.parquet", "p2c32a"), ("s_ecns", "data/dec_base_ec_ns.parquet", "F"), ("s_ecns_geo", "data/geo_ec_ns_x_preds.parquet", "geo")]:
    BT[nm] = pd.read_parquet(f).set_index(["origin", "style_color"])[c].reindex(pd.MultiIndex.from_arrays([BT.origin, BT.style_color])).values
BT.to_parquet("data/fk_err_master.parquet", index=False)
J = build("data/dec_base_ecnsnn26.parquet", "data/geo_ecnsnn_x_2026.parquet", False); J.to_parquet("data/fk_err_master_2026.parquet", index=False)
rec = BT[BT.origin >= "2025-06"]
print(BT.shape, J.shape, "| recent WAPE/R3 submitted", round((rec.F - rec.y).abs().sum() / (rec.R3 - rec.y).abs().sum(), 4), "stack", round((rec.F0 - rec.y).abs().sum() / (rec.R3 - rec.y).abs().sum(), 4),
      "| all 13", round((BT.F - BT.y).abs().sum() / (BT.R3 - BT.y).abs().sum(), 4))
print(BT.isna().mean().round(2)[lambda s: s > 0.02].to_string())
