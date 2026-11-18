"""Shared helpers for the fork's range attempts (D370-D379; Fork 2's delegation of five attempts to improve the forecast ranges).
  load()        the backtest table: the candidate system's P50 (Main's D364 file data/system_q2575_backtest.parquet, the D358 / nx5 system),
                the actuals, R3, the item descriptors of data/fk_err_master.parquet, the age x volume segment of D328, and the conformal TabPFN
                member's own log widths (data/fk_member_quantiles.parquet, as D328 / D329)
  train_mask()  which origins may train a model or a calibration for test origin o:
                  'loo'     every other origin (as the adopted D328 / D366 calibration)
                  'purged'  origins at least 4 months away (their 4-month windows do not overlap the test window)
                  'honest'  origins whose window had closed by o (o' + 4 <= o)
  score()       coverage (below / inside / above), pinball loss in units as a ratio to R3's (R3 used as every quantile), the log interval
                score, the mean log width, and coverage by segment, for one pair of bounds on a set of rows."""
import numpy as np, pandas as pd

AGE = ["under 3 months", "3-6 months", "6-12 months", "over 12 months"]


def load():
    S = pd.read_parquet("data/system_q2575_backtest.parquet")
    D = pd.read_parquet("data/fk_err_master.parquet").drop(columns=["y"])
    B = S.merge(D, on=["origin", "style_color"], how="left", validate="1:1").reset_index(drop=True)
    assert len(B) == len(S) == 11727 and B.R3.notna().all()
    B["age_b"] = pd.cut(B.age_m, [-1, 3, 6, 12, 999], labels=AGE).astype(str)
    B["vol_b"] = B.groupby("origin")["level"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])).astype(str)
    B["seg"] = B.age_b + " | " + B.vol_b
    Q = pd.read_parquet("data/fk_member_quantiles.parquet"); Q = Q[Q.member == "pf2c"].set_index(["origin", "style_color"]).reindex(pd.MultiIndex.from_arrays([B.origin, B.style_color]))
    l50 = np.log(Q.q50.values + 1)
    for c in ["q10", "q25", "q75", "q90"]: B["pw_" + c] = np.maximum(np.abs(np.log(Q[c].values + 1) - l50), 0.02)
    B["oo"] = pd.PeriodIndex(B.origin, freq="M").asi8
    B["lP"] = np.log(B.P50.values + 1); B["ly"] = np.log(B.y.values + 1); B["e"] = B.ly - B.lP
    return B


def train_mask(B, o, how):
    oo = pd.Period(o).ordinal; d = B.oo.values - oo
    if how == "loo": return d != 0
    if how == "purged": return np.abs(d) >= 4
    if how == "honest": return d <= -4
    raise ValueError(how)


def pin(y, q, tau):
    u = y - q; return np.maximum(tau * u, (tau - 1) * u)


def score(B, lo, hi, a, m, tag=None):
    """lo / hi: log bounds (log(units + 1)); a = 0.5 for P25-P75, 0.2 for P10-P90; m: rows to score."""
    ly, y, R3 = B.ly.values[m], B.y.values[m], B.R3.values[m]; lo, hi = lo[m], hi[m]
    ok = np.isfinite(lo) & np.isfinite(hi)
    below, above = ly < lo, ly > hi; t = a / 2
    ql, qh = np.exp(lo) - 1, np.exp(hi) - 1
    pr = (pin(y, ql, t).sum() + pin(y, qh, 1 - t).sum()) / (pin(y, R3, t).sum() + pin(y, R3, 1 - t).sum())
    isc = (hi - lo) + (2 / a) * np.where(below, lo - ly, 0) + (2 / a) * np.where(above, ly - hi, 0)
    seg = pd.Series(~below & ~above).groupby(B.seg.values[m]).mean()
    return dict(n=int(m.sum()), rows_ok=int(ok.sum()), below=round(float(below.mean()), 3), inside=round(float((~below & ~above).mean()), 3), above=round(float(above.mean()), 3),
                pinball_r3=round(float(pr), 4), int_score=round(float(isc.mean()), 3), log_width=round(float((hi - lo).mean()), 3),
                seg_min=round(float(seg.min()), 3), seg_max=round(float(seg.max()), 3), seg_sd=round(float(seg.std()), 3))


def factors(e, wl, wu, tail):
    """Per-side scale so that a share `tail` of e falls below -kl*wl and above ku*wu."""
    return float(np.quantile(-e / wl, 1 - tail)), float(np.quantile(e / wu, 1 - tail))
