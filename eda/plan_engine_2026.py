"""D321 (fork; proposals #1b and #7 of the error study, assigned in D314): the plan-ready engine. Turns Edikted's plan for July-October 2026
into a forecast, by the known-future tree member of D309 (Main's 'ec_ns' + every business decision of the window), fitted at the June 2026
origin on every origin whose window has closed (up to 2026-02, realised decisions).
  plan columns (one row per item; a blank means "no change"):
    style_color, planned_own_stores (max own stores selling it in Jul-Oct), planned_partners (max partners), planned_discount_change
    (web discount Jul-Oct minus Apr-Jun, as a fraction, e.g. 0.05), planned_site_discount_change (the same for the whole site; default 0),
    planned_new_colours (new colours of the item's style launched Jul-Oct), planned_same_type_launches (new items of the same product type
    launched Jul-Oct; default: what happened in July-October 2025)
  output: the system with the plan-aware tree member in place of its tree member, as a multiplier on the submitted forecast:
    plan forecast = submitted total x system(plan) / system(no change)
  so the hand rules and corrections of the submitted workbook are kept and only the plan's effect is added.
Usage:  python eda/plan_engine_2026.py <plan.csv> [out.csv]      (writes the template with  --template data/plan_template_2026.csv)
Import: E = Engine(); E.forecast(plan_df) -> DataFrame."""
import sys, numpy as np, pandas as pd, openpyxl
sys.stdout.reconfigure(encoding="utf-8")
O26 = "2026-06"; WB = "Edikted_Forecasting_Exercise_filled.xlsx"
PLAN_COLS = ["planned_own_stores", "planned_partners", "planned_discount_change", "planned_site_discount_change", "planned_new_colours", "planned_same_type_launches"]


class Engine:
    def __init__(self, threads=3):
        src = open("eda/198v_member_decisions.py", encoding="utf-8").read(); g = {"__name__": "plan_engine"}
        exec(compile(src[:src.index('B0, F0_ = sets("ec_ns"); PRED = {}')], "eda/198v_member_decisions.py", "exec"), g)   # member table + all axes, incl. realised decisions up to 2026-02
        g["ns"]["P"]["n_jobs"] = threads; self.T, self.fit, self.DEC = g["T"], g["fit"], g["DEC"]; self.B0, self.F0 = g["sets"]("ec_ns"); self.win = g["ax"]["win"]
        T = self.T; oo = pd.Period(O26).ordinal; self.tr = T[(T.o_ord + 4 <= oo) & T.ytot.notna()]; self.te = T[T.origin == O26].copy().reset_index(drop=True); assert self.tr.origin.max() == "2026-02" and len(self.te) > 0
        s = self.te.style_color.values; nz = lambda v: np.nan_to_num(v, neginf=0)
        with np.errstate(all="ignore"):
            self.te["cur_stores_b3"] = nz(self.win("n_stores", s, O26, [-2, -1, 0], "max")); self.te["cur_partners_b3"] = nz(self.win("n_partners", s, O26, [-2, -1, 0], "max"))
        ly = self.T[self.T.origin == "2025-06"].groupby("ptype").ax_h_new_type.median(); self.te["ly_same_type_launches"] = ly.reindex(self.te.ptype.astype(str).values).fillna(0).values   # Jul-Oct 2025, same type
        M = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color").reindex(self.te.style_color); self.others = (M[["rules", "nn", "pf2c", "pf2d32"]].sum(1)).values; self.corr = (M.F / M.F0).values
        ws = openpyxl.load_workbook(WB, data_only=True)["Forecast"]; self.wb = {x[1]: dict(name=x[2], total=float(x[10]), confidence=x[11]) for x in ws.iter_rows(min_row=2, values_only=True) if x[1]}
        self.base = self.system(self.inputs(pd.DataFrame({"style_color": self.te.style_color})))   # the no-change plan

    def inputs(self, plan):
        """plan -> the decision inputs for every item at June 2026 (blank = no change)."""
        P = plan.set_index("style_color").reindex(self.te.style_color); X = self.te.copy(); g = lambda c: P[c].values if c in P else np.full(len(X), np.nan)
        st = np.where(np.isnan(g("planned_own_stores")), X.cur_stores_b3, g("planned_own_stores")); pa = np.where(np.isnan(g("planned_partners")), X.cur_partners_b3, g("planned_partners"))
        dc = np.nan_to_num(g("planned_discount_change")); sd = np.nan_to_num(g("planned_site_discount_change")); nc = np.nan_to_num(g("planned_new_colours"))
        tl = np.where(np.isnan(g("planned_same_type_launches")), X.ly_same_type_launches, g("planned_same_type_launches"))
        X["ax_h_stores_chg"] = st - X.cur_stores_b3; X["ax_stores_end"] = X.ax_n_stores + X.ax_h_stores_chg; X["ax_h_partners_chg"] = pa - X.cur_partners_b3
        X["ax_h_disc_chg"] = dc; X["ax_h_disc_chg_rel"] = dc - sd; X["ax_h_new_style"] = nc; X["ax_h_new_type"] = tl; return X

    def system(self, X):
        if not hasattr(self, "_models"):   # fit once: the two feature sets of Main's member, each with the 7 decisions
            import lightgbm as lgb
            P = self.fit.__globals__["P"]; self._models = [lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(self.tr[c + self.DEC], self.tr.z) for c in (self.B0, self.F0)]
        z = np.mean([m.predict(X[c + self.DEC]) for m, c in zip(self._models, (self.B0, self.F0))], 0); tree = np.clip((4 * X.level.values + 1) * np.exp(z) - 1, 0, None)
        return (self.others + tree) / 5 * self.corr

    def forecast(self, plan):
        sysp = self.system(self.inputs(plan)); mult = (sysp + 1) / (self.base + 1); O = pd.DataFrame({"style_color": self.te.style_color, "multiplier": mult})
        O["submitted"] = [self.wb.get(s, {}).get("total", np.nan) for s in O.style_color]; O["plan_forecast"] = O.submitted * O.multiplier; return O.set_index("style_color")

    def template(self, path):
        t = self.te.set_index("style_color"); rows = []
        for s, v in self.wb.items():
            if s not in t.index: continue
            x = t.loc[s]; rows.append(dict(style_color=s, name=v["name"], product_type=str(x.ptype), submitted_total=round(v["total"]), current_own_stores=int(x.cur_stores_b3), current_partners=int(x.cur_partners_b3),
                                           current_web_discount=round(float(x.ax_disc_web), 3) if np.isfinite(x.ax_disc_web) else "", **{c: "" for c in PLAN_COLS}))
        pd.DataFrame(rows).to_csv(path, index=False); return path


if __name__ == "__main__":
    E = Engine()
    if sys.argv[1] == "--template": print("template:", E.template(sys.argv[2])); sys.exit()
    plan = pd.read_csv(sys.argv[1]); out = E.forecast(plan); tg = out[out.submitted.notna()]
    print(f"targets: submitted {tg.submitted.sum():,.0f} -> with the plan {tg.plan_forecast.sum():,.0f} ({tg.plan_forecast.sum() / tg.submitted.sum() - 1:+.1%})")
    print(tg.assign(change=tg.plan_forecast / tg.submitted - 1).sort_values("change").round(3).head(8).to_string()); print(tg.assign(change=tg.plan_forecast / tg.submitted - 1).sort_values("change").round(3).tail(8).to_string())
    if len(sys.argv) > 2: out.to_csv(sys.argv[2]); print("written", sys.argv[2])
