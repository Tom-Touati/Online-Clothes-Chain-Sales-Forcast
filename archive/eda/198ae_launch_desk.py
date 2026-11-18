"""D321 (fork; proposal #7 of the error study, assigned in D314: "For each launch publish three numbers instead of one: the forecast if rolled
out to all doors and partners, if partial, if web only"). Runs the plan engine (eda/plan_engine_2026.py) on scenarios for every item launched
in the last 6 months at June 2026, and writes the plan template for Edikted (#1a).
  scenarios   web only: no own stores beyond today's, no new partners | partial: half of the stores it is not yet in (at least 3), +1 partner |
              full: all 16 own stores, +2 partners; everything else as in the no-change plan
  check       the engine's sensitivity in the backtest sense: the measured multiples were 1.0x / 2.3x / 4.9x R3 for launches whose stores were flat /
              +2-4 / +5 or more (D276)
Writes data/fk_launch_desk_2026.csv, data/plan_template_2026.csv; adds J["launchdesk"]; stdout -> eda/launch_desk_output.txt."""
import sys, json, runpy, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 230)
E = runpy.run_path("eda/plan_engine_2026.py", run_name="engine")["Engine"](); t = E.te.set_index("style_color"); FLEET = 16
young = t[t.age_m < 6].copy(); print("items launched in the last 6 months:", len(young), "| of them targets:", int(young.index.isin(list(E.wb)).sum()), "| own stores now: median", float(young.cur_stores_b3.median()))
def plan(f_st, f_pa):
    return pd.DataFrame({"style_color": young.index, "planned_own_stores": [f_st(x) for x in young.itertuples()], "planned_partners": [f_pa(x) for x in young.itertuples()]})
SC = {"web only (no new doors or partners)": plan(lambda x: x.cur_stores_b3, lambda x: x.cur_partners_b3),
      "partial roll-out": plan(lambda x: min(FLEET, x.cur_stores_b3 + max(3, np.ceil((FLEET - x.cur_stores_b3) / 2))), lambda x: x.cur_partners_b3 + 1),
      "full roll-out (all 16 doors, +2 partners)": plan(lambda x: FLEET, lambda x: x.cur_partners_b3 + 2)}
M = pd.read_parquet("data/fk_err_master_2026.parquet").set_index("style_color"); out = young[["ptype", "age_m", "cur_stores_b3", "cur_partners_b3", "R3"]].copy(); out["system_forecast"] = M.F.reindex(out.index).round(0)
for lab, p in SC.items():
    f = E.forecast(p); out[lab] = (out.system_forecast * f.multiplier.reindex(out.index)).round(0)
out["target"] = out.index.isin(list(E.wb)); out["workbook_total"] = [E.wb.get(s, {}).get("total", np.nan) for s in out.index]; NM = pd.read_parquet("data/items.parquet").set_index("style_color").product_name; out.insert(0, "name", NM.reindex(out.index).values)
out["room_to_roll_out"] = FLEET - out.cur_stores_b3; out = out.sort_values(["room_to_roll_out", "system_forecast"], ascending=[False, False]); out.to_csv("data/fk_launch_desk_2026.csv")
k = list(SC); s = out[out.room_to_roll_out >= 5]; print("young items with room for 5+ more doors:", len(s))
print(out.head(15)[["name", "ptype", "age_m", "cur_stores_b3", "R3", "system_forecast"] + k].round(1).to_string())
ratio = lambda a, b, m: float((out.loc[m, a]).sum() / (out.loc[m, b]).sum())
summ = dict(n_young=int(len(out)), n_target_young=int(out.target.sum()), room5=int((out.room_to_roll_out >= 5).sum()), targets_room=int((out.target & (out.room_to_roll_out >= 3)).sum()),
            full_vs_web_room5=round(ratio(k[2], k[0], out.room_to_roll_out >= 5), 3), partial_vs_web_room5=round(ratio(k[1], k[0], out.room_to_roll_out >= 5), 3), full_vs_web_targets=round(ratio(k[2], k[0], out.target), 3),
            web_vs_system_all=round(ratio(k[0], "system_forecast", out.index == out.index), 3))
print(summ); tpath = E.template("data/plan_template_2026.csv"); print("template:", tpath)
J = json.load(open("data/fk_err_analysis.json")); J["launchdesk"] = dict(summary=summ, top=out.head(15).reset_index().round(1).to_dict("records")); json.dump(J, open("data/fk_err_analysis.json", "w"), indent=0); print("saved")
