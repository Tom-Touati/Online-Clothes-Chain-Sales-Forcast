"""D423: where each lab file goes in the clean repository (read by eda/publish_clean_repo.py).
  the one submission file                          -> the repository root
  what Edikted provided                            -> exercise/
  the scripts of the current solution (PIPELINE)   -> pipeline/<same name>      (paths inside rewritten by transform())
  the planning-rule agents' rules and forecasts    -> rule_agents/<origin>/      (their scratch work is archived)
  tables the pipeline reads or writes (needed_data)-> data/<same path>
  every other workbook                             -> model_submissions/<folder>/ (MODELS)
  the decision log, the EDA report, the pages      -> docs/
  everything else                                  -> archive/<lab path>         (past attempts: scripts, run logs, tables, pages)"""
import re
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]
SUBMISSION = "Edikted_Forecasting_Exercise_filled.xlsx"
NOT_PUBLISHED = {"README.md", ".gitignore", "Edikted_Forecasting_Exercise_candidate_nx13.xlsx"}   # the lab's own root files; nx13 = the submission (a duplicate, D432)
MOVED = ("candidates/", "reports/")   # D426: the lab root's workbooks and reports moved to these folders (Fork 2, user: "too many files in the main directory"); names are looked up without the folder
# stage -> scripts, in run order (the pipeline README is written from this)
STAGES = [
 ("1 Base tables from the order lines", ["02_build_aggs.py", "03_targets.py", "04_eda.py", "06_features.py"]),
 ("2 Feature tables", ["152_orders_extract.py", "153_basket_emb.py", "160_weather_fetch.py", "160b_weather_noaa.py", "161_events.py", "162_weather_event_feats.py", "163_clusters.py",
                       "177t_product_feed.py", "177u_content_emb_server.py", "181_cooc_dyn.py", "182_cooc_feats.py", "182b_fashion_feats.py", "182d_nn_content.py", "182p_planlook.py",
                       "187_explain_table.py", "180_newfeats.py", "180b_promo.py", "209_tree_temporal.py"]),
 ("3 The six members", ["76_rules_harness.py", "180q_models.py", "187_losses.py", "218a_route_tree_2026.py", "57_embed_nn.py", "197b_nn_ec.py", "196b_tabpfn.py", "199c_conformal.py", "199e_conformal_apply.py", "202_rc_model.py", "205_rc_handover.py"]),
 ("4 Base table and corrections", ["179d_base.py", "180d_newbase.py", "181a_base2026.py", "218_base_nx5r.py", "179x2_bias.py", "179b_breakout.py", "218c_geo_nx5r.py", "228_breakout_fashion.py"]),
 ("5 Ranges and confidence", ["219_ranges_nx5r.py", "198d_error_master.py", "198ac_confidence_rule.py", "222_confidence_new.py"]),
 ("6 The workbook", ["181u_final_nx13_2026.py", "181u_nx13_text.py"]),
 ("7 Evaluation quoted in the workbook", ["226_vs_baselines.py", "224_gate_nx5r.py", "221_month_split.py", "226_campaign_test.py", "217_score_new5.py", "231_sim_nx11.py", "231b_sim_naive.py", "232_metrics_table.py"]),
 ("Shared helpers", ["score_lib.py", "86_metrics_suite.py", "20b_quantiles_lean.py", "lowrun.ps1", "47_decision_sim.py", "59_sim_v2.py", "87_sim_systems.py", "157_stack_new.py", "159_scoreboard.py"]),
]
PIPELINE = [s for _, ss in STAGES for s in ss]
MODELS = {   # lab workbook -> model_submissions/<folder>  (label, decision ids)
 "Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx": "01_first_submission_2026-11-12",
 "Edikted_Forecasting_Exercise_candidate_rules.xlsx": "member_rule_agents",
 "Edikted_Forecasting_Exercise_candidate_nn.xlsx": "member_neural_network",
 "Edikted_Forecasting_Exercise_candidate_final.xlsx": "02_p2c32a_stack",
 "Edikted_Forecasting_Exercise_candidate_nx.xlsx": "03_nx_new_system",
 "Edikted_Forecasting_Exercise_candidate_nx2.xlsx": "04_nx2_new_nn",
 "Edikted_Forecasting_Exercise_candidate_nx3.xlsx": "05_nx3_world_cup_rule",
 "Edikted_Forecasting_Exercise_candidate_nx4.xlsx": "06_nx4_confidence_rule",
 "Edikted_Forecasting_Exercise_candidate_nx5_d361.xlsx": "07_nx5_huber_tree_recommender",
 "Edikted_Forecasting_Exercise_candidate_nx5.xlsx": "08_nx5_routed_launch_tree",
 "Edikted_Forecasting_Exercise_candidate_nx6.xlsx": "09_nx6_hybrid_confidence",
 "Edikted_Forecasting_Exercise_candidate_nx7.xlsx": "10_nx7_audited_text",
 "Edikted_Forecasting_Exercise_candidate_nx7c.xlsx": "11_nx7c_campaign_uplift",
 "Edikted_Forecasting_Exercise_candidate_nx8.xlsx": "12_nx8_combined",
 "Edikted_Forecasting_Exercise_candidate_nx9.xlsx": "13_nx9_lookalike_term",
 "Edikted_Forecasting_Exercise_candidate_nx10.xlsx": "14_nx10_roadmap_text",
 "Edikted_Forecasting_Exercise_candidate_nx11.xlsx": "15_nx11_metrics_cascade",
 "Edikted_Forecasting_Exercise_candidate_nx12.xlsx": "16_nx12_buying_metrics_table",
}
EXERCISE = {"Edikted_Forecasting_Exercise.xlsx": "exercise/Edikted_Forecasting_Exercise.xlsx", "item_attributes.csv": "exercise/item_attributes.csv"}
DOCS = {"log.md": "docs/decision_log.md", "EDA_REPORT.md": "docs/EDA_REPORT.md",
        "eda/item_explorer.html": "docs/pages/item_explorer.html", "eda/atlas.html": "docs/pages/atlas.html", "eda/error_study.html": "docs/pages/error_study.html"}
RULE_KEEP = re.compile(r"^agent_eval/(actuals\.parquet|[0-9]{4}-[0-9]{2}/(rules\.py|predictions\.csv|predictions_verified\.csv|notes\.md|data/.*))$")
HAND_ROOTS = ["README.md", ".gitignore", "pipeline/README.md", "data/README.md", "model_submissions/README.md", "archive/README.md", "docs/README.md", "exercise/README.md", "rule_agents/README.md"]
REQUIRED = [SUBMISSION] + ["eda/" + s for s in PIPELINE]

# tables the pipeline reads or writes: string literals data/... in the pipeline scripts, plus the f-string families they use
LIT = re.compile(r"""["'](data/[^"'{}\s]+?\.(?:parquet|csv|json|npz|npy|pkl|txt))["']""")
FAMILIES = [r"data/llm/.*", r"data/weather_raw/.*", r"data/q_backtest_ls_huber(_new5)?(_2026|_lyd)?\.parquet", r"data/srv4/q_backtest_ls_huber(_noly)?\.parquet", r"data/dec_base(_ec_ns|_ecnsnn|_ecnsnnhubrq|_ecnsnnhubrq26|_nx5r26)?\.parquet",
            r"data/bias_seg_preds_nx5r26\.parquet", r"data/breakout_preds_nx5r26_x\.parquet", r"data/geo_nx5r(_fash)?_x_(preds|2026)\.parquet", r"data/geo_nx5r_fash_2026\.parquet",
            r"data/ranges_nx5r(_fash)?_(2026\.csv|loo\.parquet)", r"data/final_2026_nx(9|10|11|12|13)_.*\.csv", r"data/scoreboard\.json", r"data/q_backtest_fktab_tabpfn_v2_dyn(_e32)?\.parquet",
            r"data/q_backtest_fkconf_pfn2dyn_seg_tails\.parquet", r"data/fk_2026_(pf2c|pf2d32|w03|qdyn)\.parquet", r"data/nn_backtest_preds_(S13ecx6|finalSec|finalSec_s10)\.parquet",
            r"data/q_backtest_nx_(ec_ns|w03ec)(_2026)?\.parquet", r"data/rc_recsys_preds(_noly)?\.parquet", r"data/fk_err_master\.parquet", r"data/confidence_new_bt\.parquet"]
PATHLIB = re.compile(r"DATA / [\"']([^\"']+\.(?:parquet|csv|json|npz))[\"']")
def needed_data(lab_files):
    lits = set()
    for s in PIPELINE:
        p = LAB / "eda" / s
        if p.suffix == ".py" and p.exists():
            txt = p.read_text(encoding="utf-8", errors="replace"); lits |= set(LIT.findall(txt))
            lits |= {"data/" + m for m in PATHLIB.findall(txt)}          # pathlib form: DATA / "name"
    fam = [re.compile("^" + f + "$") for f in FAMILIES]
    return {f for f in lab_files if f in lits or any(r.match(f) for r in fam)}
NEEDED_DATA = set()      # filled by publish_clean_repo.py with needed_data(<all lab files>)

def place(path):
    key = path.split("/", 1)[1] if path.startswith(MOVED) else path     # candidates/X.xlsx -> X.xlsx, so the folder does not change where it goes
    if key in NOT_PUBLISHED: return None
    if path == SUBMISSION: return path
    if path in EXERCISE: return EXERCISE[path]
    if key in MODELS: return f"model_submissions/{MODELS[key]}/{key}"
    if key in DOCS: return DOCS[key]
    if path.startswith("eda/") and path[4:] in PIPELINE: return "pipeline/" + path[4:]
    if RULE_KEEP.match(path): return "rule_agents/" + path[len("agent_eval/"):]
    if path in NEEDED_DATA: return path
    return "archive/" + path

_EDA = re.compile(r"(?<![A-Za-z0-9_./])eda/([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9]+)?)")
PAGE_LINKS = {"137UdxMX2e7re5W9isksTz": "atlas.html", "NkoNJTZUBirD4soyyH9sDv": "error_study.html", "ATC2p7yCELvrxsnZRgmbS8": "item_explorer.html"}   # D428: the pages' private claude.ai links -> the sibling files in docs/pages/
def transform(dst, text):
    """Rewrites lab paths inside the published pipeline scripts so they run from the repository root, and the pages' links to each other."""
    if dst.startswith("docs/pages/") and dst.endswith(".html"):
        t = re.sub(r"https://claude.ai/artifact/(" + "|".join(PAGE_LINKS) + r")(?![A-Za-z0-9])", lambda m: PAGE_LINKS[m.group(1)], text)
        return t if t != text else None
    if not (dst.startswith("pipeline/") or dst.startswith("rule_agents/")) or not dst.endswith((".py", ".ps1")): return None
    def where(tok):
        if tok in PIPELINE or tok.endswith(("_output.txt", ".err")): return "pipeline/"
        stem = tok[:-3] if tok.endswith(".py") else tok
        if "." not in stem and any(p.startswith(stem + "_") or p == stem + ".py" for p in PIPELINE) and not (LAB / "eda" / tok).exists(): return "pipeline/"   # "eda/153" = eda/153_basket_emb.py
        return "archive/eda/"
    t = _EDA.sub(lambda m: where(m.group(1)) + m.group(1), text)
    t = re.sub(r"(?<![A-Za-z0-9_./])agent_eval/", "rule_agents/", t)
    t = t.replace('"agent_eval"', '"rule_agents"')
    t = re.sub(r"""(["'])Edikted_Forecasting_Exercise\.xlsx\1""", r"\1exercise/Edikted_Forecasting_Exercise.xlsx\1", t)
    t = re.sub(r"""(["'])item_attributes\.csv\1""", r"\1exercise/item_attributes.csv\1", t)
    t = re.sub(r'WORKBOOK = ROOT / "(?:candidates/)?Edikted_Forecasting_Exercise_candidate_nx\w+\.xlsx"\s*# written', 'WORKBOOK = ROOT / "Edikted_Forecasting_Exercise_filled.xlsx"           # written: the submission', t)
    return t if t != text else None
