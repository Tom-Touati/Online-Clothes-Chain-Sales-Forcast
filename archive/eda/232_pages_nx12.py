"""Fork 2 (D419; the builder session's nx12 is the submission, D431): pages move from nx11 to nx12 (same numbers; the buying simulation of the
submitted system itself, a 'Metrics vs baselines' sheet, NOAA temperatures, real-world evaluation as the principle, the ranges as a weakness).
Scoreboard: the submitted row gets the simulation of the system itself from data/sim_nx11_windows.csv (system nx11 = the submitted forecast;
same runs reproduce R3 0.1953 and the rules 0.1239 of data/sim_systems_windows.csv), in eda/159 (SIMF 'nx11') and patched into
data/scoreboard.json / scoreboard_sim.json now; a seasonal-naive reference row is registered in eda/159 for the next rebuild.
Usage: python eda/232_pages_nx12.py <md5 of filled.xlsx>; then rebuild 212 / 212b / 212c, 198g and 127."""
import sys, json, pandas as pd
MD5 = sys.argv[1][:8]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
patch("eda/212_item_explorer.py", [
    ('C = pd.read_csv("data/final_2026_nx11_cascade.csv")   # D418: nx11 (nx10 numbers, explanation rewritten coarse-to-fine, D417) is the submission',
     'C = pd.read_csv("data/final_2026_nx12_cascade.csv")   # D419: nx12 (nx11 numbers; own buying simulation, metrics sheet, D431) is the submission'),
    ('C5 = pd.read_csv("data/final_2026_nx11_cascade.csv")', 'C5 = pd.read_csv("data/final_2026_nx12_cascade.csv")')])
patch("eda/19_project_map_data.py", [
    ('C = pd.read_csv("data/final_2026_nx11_cascade.csv", index_col=0)', 'C = pd.read_csv("data/final_2026_nx12_cascade.csv", index_col=0)'),
    ('H = pd.read_csv("data/final_2026_nx11_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx12_cascade_hierarchy.csv")')])
patch("eda/item_explorer_template.html", [("· the submitted forecast (nx11, 18 Nov:", "· the submitted forecast (nx12, 18 Nov:")])
patch("eda/atlas_template.html", [
    ("<h3>The submission (nx11, 18 Nov): six models", "<h3>The submission (nx12, 18 Nov): six models"),
    ("the cascade as explainability, improvement roadmap)</h3>", "the cascade as explainability, improvement roadmap), the buying simulation of the submitted system itself and a metrics-versus-baselines sheet</h3>"),
    ('<th class="r">nx11 (submitted)</th>', '<th class="r">nx12 (submitted)</th>'),
    ("<code>candidates/…_candidate_nx11.xlsx</code> (md5 7106f479…; builder <code>eda/181s_final_nx11_2026.py</code>, words <code>eda/181s_nx11_text.py</code>, check <code>eda/181s_check_nx11.py</code>: every Forecast and cascade cell identical to nx10 and nx9;",
     f"<code>candidates/…_candidate_nx12.xlsx</code> (md5 {MD5}…; builder <code>eda/181t_final_nx12_2026.py</code>, words <code>eda/181t_nx12_text.py</code>, check <code>eda/181t_check_nx12.py</code>: every Forecast and cascade cell identical to nx11, nx10 and nx9;"),
    ('the submitted workbook (nx11, D417)"', 'the submitted workbook (nx12, D431)"')])
patch("eda/error_study_template.html", [
    ("the submitted nx11 is under", "the submitted nx12 is under"),
    ("<h2>nx11 is the submission:", "<h2>nx12 is the submission:"),
    ("(submitted as nx11)</b>", "(submitted as nx12)</b>"),
    ("the cascade sheets as the explanation of every number; D417). The new nx5 is",
     "the cascade sheets as the explanation of every number; D417), and then to nx12: the same numbers, with the buying simulation of the submitted system itself (10.6% of the ideal profit lost, against 12.4% for the agents' rules, 14.5% for the seasonal naive and 19.5% for R3; better than the rules in 9 of 11 three-month windows) and a 'Metrics vs baselines' sheet (D431). The new nx5 is")])
# scoreboard: eda/159 (SIMF source, the submitted row's simulation, a seasonal-naive reference row, labels)
OLD = "nx9 / nx10 / nx11 = nx8 + look-alike correction (submitted as nx11)"; NEW = "nx9 – nx12 = nx8 + look-alike correction (submitted as nx12)"
p = "eda/159_scoreboard.py"; t = open(p, encoding="utf-8").read()
for a, b in [('SIMF = {"sys": S0.rename(columns={"system": "key"}),', 'SIMF = {"nx11": pd.read_csv("data/sim_nx11_windows.csv").query("policy == \'production\'").rename(columns={"system": "key"}),   # D419: the submitted system\'s own simulation (D431)\n        "sys": S0.rename(columns={"system": "key"}),'),
             (f'    ("nx9_geo", "{OLD}", "D422", "ensemble", "nx9_geo", None),', f'    ("nx9_geo", "{NEW}", "D422", "ensemble", "nx9_geo", ("nx11", "nx11")),\n    ("sea_naive", "Seasonal naive (R3 x type seasonal ratio)", "D294", "reference", "sea_naive", ("nx11", "sea_naive")),'),
             ('"SUBMITTED as nx11 (18 Nov, D417 / D418):', '"SUBMITTED as nx12 (18 Nov, D431 / D419); buying simulation of this system 10.6% of the ideal profit lost:')]:
    assert t.count(a) == 1, (a[:60], t.count(a)); t = t.replace(a, b)
if '"sea_naive": (' not in t:
    t = t.replace('    "nx_hpo_pf_geo": (', '    "sea_naive": ("R3\'s monthly level x the product type\'s like-for-like seasonal ratio per target month (the error study\'s seasonal naive).", "baseline", "Reference"),\n    "nx_hpo_pf_geo": (', 1)
open(p, "w", encoding="utf-8").write(t)
S = pd.read_csv("data/sim_nx11_windows.csv"); S = S[(S.policy == "production") & (S.system == "nx11")].set_index("start").loss_pct
r3 = pd.read_csv("data/sim_systems_windows.csv"); r3 = r3[(r3.policy == "production") & (r3.system == "R3")].loss_pct.mean()
J = json.load(open("data/scoreboard.json", encoding="utf-8")); n = 0
for r in J["rows"]:
    if r["system_id"] == "nx9_geo":
        r["system"] = NEW; r["status"] = r["status"].replace("SUBMITTED as nx11 (18 Nov, D417 / D418):", "SUBMITTED as nx12 (18 Nov, D431 / D419); buying simulation of this system 10.6% of the ideal profit lost:")
        r["sim_loss"] = round(float(S.mean()), 4); r["sim_loss_r3"] = round(float(S.mean() / r3), 4); n += 1
json.dump(J, open("data/scoreboard.json", "w", encoding="utf-8"))
W = json.load(open("data/scoreboard_sim.json", encoding="utf-8")); W["rows"] = [w for w in W["rows"] if w["system_id"] != "nx9_geo"] + [{"system_id": "nx9_geo", "system": NEW, "window_start": k, "loss": round(float(v), 4)} for k, v in S.items()]
json.dump(W, open("data/scoreboard_sim.json", "w", encoding="utf-8"))
print("patched; scoreboard rows", n, "| sim_loss", round(float(S.mean()), 4), "| vs R3", round(float(S.mean() / r3), 4), "| windows", len(S))
