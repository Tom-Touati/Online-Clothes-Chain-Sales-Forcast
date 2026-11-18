"""Fork 2 (D419f; builder session, after D435: "nx16 is now the submission ... Please move the four pages to nx16").
Pages move from nx13 to nx16 (D433 nx14, D434 nx15, D435 nx16: the same numbers as nx9-nx13; the Model explanation gained the colour-coded metrics
table, last as every metric for SYS and, as a ratio to SYS, for R3, SN, AR and each of SYS's six models alone, Fork 2's D419e). Here: the submission
label nx13 -> nx16 in the explorer, atlas, error study and scoreboard, the file line (md5, builder, words, check), and one sentence on nx14-nx16.
Usage: python eda/236_pages_nx16.py <md5 of filled.xlsx>; then rebuild 212 / 212b / 212c, 19 + 127, 198g and 159."""
import sys, json
MD5 = sys.argv[1][:8]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
patch("eda/212_item_explorer.py", [
    ('C = pd.read_csv("data/final_2026_nx13_cascade.csv")   # nx13 (nx12 numbers; abbreviations SYS / R3 / SN / AR, D432) is the submission',
     'C = pd.read_csv("data/final_2026_nx16_cascade.csv")   # nx16 (nx9-nx13 numbers; per-model metrics table, D435) is the submission'),
    ('C5 = pd.read_csv("data/final_2026_nx13_cascade.csv")', 'C5 = pd.read_csv("data/final_2026_nx16_cascade.csv")')])
patch("eda/19_project_map_data.py", [
    ('C = pd.read_csv("data/final_2026_nx13_cascade.csv", index_col=0)', 'C = pd.read_csv("data/final_2026_nx16_cascade.csv", index_col=0)'),
    ('H = pd.read_csv("data/final_2026_nx13_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx16_cascade_hierarchy.csv")')])
patch("eda/item_explorer_template.html", [("· the submitted forecast (nx13, 18 Nov:", "· the submitted forecast (nx16, 18 Nov:")])
TABLE = ("a colour-coded table of every metric for SYS and, as a ratio to it, for R3, SN, AR and each of SYS's six models alone "
         "(SYS lowest on every metric)")
patch("eda/atlas_template.html", [
    ("<h3>The submission (nx13, 18 Nov): six models", "<h3>The submission (nx16, 18 Nov): six models"),
    ("the buying simulation of the submitted system itself and a metrics-versus-baselines sheet</h3>",
     f"the buying simulation of the submitted system itself, a metrics-versus-baselines sheet and, in the Model explanation, {TABLE}</h3>"),
    ('<th class="r">nx13 (submitted)</th>', '<th class="r">nx16 (submitted)</th>'),
    ("<code>candidates/…_candidate_nx13.xlsx</code> (md5 b177d24e…; builder <code>eda/181u_final_nx13_2026.py</code>, words <code>eda/181u_nx13_text.py</code>, check <code>eda/181u_check_nx13.py</code>: every Forecast and cascade cell identical to nx12, nx11, nx10 and nx9;",
     f"<code>candidates/…_candidate_nx16.xlsx</code> (md5 {MD5}…; builder <code>eda/181x_final_nx16_2026.py</code>, words <code>eda/181x_nx16_text.py</code>, check <code>eda/181x_check_nx16.py</code>: every Forecast and cascade cell identical to nx11, as in nx14 and nx15, and nx11 identical to nx10 and nx9;"),
    ('the submitted workbook (nx13, D432)"', 'the submitted workbook (nx16, D435)"')])
patch("eda/error_study_template.html", [
    ("the submitted nx13 is under", "the submitted nx16 is under"),
    ("<h2>nx13 is the submission:", "<h2>nx16 is the submission:"),
    ("always in that order (D432). The new nx5 is",
     f"always in that order (D432), and then to nx14, nx15 and nx16: the same numbers, with {TABLE.replace('a colour-coded', 'a colour-coded Model explanation')} (D433–D435). The new nx5 is"),
    ("(submitted as nx13)</b>", "(submitted as nx16)</b>")])
# scoreboard: the SYS row's name and status (scores unchanged)
OLD, NEW = "SYS: nx9 – nx13 = nx8 + look-alike correction (submitted as nx13)", "SYS: nx9 – nx16 = nx8 + look-alike correction (submitted as nx16)"
S_OLD, S_NEW = "SUBMITTED as nx13 (18 Nov, D432; same numbers as nx12, D431);", "SUBMITTED as nx16 (18 Nov, D435; same numbers as nx9 – nx15);"
patch("eda/159_scoreboard.py", [(f'"{OLD}"', f'"{NEW}"'), (S_OLD, S_NEW)])
J = json.load(open("data/scoreboard.json", encoding="utf-8")); n = 0
for r in J["rows"]:
    if r["system_id"] == "nx9_geo":
        assert r["system"] == OLD and S_OLD in r["status"]; r["system"] = NEW; r["status"] = r["status"].replace(S_OLD, S_NEW); n += 1
json.dump(J, open("data/scoreboard.json", "w", encoding="utf-8"))
W = json.load(open("data/scoreboard_sim.json", encoding="utf-8"))
for w in W["rows"]:
    if w["system_id"] == "nx9_geo": w["system"] = NEW
json.dump(W, open("data/scoreboard_sim.json", "w", encoding="utf-8")); print("patched; scoreboard rows renamed:", n)
