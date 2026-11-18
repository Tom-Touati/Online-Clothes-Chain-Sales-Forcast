"""Fork 2 (D418; user: "update the pages and republish the repo when nx11 is in"): moves the page sources from nx10 to nx11, the submission built
from the D417 draft (every Forecast and cascade cell identical to nx10; the Model explanation rewritten coarse-to-fine with all the main metrics
and the cascade as explainability). Usage: python eda/231_pages_nx11.py <md5 of filled.xlsx>; then rebuild 212 / 212b / 212c, 198g and 127."""
import sys, json
MD5 = sys.argv[1][:8]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
patch("eda/212_item_explorer.py", [
    ('C = pd.read_csv("data/final_2026_nx10_cascade.csv")   # D416: nx10 (nx9 numbers, repaired text + roadmap, D424) is the submission',
     'C = pd.read_csv("data/final_2026_nx11_cascade.csv")   # D418: nx11 (nx10 numbers, explanation rewritten coarse-to-fine, D417) is the submission'),
    ('C5 = pd.read_csv("data/final_2026_nx10_cascade.csv")', 'C5 = pd.read_csv("data/final_2026_nx11_cascade.csv")')])
patch("eda/19_project_map_data.py", [
    ('C = pd.read_csv("data/final_2026_nx10_cascade.csv", index_col=0)', 'C = pd.read_csv("data/final_2026_nx11_cascade.csv", index_col=0)'),
    ('H = pd.read_csv("data/final_2026_nx10_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx11_cascade_hierarchy.csv")')])
s = open("eda/item_explorer_template.html", encoding="utf-8").read(); i = s.index("· the submitted forecast (nx10, 1"); j = s.index("</p>", i)
s = s[:i] + "· the submitted forecast (nx11, 18 Nov: six models incl. a launch-weighted tree, look-alike corrections, campaign uplift, confidence by the brief's scale)" + s[j:]
open("eda/item_explorer_template.html", "w", encoding="utf-8").write(s)
patch("eda/atlas_template.html", [
    ("<h3>The submission (nx10, 18 Nov): six models", "<h3>The submission (nx11, 18 Nov): six models"),
    ("and the repaired texts with an improvement roadmap</h3>", "and a Model explanation written coarse to fine (summary, models, information, all the main metrics, the cascade as explainability, improvement roadmap)</h3>"),
    ('<th class="r">nx10 (submitted)</th>', '<th class="r">nx11 (submitted)</th>'),
    ("<code>candidates/…_candidate_nx10.xlsx</code> (md5 0d64797f…; builder <code>eda/181r_final_nx10_2026.py</code>, check <code>eda/181r_check_nx10.py</code>: every Forecast cell identical to nx9;",
     f"<code>candidates/…_candidate_nx11.xlsx</code> (md5 {MD5}…; builder <code>eda/181s_final_nx11_2026.py</code>, words <code>eda/181s_nx11_text.py</code>, check <code>eda/181s_check_nx11.py</code>: every Forecast and cascade cell identical to nx10 and nx9;"),
    ('the submitted workbook (nx10, D424)"', 'the submitted workbook (nx11, D417)"')])
patch("eda/error_study_template.html", [
    ("the submitted nx10 is under", "the submitted nx11 is under"),
    ("<h2>nx10 is the submission:", "<h2>nx11 is the submission:"),
    ("(submitted as nx10)</b>", "(submitted as nx11)</b>")])
s = open("eda/error_study_template.html", encoding="utf-8").read()
a = "(D413 / D424). The new nx5 is"
assert s.count(a) == 1
s = s.replace(a, "(D413 / D424), and then to nx11: the same numbers again, with the Model explanation rewritten from coarse to fine (a summary, the six models, the information axis by axis, one metric per aspect of accuracy, the buying simulation, and the cascade sheets as the explanation of every number; D417). The new nx5 is")
open("eda/error_study_template.html", "w", encoding="utf-8").write(s)
# scoreboard label (scores unchanged)
OLD = "nx9 / nx10 = nx8 + look-alike correction (submitted as nx10)"; NEW = "nx9 / nx10 / nx11 = nx8 + look-alike correction (submitted as nx11)"
p = "eda/159_scoreboard.py"; t = open(p, encoding="utf-8").read(); assert t.count(f'"{OLD}"') == 1
t = t.replace(f'"{OLD}"', f'"{NEW}"').replace('"SUBMITTED as nx10 (18 Nov, D424):', '"SUBMITTED as nx11 (18 Nov, D417 / D418):'); open(p, "w", encoding="utf-8").write(t)
J = json.load(open("data/scoreboard.json", encoding="utf-8")); n = 0
for r in J["rows"]:
    if r["system_id"] == "nx9_geo":
        r["system"] = NEW; r["status"] = r["status"].replace("SUBMITTED as nx10 (18 Nov, D424):", "SUBMITTED as nx11 (18 Nov, D417 / D418):"); n += 1
json.dump(J, open("data/scoreboard.json", "w", encoding="utf-8")); print("pages patched; scoreboard rows relabelled:", n)
