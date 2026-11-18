"""Fork 2 (D420-F2 = Fork 2's last id in D410-D419 is used; logged as D419b; user, via the builder session: "its unclear what is within the parantheses ... set abbreviations and use them consistently").
Pages move to nx13 (the submission, D432: same numbers, abbreviations SYS = submitted system, R3, SN = seasonal naive, AR = agents' rules, always
in that order, every value in a list named). Here: the submission label nx12 -> nx13; every unlabelled list of figures in Fork 2's page text gets
each value named; the baselines follow SYS, R3, SN, AR; the scoreboard's names carry the abbreviations.
Usage: python eda/233_pages_nx13.py <md5 of filled.xlsx>; then rebuild 212 / 212b / 212c, 198g and 127."""
import sys, json
MD5 = sys.argv[1][:8]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
patch("eda/212_item_explorer.py", [
    ('C = pd.read_csv("data/final_2026_nx12_cascade.csv")   # D419: nx12 (nx11 numbers; own buying simulation, metrics sheet, D431) is the submission',
     'C = pd.read_csv("data/final_2026_nx13_cascade.csv")   # nx13 (nx12 numbers; abbreviations SYS / R3 / SN / AR, D432) is the submission'),
    ('C5 = pd.read_csv("data/final_2026_nx12_cascade.csv")', 'C5 = pd.read_csv("data/final_2026_nx13_cascade.csv")')])
patch("eda/19_project_map_data.py", [
    ('C = pd.read_csv("data/final_2026_nx12_cascade.csv", index_col=0)', 'C = pd.read_csv("data/final_2026_nx13_cascade.csv", index_col=0)'),
    ('H = pd.read_csv("data/final_2026_nx12_cascade_hierarchy.csv")', 'H = pd.read_csv("data/final_2026_nx13_cascade_hierarchy.csv")')])
patch("eda/item_explorer_template.html", [("· the submitted forecast (nx12, 18 Nov:", "· the submitted forecast (nx13, 18 Nov:")])
patch("eda/atlas_template.html", [
    ("<h3>The submission (nx12, 18 Nov): six models", "<h3>The submission (nx13, 18 Nov): six models"),
    ('<th class="r">nx12 (submitted)</th>', '<th class="r">nx13 (submitted)</th>'),
    ("<code>candidates/…_candidate_nx12.xlsx</code> (md5 01ec09df…; builder <code>eda/181t_final_nx12_2026.py</code>, words <code>eda/181t_nx12_text.py</code>, check <code>eda/181t_check_nx12.py</code>: every Forecast and cascade cell identical to nx11, nx10 and nx9;",
     f"<code>candidates/…_candidate_nx13.xlsx</code> (md5 {MD5}…; builder <code>eda/181u_final_nx13_2026.py</code>, words <code>eda/181u_nx13_text.py</code>, check <code>eda/181u_check_nx13.py</code>: every Forecast and cascade cell identical to nx12, nx11, nx10 and nx9;"),
    ('the submitted workbook (nx12, D431)"', 'the submitted workbook (nx13, D432)"'),
    # unlabelled lists
    ("(all items: 74 / 19 / 7)", "(all items: online 74%, retail 19%, drop 7%)"),
    ("Level with D361's nx5 (0.3811 / 0.2924 / 0.4358).", "Level with D361's nx5 (recent 0.3811, held out 0.2924, all 13 0.4358)."),
    ('<td class="r" colspan="3">−1.0% / +1.3% / +10.8%; largest', '<td class="r" colspan="3">10th −1.0%, 50th +1.3%, 90th +10.8%; largest'),
    ("(levels 1 / 2 / 3 land within ±20% 58% / 44% / 28% of the time in the backtest)", "(within ±20% in the backtest: level 1 58% of the time, level 2 44%, level 3 28%)")])
patch("eda/error_study_template.html", [
    ("the submitted nx12 is under", "the submitted nx13 is under"),
    ("<h2>nx12 is the submission:", "<h2>nx13 is the submission:"),
    ("(submitted as nx12)</b>", "(submitted as nx13)</b>"),
    ("with the buying simulation of the submitted system itself (10.6% of the ideal profit lost, against 12.4% for the agents' rules, 14.5% for the seasonal naive and 19.5% for R3; better than the rules in 9 of 11 three-month windows) and a 'Metrics vs baselines' sheet (D431). The new nx5 is",
     "with the buying simulation of the submitted system itself (share of the ideal profit lost: SYS 10.6%, R3 19.5%, SN 14.5%, AR 12.4%; SYS better than AR in 9 of 11 three-month windows) and a 'Metrics vs baselines' sheet (D431), and then to nx13: the same numbers with fixed abbreviations, used here too: SYS = the submitted system, R3 = the last three months x4, SN = seasonal naive, AR = the agents' rules, always in that order (D432). The new nx5 is"),
    # unlabelled lists
    ("gives 0.3811 recent / 0.4358 all 13 against the submitted 0.3851 / 0.4399", "gives recent 0.3811 and all 13 0.4358, against the then-submitted nx4's recent 0.3851 and all 13 0.4399"),
    ("is the best raw backtest, 0.3786 / 0.4333, but fails", "is the best raw backtest (recent 0.3786, all 13 0.4333) but fails"),
    ("moves four model-forecast items from 3 to 2 (levels 10 / 86 / 4)", "moves four model-forecast items from 3 to 2 (10 items at level 1, 86 at level 2, 4 at level 3)"),
    ("the first layer alone is the best no-plan model (0.406 / 0.462)", "the first layer alone is the best no-plan model (recent 0.406, all 13 0.462)")])
# scoreboard: abbreviations in the names (scores unchanged)
REN = {"nx9_geo": ("nx9 – nx12 = nx8 + look-alike correction (submitted as nx12)", "SYS: nx9 – nx13 = nx8 + look-alike correction (submitted as nx13)"),
       "R3": ("R3 (last 3 months x 4)", "R3 (last 3 months x 4)"),
       "rules": ("R3 + rules (rule agents)", "AR: agents' rules (with R3)"),
       "sea_naive": ("Seasonal naive (R3 x type seasonal ratio)", "SN: seasonal naive (R3 x type seasonal ratio)")}
p = "eda/159_scoreboard.py"; t = open(p, encoding="utf-8").read()
for k, (a, b) in REN.items():
    if a != b: assert t.count(f'"{a}"') == 1, (k, t.count(f'"{a}"')); t = t.replace(f'"{a}"', f'"{b}"')
t = t.replace('"SUBMITTED as nx12 (18 Nov, D431 / D419);', '"SUBMITTED as nx13 (18 Nov, D432; same numbers as nx12, D431);'); open(p, "w", encoding="utf-8").write(t)
J = json.load(open("data/scoreboard.json", encoding="utf-8")); n = 0
for r in J["rows"]:
    if r["system_id"] in REN: r["system"] = REN[r["system_id"]][1]; n += 1
    if r["system_id"] == "nx9_geo": r["status"] = r["status"].replace("SUBMITTED as nx12 (18 Nov, D431 / D419);", "SUBMITTED as nx13 (18 Nov, D432; same numbers as nx12, D431);")
json.dump(J, open("data/scoreboard.json", "w", encoding="utf-8"))
W = json.load(open("data/scoreboard_sim.json", encoding="utf-8"))
for w in W["rows"]:
    if w["system_id"] in REN: w["system"] = REN[w["system_id"]][1]
json.dump(W, open("data/scoreboard_sim.json", "w", encoding="utf-8")); print("patched; scoreboard rows renamed:", n)
