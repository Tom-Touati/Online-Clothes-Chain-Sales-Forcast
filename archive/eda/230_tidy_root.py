"""Fork 2 (user, 18 Nov: "there are too many files in the main directory. everything that isnt relevant to the submission files should be kept in a
separate dir"). Moves every root file that is not the submission or its inputs into two folders, and patches the scripts that name them:
  candidates/   every Edikted_Forecasting_Exercise_candidate_*.xlsx and the first submission (..._filled_original_2026-11-12.xlsx)
  reports/      the markdown reports (EDA, error analysis, overnight, rule agents, explainability, data request, expectation, decomposition)
Kept at the root: Edikted_Forecasting_Exercise_filled.xlsx (the submission), Edikted_Forecasting_Exercise.xlsx, Dataset.parquet and item_attributes.csv
(Edikted's inputs), README.md, log.md and SYNC.md (the decision log and the sessions' coordination file), .gitignore.
Scripts: in eda/*.py and agent_eval/*.py a quoted root path to a moved file gets the folder prefix (only string literals that start with the file name,
so paths already under a folder are untouched). Usage: python eda/230_tidy_root.py            (dry run: lists moves and script edits)
                                                         python eda/230_tidy_root.py --apply"""
import os, re, sys, glob, shutil
APPLY = "--apply" in sys.argv
moves = {f: "candidates" for f in glob.glob("Edikted_Forecasting_Exercise_candidate_*.xlsx")}
moves["Edikted_Forecasting_Exercise_filled_original_2026-11-12.xlsx"] = "candidates"
for f in ["DECOMPOSITION_ERRORS_REPORT.md", "EDA_REPORT.md", "EDIKTED_DATA_REQUEST.md", "ERROR_ANALYSIS_REPORT.md", "FIXABLE_ERRORS_REPORT.md", "FORECAST_EXPECTATION.md",
          "OVERNIGHT_M5_REPORT.md", "OVERNIGHT_REPORT.md", "RULE_AGENTS_REPORT.md", "SUBMISSION_EXPLAINABILITY.md"]:
    if os.path.exists(f): moves[f] = "reports"
moves = {f: d for f, d in moves.items() if os.path.exists(f)}
print(f"{len(moves)} files to move:"); [print(f"   {f} -> {d}/") for f, d in sorted(moves.items(), key=lambda x: (x[1], x[0]))]
# patterns: a quote, then the file name (no folder before it); candidate names also as prefixes built in code, e.g. "Edikted_Forecasting_Exercise_candidate_nx" + tag
pats = [(re.compile(r'(["\'])(' + re.escape(f) + r')'), d) for f, d in moves.items()]
pats.append((re.compile(r'(["\'])(Edikted_Forecasting_Exercise_candidate_)'), "candidates"))
edits = {}
for p in sorted(glob.glob("eda/*.py") + glob.glob("agent_eval/*.py")):
    if p.endswith(("230_tidy_root.py", "publish_map.py")): continue   # publish_map.py: its owner (D426) made it prefix-agnostic
    s = open(p, encoding="utf-8").read(); t = s
    for rx, d in pats: t = rx.sub(lambda m: m.group(1) + d + "/" + m.group(2), t)
    t = re.sub(r'(candidates|reports)/(candidates|reports)/', r'\1/', t)   # never double-prefix
    if t != s: edits[p] = (s, t)
print(f"\n{len(edits)} scripts with path edits:", ", ".join(sorted(os.path.basename(p) for p in edits)))
if APPLY:
    for d in ("candidates", "reports"): os.makedirs(d, exist_ok=True)
    for f, d in moves.items(): shutil.move(f, os.path.join(d, f))
    for p, (s, t) in edits.items(): open(p, "w", encoding="utf-8").write(t)
    print("applied")
else: print("\n(dry run; nothing moved)")
