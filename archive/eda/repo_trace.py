"""D423: static dependency trace of the current solution (for the repository clean-up, D420). For every script in eda/ (and the rule agents),
finds the files it reads and writes (string literals "data/...", "agent_eval/...", "eda/..." on the same line as a read or write call, plus
the f-string families listed in FSTRINGS), then walks back from the submission builder and the evaluation scripts to every producer.
Output: a JSON (scripts on the path with their reads / writes, needed files, files without a producer) to the path given as argument."""
import re, json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
OUT = sys.argv[1] if len(sys.argv) > 1 else "repo_trace.json"
LIT = re.compile(r"""["'](?P<p>(?:data|agent_eval|eda|out)/[^"'{}\s]+?\.(?:parquet|csv|json|npz|npy|txt|pkl|py|xlsx|html))["']""")
WRITE = re.compile(r"to_parquet|to_csv|json\.dump|np\.save|savez|open\([^)]*['\"]w|\.write\(|\.save\(|to_json|copyfile")
READ = re.compile(r"read_parquet|read_csv|json\.load|np\.load|open\(|exec\(|load_workbook|spec_from_file|ParquetFile|read_schema|read_json|copyfile")
SCRIPTS = sorted(set(Path("eda").glob("*.py")) | set(Path("agent_eval").glob("*/rules.py")) | set(Path("agent_eval").glob("*/research/*.py")))
info = {}
for p in SCRIPTS:
    try: txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception: continue
    reads, writes = set(), set()
    for line in txt.splitlines():
        lits = [m.group("p") for m in LIT.finditer(line)]
        if not lits: continue
        w, r = bool(WRITE.search(line)), bool(READ.search(line))
        for l in lits:
            if l.endswith(".py"): reads.add(l); continue
            if w and not r: writes.add(l)
            elif r and not w: reads.add(l)
            elif w and r:   # both on one line: the literal right after a write call is written
                i = line.find(l); before = line[:i]
                (writes if WRITE.search(before[-60:]) else reads).add(l)
            else: reads.add(l)          # e.g. a path in a dict of inputs
    info[str(p).replace("\\", "/")] = dict(reads=sorted(reads), writes=sorted(writes), lines=txt.count("\n") + 1)
writers = {}
for s, d in info.items():
    for f in d["writes"]: writers.setdefault(f, []).append(s)
# f-string families the literal scan cannot see: (file pattern -> producer script), from the code and log.md
FSTRINGS = {
    "data/q_backtest_ls_": "eda/187_losses.py", "data/dec_base_": "eda/180d_newbase.py", "data/bias_seg_preds": "eda/179x2_bias.py",
    "data/breakout_preds": "eda/179b_breakout.py", "data/geo_nx5r_x_": "eda/218c_geo_nx5r.py", "data/geo_nx5r_fash_": "eda/228_breakout_fashion.py",
    "data/ranges_nx5r": "eda/219_ranges_nx5r.py", "data/q_backtest_nx_": "eda/180q_models.py", "data/q_backtest_fktab_tabpfn_": "eda/196b_tabpfn.py",
    "data/fk_2026_pf2": "eda/196b_tabpfn.py", "data/final_2026_nx9": "eda/181q_final_nx9_2026.py", "data/hpo_tree_": "eda/220_tree_hpo.py",
}
def producers(f):
    if f in writers: return writers[f]
    for k, s in FSTRINGS.items():
        if f.startswith(k): return [s]
    return []
START = ["eda/181q_final_nx9_2026.py", "eda/181q_check_nx9.py", "eda/226_vs_baselines.py", "eda/224_gate_nx5r.py", "eda/222_confidence_new.py", "eda/198ac_confidence_rule.py",
         "eda/221_month_split.py", "eda/226_campaign_test.py", "eda/217_score_new5.py", "eda/219_ranges_nx5r.py", "eda/218_base_nx5r.py", "eda/218c_geo_nx5r.py", "eda/228_breakout_fashion.py"]
EXTRA_READS = {   # reads the scan misses (f-strings, environment variables), from the code and the commands in log.md
    "eda/181q_final_nx9_2026.py": ["data/items.parquet", "data/monthly_item.parquet", "data/features.parquet", "data/daily_item_chan.parquet", "agent_eval/2026-06/predictions_verified.csv",
        "data/q_backtest_ls_huber_routed_2026.parquet", "data/q_backtest_ls_huber_2026.parquet", "data/nn_backtest_preds_finalSec.parquet", "data/nn_backtest_preds_finalSec_s10.parquet",
        "data/fk_2026_pf2c.parquet", "data/fk_2026_pf2d32.parquet", "data/rc_recsys_preds.parquet", "data/geo_nx5r_fash_2026.parquet", "data/ranges_nx5r_fash_2026.csv",
        "data/fk_confidence_2026.csv", "data/q_backtest_nx_w03ec_2026.parquet", "data/fk_2026_w03.parquet", "data/cand_rwxs_final_forecast_q3.csv", "eda/181q_nx9_text.py"],
    "eda/218_base_nx5r.py": ["data/dec_base_ecnsnnhubrq26.parquet", "data/q_backtest_ls_huber.parquet", "data/q_backtest_ls_huber_new5.parquet", "data/q_backtest_ls_huber_2026.parquet", "data/q_backtest_ls_huber_new5_2026.parquet"],
    "eda/218c_geo_nx5r.py": ["data/dec_base_nx5r26.parquet", "data/bias_seg_preds_nx5r26.parquet", "data/breakout_preds_nx5r26_x.parquet"],
    "eda/219_ranges_nx5r.py": ["data/geo_nx5r_fash_x_preds.parquet", "data/geo_nx5r_x_preds.parquet"],
    "eda/179x2_bias.py": ["data/dec_base_nx5r26.parquet"], "eda/179b_breakout.py": ["data/dec_base_nx5r26.parquet"],
    "eda/180d_newbase.py": ["data/dec_base.parquet"],
}
need_s, need_f, seen, queue = set(), {}, set(), list(START)
while queue:
    s = queue.pop()
    if s in seen or s not in info: continue
    seen.add(s); need_s.add(s)
    for f in sorted(set(info[s]["reads"]) | set(EXTRA_READS.get(s, []))):
        if f.endswith(".py"):
            if f not in seen: queue.append(f)
            continue
        pr = producers(f); need_f.setdefault(f, dict(producers=pr, used_by=[]))["used_by"].append(s)
        for q in pr:
            if q not in seen: queue.append(q)
for f in list(need_f): need_f[f]["exists"] = Path(f).exists(); need_f[f]["bytes"] = Path(f).stat().st_size if Path(f).exists() else 0
res = dict(scripts={s: info[s] for s in sorted(need_s)}, files=dict(sorted(need_f.items())), no_producer=sorted(f for f, d in need_f.items() if not d["producers"]),
           missing=sorted(f for f, d in need_f.items() if not d["exists"]))
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print("scripts on the path:", len(need_s), "| files:", len(need_f), f"({sum(d['bytes'] for d in need_f.values()) / 1e6:.0f} MB) | without producer:", len(res["no_producer"]), "| missing:", len(res["missing"]))
