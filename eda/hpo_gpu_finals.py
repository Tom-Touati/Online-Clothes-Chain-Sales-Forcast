"""D325 / D326 (fork; overnight tuning): the final runs for a tuned member, ON THE GPU in ~/edikted_fk/hpo, from out/hpo_<model>_best.json:
  nn      all 13 origins with the best settings, seeds 0-2 and 10-12 (as the submitted 6-seed member S13ecx6), and seeds 0-2 with last year's
          same window removed (LYREMOVE=1, for the D315 / D324 gate)
  tabpfn  all 13 origins with the best settings, and again with last year's window removed
If the best trial is the current settings, nothing is rerun (the stored member stands). Usage: python hpo_gpu_finals.py nn|tabpfn"""
import sys, os, json, subprocess
MODEL = sys.argv[1]; B = json.load(open(f"out/hpo_{MODEL}_best.json")); best, base = B["best"], B["base"]
KEYS = ["dim", "epochs", "lr", "wd", "drop", "pen", "inter"] if MODEL == "nn" else ["est", "temp", "cap", "abs_"]
same = all(str(best[k]) == str(base[k]) for k in KEYS); print("best", {k: best[k] for k in KEYS}, "| current", {k: base[k] for k in KEYS}, "| same as current:", same, flush=True)
if same: sys.exit(0)
def fmt(k, v): return f"{k}={int(v) if k in ('dim', 'epochs', 'inter', 'est', 'cap', 'abs_') else v}".replace("abs_=", "abs=")
P = [fmt(k, best[k]) for k in KEYS]; env = dict(os.environ); envly = dict(os.environ, LYREMOVE="1")
if MODEL == "nn":
    for args, e in [(["seeds=3", "seed0=0", "tag=_HPOa"], env), (["seeds=3", "seed0=10", "tag=_HPOb"], env), (["seeds=3", "seed0=0", "tag=_HPOly"], envly)]:
        r = subprocess.run(["../venv/bin/python", "eda/197b_nn_ec.py", "seq=1"] + args + P, env=e, capture_output=True, text=True); print(args[-1], "exit", r.returncode, r.stderr[-300:] if r.returncode else "", flush=True)
else:
    for out, e in [("out/hpo_pf_final.parquet", env), ("out/hpo_pf_final_ly.parquet", envly)]:
        r = subprocess.run(["../venv_tab/bin/python", "eda/196b_tabpfn.py", "v2", "feat=dyn", f"out={out}"] + P, env=e, capture_output=True, text=True); print(out, "exit", r.returncode, (r.stdout + r.stderr)[-300:] if r.returncode else "", flush=True)
