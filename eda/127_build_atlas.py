"""Build the atlas page (D132): data/project_map.json (eda/19, then eda/124 for the evaluation section) inserted into eda/atlas_template.html
-> eda/atlas.html (the file published as the 'Edikted Forecast Atlas' artifact). '</' inside the JSON is escaped so the data block cannot end
the script tag early.
D275: the template was brought to the submitted system of D274 by eda/127b_atlas_patch_d275.py (idempotent; already applied)."""
import json, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
if "--no-data" not in sys.argv:
    for s in ["eda/19_project_map_data.py", "eda/124_atlas_eval_data.py"]:
        r = subprocess.run([sys.executable, s], capture_output=True, text=True, encoding="utf-8"); print(s, "exit", r.returncode, r.stdout[-300:], r.stderr[-800:])
        assert r.returncode == 0
J = json.load(open("data/project_map.json", encoding="utf-8"))
J["features"] = json.load(open("data/feature_inventory.json", encoding="utf-8"))   # D343: every model input (eda/206)
t = open("eda/atlas_template.html", encoding="utf-8").read()
assert t.count("__DATA__") == 1
html = t.replace("__DATA__", json.dumps(J, ensure_ascii=False).replace("</", "<\\/"))
open("eda/atlas.html", "w", encoding="utf-8").write(html); print("wrote eda/atlas.html", len(html) // 1024, "KB | log entries:", len(J["log"]), "| eval:", "eval" in J)
