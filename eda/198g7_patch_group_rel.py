"""D282 (fork): one-off patch after the follow-up comment "perhaps show that also relative to all other groups": one sentence with
the relative-feature result (eda/198m) in the group paragraph of the error study, and its tokens in the builder. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if "{{gs_nr}}" in s: raise SystemExit("already patched")
a = "includes zero, and the submitted system scores {{ratio_recent}}.</p>"; assert a in s
s = s.replace(a, a[:-4] + " Measured against all other groups instead (the group's share of all units, and its trend, newest-month turn and year-on-year change against the rest; {{gs_nr}} inputs) it is the same: {{gs_r_w}}, and {{gs_r_hit}} within ±20%.</p>", 1)
open(p, "w", encoding="utf-8").write(s)
b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
t = t.replace(x, x + '\n    gs_nr=str(GS["n_rel"]), gs_r_w=f3(GS["rows"][6]["wape"]), gs_r_hit=pc(GS["rows"][6]["hit"], 1),')
open(b, "w", encoding="utf-8").write(t); print("patched")
