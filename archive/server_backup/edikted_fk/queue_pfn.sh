while kill -0  2>/dev/null; do sleep 20; done
venv_pfn/bin/python 196b_tabpfn.py v3.5-fast feat=dyn_lean > out/tabpfn_feat2_run.txt 2>&1
venv_pfn/bin/python 196b_tabpfn.py v2 feat=dyn >> out/tabpfn_feat2_run.txt 2>&1
