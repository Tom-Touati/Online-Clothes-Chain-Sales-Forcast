while pgrep -f 196c_tabpfn_ft >/dev/null; do sleep 30; done
venv_pfn/bin/python 196b_tabpfn.py v2 est=32 feat=sel > out/tabpfn_sel_run.txt 2>&1
