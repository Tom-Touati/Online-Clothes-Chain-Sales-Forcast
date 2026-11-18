while pgrep -f "196b_tabpfn.py v2 est=32" >/dev/null; do sleep 20; done
venv_pfn/bin/python 196b_tabpfn.py v2 feat=dyn orig=2026-06 > out/tabpfn_2026_e8_run.txt 2>&1
