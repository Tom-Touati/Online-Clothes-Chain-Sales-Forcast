while kill -0 54745 2>/dev/null; do sleep 20; done
venv_pfn/bin/python 196c_tabpfn_ft.py v2 tl=480 > out/tabpfn_ft_run.txt 2>&1
