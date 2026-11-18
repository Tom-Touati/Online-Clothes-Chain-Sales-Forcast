"""Run a script at below-normal priority with 3 threads (user: "you can use the laptop but dont let it lag the computer").
Usage: python eda/lowprio.py <script.py> [args]"""
import os, sys, ctypes, runpy
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[k] = "3"
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
script = sys.argv[1]; sys.argv = sys.argv[1:]; runpy.run_path(script, run_name="__main__")
