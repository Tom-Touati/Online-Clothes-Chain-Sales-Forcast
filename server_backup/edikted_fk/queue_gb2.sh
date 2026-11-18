while kill -0  2>/dev/null; do sleep 20; done
venv/bin/python 193b_gbdt_variants.py wvol05 wvol1 wvol03 jobs=3 > out/gb_remote_run2.txt 2>&1
