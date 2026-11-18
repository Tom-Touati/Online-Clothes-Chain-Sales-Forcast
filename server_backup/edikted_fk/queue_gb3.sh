while kill -0 72036 2>/dev/null; do sleep 20; done
venv/bin/python 193b_gbdt_variants.py tw15 tw12 winmix win12 jobs=4 > out/gb_remote_run3.txt 2>&1
