while pgrep -f "193b_gbdt_variants.py tw15" >/dev/null; do sleep 30; done
venv/bin/python 193b_gbdt_variants.py sel+wvol03 jobs=4 > out/gb_remote_run5.txt 2>&1
