cd /c/Users/tomto/workspaces/edikted
until [ $(grep -c "extra features" eda/q_fashA_run.txt) -ge 3 ] && [ $(grep -c "extra features" eda/q_fashB_run.txt) -ge 3 ]; do sleep 30; if grep -q Traceback eda/q_fashA_run.txt eda/q_fashB_run.txt; then break; fi; done
.venv/Scripts/python.exe eda/193c_tune_select.py tune jobs=4 > eda/tune_output.txt 2>&1 &
.venv/Scripts/python.exe eda/193c_tune_select.py select jobs=4 > eda/select_output.txt 2>&1 &
.venv/Scripts/python.exe eda/193b_gbdt_variants.py seeds5 jobs=2 > eda/gb_seeds_run.txt 2>&1
wait
.venv/Scripts/python.exe eda/193b_gbdt_variants.py tuned jobs=4 > eda/gb_tuned_run.txt 2>&1 &
.venv/Scripts/python.exe eda/193b_gbdt_variants.py sel jobs=4 > eda/gb_sel_run.txt 2>&1 &
wait
.venv/Scripts/python.exe eda/193b_gbdt_variants.py winmix jobs=4 > eda/gb_winmix_run.txt 2>&1 &
.venv/Scripts/python.exe eda/193b_gbdt_variants.py tuned+sel+seeds5 jobs=4 > eda/gb_combo_run.txt 2>&1 &
wait
echo QUEUE DONE
