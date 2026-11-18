import sys; sys.path.insert(0,"research"); from bt import run
run("v6 DRIFT_ADJ 1.03")
run("DRIFT_ADJ 1.0", DRIFT_ADJ=1.0); run("DRIFT_ADJ 1.02", DRIFT_ADJ=1.02); run("DRIFT_ADJ 1.04", DRIFT_ADJ=1.04); run("DRIFT_ADJ 1.05", DRIFT_ADJ=1.05)
run("1.03 + W14 0.5", W14=0.5); run("1.03 + POW 0.03", POW=0.03); run("1.03 + RET_POW 0.4", RET_POW=0.4)
run("1.03 + SEAS_POW 0.8", SEAS_POW=0.8); run("1.03 + other 0.9^h", OTHER_PATH=False)
