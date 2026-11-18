from bt import run
V = [("AMP floor 0.85", {"AMP_CLIP": (0.85, 1.0)}),
     ("AMP floor 1.0", {"AMP_CLIP": (1.0, 1.0)}),
     ("SEAS_ANCHOR 0.5", {"SEAS_ANCHOR": 0.5}),
     ("POW 0.05", {"POW": 0.05}),
     ("POW 0.1", {"POW": 0.1}),
     ("AGE_FADE 0.97", {"AGE_FADE": 0.97}),
     ("RAMP1 1.6 RAMP2 1.25", {"RAMP1": 1.6, "RAMP2": 1.25}),
     ("DRIFT_ADJ 1.02", {"DRIFT_ADJ": 1.02}),
     ("WLONG 0.1", {"WLONG": 0.1}),
     ("CH_K 40", {"CH_K": 40})]
for n, o in V: run(n, o)
