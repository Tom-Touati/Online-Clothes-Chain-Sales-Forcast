"""D423: the tree member at the June 2026 origin with the launch routing (D389h / D389i), as a script (it was an inline step in D389i).
Items under six months old at the origin use the Huber tree trained with new items' rows counted five times (pipeline/187_losses.py variant
huber_new5, ORIGS=2026-06 OSUF=_2026); the others use the plain Huber tree (variant huber). Age at the origin is read from the correction
base table (data/dec_base_nx5r26.parquet, pipeline/218_base_nx5r.py), so the routing is the one the corrections were fitted on.
Writes data/q_backtest_ls_huber_routed_2026.parquet (the plain tree's columns, q50 routed, flag routed_new5)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
base = pd.read_parquet(DATA / "dec_base_nx5r26.parquet"); age = base[base.origin == "2026-06"].set_index("style_color").age_m
plain = pd.read_parquet(DATA / "q_backtest_ls_huber_2026.parquet").drop_duplicates("style_color").set_index("style_color")
launch = pd.read_parquet(DATA / "q_backtest_ls_huber_new5_2026.parquet").drop_duplicates("style_color").set_index("style_color").q50
new = age.reindex(plain.index).fillna(99) < 6
out = plain.copy(); out["q50"] = np.where(new, launch.reindex(plain.index), plain.q50); out["routed_new5"] = new.values
out.reset_index().to_parquet(DATA / "q_backtest_ls_huber_routed_2026.parquet" if "--write" in sys.argv else DATA / "_check_routed_2026.parquet", index=False)
print("items", len(out), "| routed to the launch tree:", int(new.sum()))
