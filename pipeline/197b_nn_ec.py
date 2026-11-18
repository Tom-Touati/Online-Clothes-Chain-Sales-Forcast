"""D272 (fork; user request via Main): the sales-curve NN (pipeline/57_embed_nn.py seq=1, the S13 / finalS family) with Main's D253 inputs
added as numeric features: EV2 (15 more event windows + calendar counts, c_pay dropped) and CURVE (launch curves), from
data/nf2_feats.parquet, joined on (origin, style_color) and repeated over the four horizons. Everything else is pipeline/57 unchanged
(executed from its source with one inserted block). Usage as pipeline/57, e.g.
  python pipeline/197b_nn_ec.py seq=1 seeds=3 seed0=0 tag=_S13ec          (backtest, 13 origins)
  python pipeline/197b_nn_ec.py seq=1 seeds=3 seed0=0 origins=2026-06 tag=_finalSec   (deliverable)
ec=0 runs the unchanged model (control on the same machine)."""
import sys, json
ARGS0 = dict(a.split("=") for a in sys.argv[1:]); EC = int(ARGS0.pop("ec", 1)); sys.argv = sys.argv[:1] + [f"{k}={v}" for k, v in ARGS0.items()]
src = open("pipeline/57_embed_nn.py", encoding="utf-8").read()
inj = '''
if EC_:
    _B = json.load(open("data/nf2_meta.json")); _X = [c for c in _B["EV2"] if c != "c_pay"] + _B["CURVE"]
    F = F.merge(pd.read_parquet("data/nf2_feats.parquet")[["origin", "style_color"] + _X], on=["origin", "style_color"], how="left"); NUM += _X
    print("D272: added", len(_X), "EV2 + CURVE features", flush=True)
'''
anchor = "NUM += OCC\n"; assert anchor in src
src = src.replace(anchor, anchor + inj, 1)
g = {"__name__": "__main__", "EC_": EC, "json": json}
exec(compile(src, "pipeline/57_embed_nn.py", "exec"), g)
