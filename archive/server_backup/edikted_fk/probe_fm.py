import torch, numpy as np, inspect, time
import chronos
print("chronos exports:", [n for n in dir(chronos) if "Pipeline" in n])
from chronos import BaseChronosPipeline
x = [torch.tensor(np.abs(np.sin(np.arange(60) / 4)) * 10 + i, dtype=torch.float32) for i in range(3)]
for name in ["amazon/chronos-bolt-base", "amazon/chronos-2"]:
    t0 = time.time()
    try:
        p = BaseChronosPipeline.from_pretrained(name, device_map="cuda")
        print(name, type(p).__name__, "load", round(time.time() - t0, 1), "s")
        print("  predict_quantiles sig:", inspect.signature(p.predict_quantiles))
        q, m = p.predict_quantiles(x, prediction_length=6, quantile_levels=[0.1, 0.5, 0.9])
        print("  quantiles", [tuple(a.shape) for a in q] if isinstance(q, list) else tuple(q.shape))
        if hasattr(p, "embed"):
            e = p.embed(x); e0 = e[0] if isinstance(e, tuple) else e
            print("  embed:", type(e0), [tuple(a.shape) for a in e0] if isinstance(e0, list) else tuple(e0.shape))
        else: print("  no embed method")
    except Exception as ex: print(name, "FAILED", repr(ex)[:300])
import timesfm
print("timesfm exports:", [n for n in dir(timesfm) if not n.startswith("_")][:40])
try:
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    pf, qf = m.forecast(horizon=6, inputs=[a.numpy() for a in x]); print("timesfm 2.5 point", pf.shape, "quantiles", qf.shape)
except Exception as ex: print("timesfm FAILED", repr(ex)[:400])
