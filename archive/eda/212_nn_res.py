"""Fork 2 (user: "ensembling the tabpfn and the recommender twin tree as first layer, then training the rest of the models as secondary stacked
on the residuals"). The NN member (eda/197b: eda/57 seq=1 + EV2 + CURVE) trained on the residual of the first layer: base b = mean in z of
TabPFN's and the recommender twin's walk-forward predictions (data/tabpfn_wf.parquet, data/rc_recsys_preds_x_wf.parquet; rcol=f_aug_none_gbm
or f_aug_plan_gbm), z = log((total + 1) / (4 level + 1)). The monthly target rlog = log((y_m + 1) / (level + 1)) is shifted by b (the same
shift for the four months), and b is added back to the predicted quantiles. Rows without a base (origins before 2024-07) are dropped.
Usage: python eda/212_nn_res.py seq=1 seeds=3 seed0=0 rcol=f_aug_none_gbm tag=_resNN"""
import sys, json
ARGS0 = dict(a.split("=") for a in sys.argv[1:]); RCOL = ARGS0.pop("rcol", "f_aug_none_gbm"); sys.argv = sys.argv[:1] + [f"{k}={v}" for k, v in ARGS0.items()]
src = open("eda/57_embed_nn.py", encoding="utf-8").read()
inj = '''
_B = json.load(open("data/nf2_meta.json")); _X = [c for c in _B["EV2"] if c != "c_pay"] + _B["CURVE"]
F = F.merge(pd.read_parquet("data/nf2_feats.parquet")[["origin", "style_color"] + _X], on=["origin", "style_color"], how="left"); NUM += _X
_w = pd.read_parquet("data/tabpfn_wf.parquet"); _r = pd.read_parquet("data/rc_recsys_preds_x_wf.parquet")[["origin", "style_color", RCOL_]]
_lv = pd.read_parquet("data/fk_explain_table.parquet", columns=["origin", "style_color", "level"]).rename(columns={"level": "_lvl4"})
_b = _w.merge(_r, on=["origin", "style_color"], how="outer").merge(_lv, on=["origin", "style_color"], how="left")
_b["_b"] = np.nanmean(np.stack([_b.z_pf.values, np.log((_b[RCOL_].values + 1) / (4 * _b._lvl4.values + 1))]), 0)
F = F.merge(_b[["origin", "style_color", "_b"]], on=["origin", "style_color"], how="left"); F = F[F._b.notna()].reset_index(drop=True)
F["rlog"] = F.rlog - F._b
print("residual NN: base rows", int(F._b.notna().sum()), "| origins", F.origin.min(), "..", F.origin.max(), flush=True)
'''
anchor = "NUM += OCC\n"; assert anchor in src; src = src.replace(anchor, anchor + inj, 1)
a2 = "te[qn] = ((te.level + 1) * np.exp(Q[:, j]) - 1).clip(lower=0)"; assert a2 in src; src = src.replace(a2, "te[qn] = ((te.level + 1) * np.exp(Q[:, j] + te._b.values) - 1).clip(lower=0)", 1)
g = {"__name__": "__main__", "json": json, "RCOL_": RCOL}
exec(compile(src, "eda/57_embed_nn.py", "exec"), g)
