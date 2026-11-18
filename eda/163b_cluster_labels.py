"""D205 (fork): the D158 multi-view cluster labels at every origin (eda/163 re-run unchanged, k-means seed 0), saved per item for the
cluster-history features (eda/182b). eda/163's own outputs (cluster_feats, clusters_2026-06) are not rewritten.
Writes data/fk_cluster_labels.parquet (origin, style_color, cluster; -1 = not alive in the last 12 months)."""
src = open("eda/163_clusters.py", encoding="utf-8").read()
src = src.replace("    rows.append(f)\n", "    rows.append(f); LAB.append(pd.DataFrame({'origin': o, 'style_color': items, 'cluster': lab}))\n")
src = src.replace("rows, t0 = [], time.time()", "rows, t0, LAB = [], time.time(), []")
src = src.replace('pd.DataFrame({"style_color": items, "cluster": lab, **{f"x{j}": X[:, j] for j in range(X.shape[1])}}).to_parquet("data/clusters_2026-06.parquet", index=False)', "pass")
src = src.replace('pd.concat(rows, ignore_index=True).to_parquet("data/cluster_feats.parquet", index=False); print("saved")',
                  'pd.concat(LAB, ignore_index=True).to_parquet("data/fk_cluster_labels.parquet", index=False); print("saved labels")')
assert "fk_cluster_labels" in src and "LAB.append" in src and "cluster_feats.parquet" not in src.split("WF = ")[1]
exec(compile(src, "eda/163_clusters.py", "exec"))
