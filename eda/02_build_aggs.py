import sys, duckdb
p = sys.argv[1]
con = duckdb.connect(); con.execute("SET memory_limit='1500MB'; SET threads=4")
con.execute(f"CREATE VIEW T AS SELECT * FROM read_parquet('{p}')")
# daily item x class x channel-group
con.execute("""COPY (
 SELECT style_color, ny_date, class,
        CASE WHEN class='ONLINE' THEN o_type WHEN class='DROP' THEN o_type ELSE 'RETAIL' END AS chan,
        cancelled,
        sum(qty_ordered) qty, count(*) n_lines, count(distinct order_name) n_orders,
        sum(avg_discounts*qty_ordered) disc_w, count(distinct size) n_sizes, max(qty_ordered) max_line_qty
 FROM T GROUP BY ALL) TO 'data/daily_item_chan.parquet' (FORMAT parquet)""")
# item master + consistency checks
con.execute("""COPY (
 SELECT style_color, any_value(product_name) product_name, count(distinct product_name) n_names,
        any_value(product_type) product_type, count(distinct product_type) n_types,
        any_value(product_collection) product_collection, count(distinct product_collection) n_coll,
        min(published_date) published_date, count(distinct published_date) n_pub,
        any_value(color) color, any_value(color_num) color_num, count(distinct color_num) n_colnum,
        any_value(occasion) occasion, count(distinct coalesce(occasion,'')) n_occ,
        count(distinct sku) n_skus, count(distinct size) n_sizes,
        min(ny_date) first_sale, max(ny_date) last_sale,
        sum(CASE WHEN NOT cancelled THEN qty_ordered ELSE 0 END) qty
 FROM T GROUP BY 1) TO 'data/items.parquet' (FORMAT parquet)""")
# daily size mix
con.execute("""COPY (SELECT style_color, date_trunc('month', ny_date) m, size, sum(qty_ordered) qty FROM T WHERE NOT cancelled GROUP BY ALL) TO 'data/monthly_item_size.parquet' (FORMAT parquet)""")
# daily store
con.execute("""COPY (SELECT ny_date, class, otype, cancelled, sum(qty_ordered) qty, count(distinct order_name) n_orders, sum(avg_discounts*qty_ordered) disc_w, max(qty_ordered) max_line FROM T GROUP BY ALL) TO 'data/daily_otype.parquet' (FORMAT parquet)""")
print(con.execute("select count(*) from 'data/daily_item_chan.parquet'").fetchall())
# artefact probes
q = lambda s: print(con.execute(s).fetchdf().to_string())
q("SELECT sum((ny_date <> CAST(ny_timestamp AS DATE))::int) date_mismatch, sum((avg_discounts<0)::int) neg_disc, sum((avg_discounts>=1)::int) disc_ge1, sum((avg_discounts=0)::int) disc0 FROM T")
q("SELECT class, cancelled, round(avg(avg_discounts),3) d, count(*) n FROM T GROUP BY ALL ORDER BY 1,2")
q("SELECT class, count(distinct order_name) orders, round(sum(qty_ordered)/count(distinct order_name),2) units_per_order, max(qty_ordered) maxq FROM T GROUP BY 1")
q("WITH o AS (SELECT order_name, class, o_type, sum(qty_ordered) u FROM T GROUP BY ALL) SELECT class, quantile_cont(u,[0.5,0.9,0.99,0.999,1]) qs FROM o GROUP BY 1")
q("WITH d AS (SELECT order_name, sku, ny_timestamp, count(*) c FROM T GROUP BY ALL HAVING count(*)>1) SELECT count(*) dup_groups, sum(c) dup_lines FROM d")
q("SELECT class, sum((qty_ordered>=5)::int) big_lines, sum(CASE WHEN qty_ordered>=5 THEN qty_ordered END) big_qty FROM T GROUP BY 1")
