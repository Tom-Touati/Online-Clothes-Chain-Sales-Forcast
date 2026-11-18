import sys, duckdb
p = sys.argv[1]
con = duckdb.connect()
con.execute("SET memory_limit='1500MB'; SET threads=4")
con.execute(f"CREATE VIEW T AS SELECT * FROM read_parquet('{p}')"); q = lambda s: con.execute(s).fetchdf()
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
print(q("DESCRIBE SELECT * FROM T"))
print(q("SELECT count(*) n, count(distinct order_name) orders, count(distinct line_id) lines, count(distinct style_color) items, count(distinct sku) skus, min(ny_date) d0, max(ny_date) d1, sum(qty_ordered) qty, min(qty_ordered) qmin, max(qty_ordered) qmax, avg(qty_ordered) qavg FROM T"))
print(q("SELECT * FROM T USING SAMPLE 5").T)
for c in ["class","otype","o_type","store","cancelled","occasion","color_num","size","country","state"]:
    try:
        print("\n--", c); print(q(f"SELECT {c}, count(*) n, sum(qty_ordered) qty FROM T GROUP BY 1 ORDER BY qty DESC LIMIT 25"))
    except Exception as e: print("ERR", c, e)
print(q("SELECT " + ", ".join(f"sum(({c} IS NULL)::int) AS {c}" for c in ["order_name","line_id","ny_date","style_color","sku","size","color","product_name","product_type","product_collection","published_date","qty_ordered","avg_discounts","class","otype","o_type","store","cancelled","occasion","color_num"]) + " FROM T").T)
print(q("SELECT qty_ordered, count(*) n FROM T GROUP BY 1 ORDER BY 1 LIMIT 40"))
print(q("SELECT quantile_cont(avg_discounts,[0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1]) dq, quantile_cont(qty_ordered,[0.5,0.9,0.99,0.999,0.9999,1]) qq FROM T"))
