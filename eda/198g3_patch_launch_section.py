"""D279 (fork): one-off patch adding the per-store launch-curve test (user comment on the page) to eda/error_study_template.html and
its tokens to eda/198g_build_error_study.py. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if 'id="launch-curves"' in s: raise SystemExit("already patched")


def rep(a, b):
    global s
    assert a in s, a[:80]; s = s.replace(a, b, 1)


PANEL = '''  <div class="panel" id="launch-curves" style="margin-top:18px"><h3>Per-store launch curves as inputs</h3>
    <p>Tested after a comment on this page: every item has a launch curve at each place it reaches, and the places it reached first could inform the rest. The data holds {{ln_arrivals}} such arrivals across {{ln_stores}} own stores, {{ln_partners}} partners, edikted.com and TikTok. Each arrival gets a launch index: its first-14-day units against the norm of that location.</p>
    <div class="tscroll"><table><thead><tr><th>Test</th><th>Result</th></tr></thead><tbody>
      <tr><td>As inputs to the item forecast: {{ln_feats}} features known at the origin (stores reached and still to reach, launch index per store, first and latest store, rate per door, partner, web and TikTok indices)</td><td>Rank correlation with the forecast's residual: {{ln_rho_all}} at most over all items, {{ln_rho_gap}} for items in some stores but not all (both the web launch index). A time-honest correction with them scores {{ln_corr}} against {{ln_ref}} without (interval {{ln_ci_lo}} to {{ln_ci_hi}}). No gain.</td></tr>
      <tr><td>Predicting the roll-out itself: will the item gain two or more stores in the window?</td><td>Predictable from level, age and current stores (AUC {{ln_auc0}}); the launch curves take it to {{ln_auc1}}.</td></tr>
      <tr><td>A later store from earlier stores: arrivals at least 14 days after an earlier store's</td><td>Correlation between its launch index and the earlier stores' {{ln_cross}}; out-of-sample R² about zero. A store sells a median of {{ln_u14}} units of an item in its first 14 days, and an arrival is only visible as the first sale, so a slow store looks like a late arrival.</td></tr>
      <tr><td>Same month, different places</td><td>An item's units correlate {{ln_ss}} between pairs of own stores, {{ln_wrate}} between the web and its rate per store, and {{ln_wt}} between the web and TikTok. The web starts first for {{ln_webfirst}} of items, by a median of {{ln_lead}} days.</td></tr>
    </tbody></table></div>
    <p style="margin-top:10px">Appeal does carry between places, so other stores are informative. For the four-month item total it changes nothing, because the models already see that appeal through the item's level and momentum: the web is two thirds of units and leads the stores by a month. The place it would pay is store level, for allocation and replenishment, and that needs receipts: the date and units each store received, so that sell-through (sold ÷ received) replaces first-sale dating. One cohort effect remains visible: items in no own store, or in under half the fleet, are under-forecast by {{ln_b_none}} and {{ln_b_half}}; together they are {{ln_u_part}} of units.</p></div>
'''
rep('</section>\n\n<section id="overfit">', PANEL + '</section>\n\n<section id="overfit">')
rep("Until the plan exists as data, use each launch's purchase-order quantity: one number that carries the merchants' conviction.</dd>",
    "Until the plan exists as data, use each launch's purchase-order quantity: one number that carries the merchants' conviction. Ask for receipts per store as well (date and units): sales alone cannot date a store arrival, and per-store launch curves built from first sales added nothing.</dd>")
open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    ln_arrivals=it(LN["cross_store_arrivals"]), ln_stores=str(LN["n_stores"]), ln_partners=str(LN["n_partners"]), ln_feats=str(len(LN["sig_all"])),
    ln_rho_all=sf(LN["sig_all"][0]["rho"], 2), ln_rho_gap=sf(LN["sig_gap"][0]["rho"], 2), ln_corr=f3(LN["correction"][2]["wape"]), ln_ref=f3(LN["correction"][1]["wape"]),
    ln_ci_lo=sf(LN["boot_launch_vs_ref"][0], 4), ln_ci_hi=sf(LN["boot_launch_vs_ref"][1], 4), ln_auc0=f2(LN["rollout"]["auc_base"]), ln_auc1=f2(LN["rollout"]["auc_launch"]),
    ln_cross=sf(LN["cross_store"]["corr"], 2), ln_u14=it(LN["cross_store"]["med_u14"]), ln_ss=f2(LN["carry"]["store_store"]), ln_wrate=f2(LN["carry"]["web_vs_rate_per_store"]), ln_wt=f2(LN["carry"]["web_tiktok"]),
    ln_webfirst=pc(LN["carry"]["web_first"]), ln_lead=it(LN["carry"]["lead_median"]), ln_b_none=pc(LN["rollout"]["by_gap"][0]["bias"]), ln_b_half=pc(LN["rollout"]["by_gap"][1]["bias"]),
    ln_u_part=pc(LN["rollout"]["by_gap"][0]["units"] + LN["rollout"]["by_gap"][1]["units"]),'''
t = t.replace(x, x + "\n" + add)
y = 'c26 = pd.read_csv("data/final_2026_nx2_cascade.csv", index_col=0)'; assert y in t
t = t.replace(y, 'LN = J["launch"]; _ar = pd.read_parquet("data/fk_store_arrivals.parquet"); LN["cross_store_arrivals"] = len(_ar); LN["n_stores"] = int(_ar[_ar.kind == "store"]["loc"].nunique()); LN["n_partners"] = int(_ar[_ar.kind == "partner"]["loc"].nunique())\n' + y)
open(b, "w", encoding="utf-8").write(t)
print("patched")
