"""D282 (fork): one-off patch after a comment on the error study ("can we have features saying what was the recent group total? and
if more stores had opened"): two paragraphs with the eda/198m result in the panel 'Is the total wrong, or the split between items?'
and their tokens in the builder. Idempotent."""
p = "eda/error_study_template.html"; s = open(p, encoding="utf-8").read()
if "{{gs_ng}}" in s: raise SystemExit("already patched")
a = "Our group totals are off by about ±{{ce_sd_w}} weighted by units.</p>\n"; assert a in s
NEW = '''      <p><b>What about the group's recent sales?</b> Also tested after a comment: no measurable gain. {{gs_ng}} inputs: for each item's type × look group, its look and its product type, the sales of the last three months, the trend against the three months before, the newest-month turn and the change on a year ago. The year-on-year change leans the right way (rank correlation with the miss {{gs_rho_yoy}}, same sign at {{gs_same_yoy}} of 13 origins). In a time-honest correction the block gives {{gs_g_w}} against {{gs_ref_w}} without it, and {{gs_g_hit}} within ±20% against {{gs_ref_hit}}. The difference ({{gs_ci_lo}} to {{gs_ci_hi}}) includes zero, and the submitted system scores {{ratio_recent}}.</p>
      <p><b>And new stores opening?</b> No gain either. {{gs_ns}} inputs: own stores trading at the origin, opened in the last three months and opening in the forecast window, each also weighted by the item's own-store share. They carry nothing at item level (rank correlation {{gs_rho_open}}; {{gs_s_w}} with them). Stores do drive volume: year on year, own-store sales grew {{gs_yoy_sales}}× as the stores grew {{gs_yoy_stores}}×. But an opening is the same fact for every item at an origin, so it cannot say which items gain. What does say it is the number of stores each item is itself added to during the window (rank correlation {{gs_rho_hind}}, seen afterwards). That is the allocation plan: the business holds it, the data does not.</p>
'''
s = s.replace(a, a + NEW, 1); open(p, "w", encoding="utf-8").write(s)

b = "eda/198g_build_error_study.py"; t = open(b, encoding="utf-8").read()
x = '    built="17 Nov 2026",'; assert x in t
add = '''    gs_ng=str(GS["n_group"]), gs_ns=str(GS["n_store1"]), gs_rho_yoy=sf(_gsig["g_tl_yoy"]["rho"], 2), gs_same_yoy=str(_gsig["g_tl_yoy"]["same"]), gs_g_w=f3(GS["rows"][2]["wape"]), gs_ref_w=f3(GS["rows"][1]["wape"]),
    gs_g_hit=pc(GS["rows"][2]["hit"], 1), gs_ref_hit=pc(GS["rows"][1]["hit"], 1), gs_ci_lo=sf(GS["rows"][2]["ci"][0], 4), gs_ci_hi=sf(GS["rows"][2]["ci"][1], 4), gs_rho_open=sf(_gsig["expo_future"]["rho"], 2), gs_s_w=f3(GS["rows"][4]["wape"]),
    gs_yoy_sales=_r(float(np.median([v[0] for v in _yy])), 1), gs_yoy_stores=_r(float(np.median([v[1] for v in _yy])), 1), gs_rho_hind=sf(_hind["h_stores_chg"]["rho"], 2),'''
t = t.replace(x, x + "\n" + add)
y = 'T = dict(\n'; assert y in t
pre = '''GS = J["groupstore"]; _gp = {d["m"]: d for d in GS["per_store"]}; _gsig = {d["key"]: d for d in GS["sig"]}; _hind = {d["key"]: d for d in J["sig_hind"]}
_yy = [(_gp[m]["retail"] / _gp[q]["retail"], _gp[m]["stores"] / _gp[q]["stores"]) for m in _gp for q in [str(pd.Period(m, "M") - 12)] if q in _gp]   # same month a year earlier
'''
t = t.replace(y, pre + y, 1); open(b, "w", encoding="utf-8").write(t)
print("patched")
