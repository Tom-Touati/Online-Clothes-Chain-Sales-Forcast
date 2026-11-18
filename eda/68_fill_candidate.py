"""Candidate workbook (D085/D086): eda/33_fill_workbook_v3.py run from its own source on data/cand_nn_final_forecast_q4.csv,
with the model-description text updated for the NN blend; then eda/34 consistency checks on the candidate file + workbook.
Writes Edikted_Forecasting_Exercise_candidate_nn.xlsx; the submitted workbook is not touched."""
import sys, io, contextlib
sys.stdout.reconfigure(encoding="utf-8")
SRC, DST = "Edikted_Forecasting_Exercise.xlsx", "Edikted_Forecasting_Exercise_candidate_nn.xlsx"
s = open("eda/33_fill_workbook_v3.py", encoding="utf-8").read()
REP = [('read_csv("data/final_forecast_q4.csv"', 'read_csv("data/cand_nn_final_forecast_q4.csv"'),
       ("A quantile model of each item's four-month total, plus each item's own year-on-year trend where it has one.",
        "Two models of each item's four-month total, combined, plus each item's own year-on-year trend where it has one."),
       ("Inputs: the last six months relative to level, age, product type, start month, occasion tags, colour number, 7- and 28-day pace, channel mix, discount, like-for-like seasonal ratios and the tagger attributes.\\n",
        "Inputs: the last six months relative to level, age, product type, start month, occasion tags, colour number, 7- and 28-day pace, channel mix, discount, like-for-like seasonal ratios and the tagger attributes. "
        "The second model is a neural network with the same inputs that learns embeddings for the product hierarchy (product group, type, style, item; a new item falls back to its style and type), reads the last 26 weeks of sales with a small recurrent layer, and predicts monthly quantiles. "
        "The answer distribution is centred on the geometric mean of the two models' medians.\\n"),
       ("The quantile median scores WAPE 0.32 on the nine recent origins (0.34 for my earlier monthly model).",
        "The combined median scores WAPE 0.31 on the nine recent origins (0.32 for the quantile model alone, 0.34 for my earlier monthly model), and in a simulation of daily buying (3-week lead time, one order per item per week) it lost less profit than the quantile model alone at 10 of 10 replayed origins."),
       ("about 56-60%, 41% and 27% of the time", "about 52-66%, 39-44% and 26-28% of the time"),     # data/cand_nn_22_output.txt section 4 (all items / targets)
       ('src, dst = sys.argv[1], sys.argv[2]', f'src, dst = "{SRC}", "{DST}"')]
for a, b in REP:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
exec(compile(s, "eda/33_cand", "exec"), {"__name__": "__main__"})
c = open("eda/34_final_consistency.py", encoding="utf-8").read().replace('read_csv("data/final_forecast_q4.csv"', 'read_csv("data/cand_nn_final_forecast_q4.csv"')
sys.argv = ["x", DST]; buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(c, "eda/34_cand", "exec"), {"__name__": "__main__"})
out = buf.getvalue(); open("data/cand_nn_34_output.txt", "w", encoding="utf-8").write(out)
print(out[out.find("== D"):][:1500])
