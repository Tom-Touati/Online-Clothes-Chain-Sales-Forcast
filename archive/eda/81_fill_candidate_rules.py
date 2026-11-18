"""Candidate workbook centred on rules + quantile model + NN (D105): eda/68_fill_candidate.py run from its own source with the
input switched to data/cand_rnn_final_forecast_q4.csv and the model text describing the three-way centre.
Writes Edikted_Forecasting_Exercise_candidate_rules.xlsx; the submitted and NN-candidate workbooks are not touched."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
s = open("eda/68_fill_candidate.py", encoding="utf-8").read()
E = [('"candidates/Edikted_Forecasting_Exercise_candidate_nn.xlsx"', '"candidates/Edikted_Forecasting_Exercise_candidate_rules.xlsx"'),
     ('"Two models of each item\'s four-month total, combined, plus', '"Two models and a rule set for each item\'s four-month total, combined, plus'),
     (r'''"The answer distribution is centred on the geometric mean of the two models' medians.\\n"''',
      '"A third, rule-based forecast comes from explicit planning rules (recent daily rate x site and channel plan x like-for-like seasonal index by product type x young-item ramp and fade x occasion indices), written from the data up to each month end and revised monthly against how earlier forecasts did. "\n'
      r'''        "The answer distribution is centred on the geometric mean of the three medians.\\n"'''),
     ('"The combined median scores WAPE 0.31 on the nine recent origins (0.32 for the quantile model alone, 0.34 for my earlier monthly model), and in a simulation of daily buying (3-week lead time, one order per item per week) it lost less profit than the quantile model alone at 10 of 10 replayed origins."',
      '"The combined median scores WAPE 0.30 on the nine recent origins (0.31 without the rules, 0.33 for the quantile model alone, 0.34 for my earlier monthly model)."'),
     ('"about 52-66%, 39-44% and 26-28% of the time"', '"about 53-59%, 40-43% and 28-30% of the time"'),     # data/cand_rnn_22_output.txt section 4 (all items / targets)
     ('"data/cand_nn_final_forecast_q4.csv"', '"data/cand_rnn_final_forecast_q4.csv"'),
     ('"data/cand_nn_34_output.txt"', '"data/cand_rnn_34_output.txt"')]
for a, b in E:
    n = s.count(a); assert n >= 1, a[:70]
    s = s.replace(a, b)
exec(compile(s, "eda/81_from_68", "exec"), {"__name__": "__main__"})
