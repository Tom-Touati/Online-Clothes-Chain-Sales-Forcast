# The planning rules

One folder per backtest month-end (2025-02 … 2026-02) and for the June 2026 forecast (`2026-06`). In each: `rules.py` (the rules,
written by a fresh AI agent at that month-end with only the data available then, revising the previous month-end's rules: recent daily rate × seasonal and event indices by product type ×
launch ramp and fade), `data/` (the data cut it reads), `predictions.csv` and `predictions_verified.csv` (its monthly forecasts, checked
by `pipeline/76_rules_harness.py`), `notes.md`. `actuals.parquet` holds the actual units used to score them. The agents' working
material (earlier versions, feedback, research harness) is in `archive/agent_eval/`.
