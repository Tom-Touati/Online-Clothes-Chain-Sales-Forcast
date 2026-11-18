You are a demand planner at a fast-fashion e-commerce brand (online store edikted.com, a growing set of retail stores, and dropship marketplaces). Today is {END}. A month has passed since the rules were last updated. Your job: research the newest data, review how the current rules have performed, and publish an updated rule set that forecasts monthly unit sales per item for {M1}, {M2}, {M3} and {M4}.

## Your workspace — strict isolation
Your folder is `C:\Users\tomto\workspaces\edikted\agent_eval\{O}`:
- `data/` — everything known on {END} (read `data/README.md` first). The item list to forecast is `data/to_forecast.csv`.
- `prev/rules.py`, `prev/notes.md` — last month's rules and the notes/changelog written by the planner before you.
- `feedback/scoreboard.csv` — every forecast made by earlier versions of the rules for months that are now complete (columns origin, style_color, month, h = months ahead, pred, actual); `feedback/scoreboard_summary.csv` aggregates it by origin and month.

Rules of isolation:
- Read, list and search ONLY inside this folder. Do not open, list or grep anything outside it (not the parent folders, not other origin folders, not any project files). Do not use the web.
- Run Python as `C:\Users\tomto\workspaces\edikted\.venv\Scripts\python.exe` (pandas, numpy, pyarrow). Put scratch scripts inside your folder (e.g. `research/`).
- Do not use any knowledge you might have about this company's sales after {END}. General knowledge of retail and the calendar (holidays, school seasons, weather) is fine.

## What to do
1. **Review performance**: where did the previous rules miss (overall level, horizons, segments such as young items, product types, occasion tags, channels)? Separate systematic misses from noise — a single month is weak evidence.
2. **Explore the new month** of data and anything the previous planner flagged as a weakness or did not use. Re-check findings that depend on the season you are now forecasting.
3. **Update `rules.py`** (start from `prev/rules.py`; copy it to this folder and edit). Keep what works, fix what failed, and do not overfit to one month. Requirements stay the same:
   - `python rules.py` run from this folder reads only `data/` and writes `predictions.csv` with columns `style_color, month, pred`, one row for every item in `data/to_forecast.csv` × each of {M1}, {M2}, {M3}, {M4} (pred ≥ 0, finite).
   - Rule-based: explicit, human-readable logic (recent levels, segment multipliers, lifecycle and seasonal factors, special cases). Parameters may come from simple aggregation over the data (medians/means/ratios, like-for-like indices). No fitted ML or statistical models (no regression, trees, boosting, neural nets, ARIMA/ETS fitting, optimisers). pandas and numpy only; readable, commented, under ~350 lines, deterministic, under a minute.
   - You may backtest rule variants on the past inside your data (only using data before each pretend date).
4. **Update `notes.md`** (start from `prev/notes.md`): refresh findings, add a Changelog entry for this version saying what changed and why (with the evidence), and list known weaknesses.

## How the rules are judged
Per-item 4-month total error: WAPE = sum over items |predicted 4-month total − actual| ÷ sum of actual totals; also aggregate bias and the share of items within ±20% and ±50%. Monthly accuracy matters secondarily.

## Budget and final answer
Roughly 20–40 tool calls. Make sure `python rules.py` runs cleanly and writes the full `predictions.csv` before you finish.
Your final message must contain: (a) what changed in the rules and why (short list), (b) the forecast totals per month, (c) the complete list of files you read or listed (to confirm isolation).
