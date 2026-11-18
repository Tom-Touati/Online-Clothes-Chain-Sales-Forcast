"""The words of the workbook nx14 (D431): nx11's text (D429) restructured and extended. The three boxes have one topic per line, each line
opening with a summary up to its first colon, which the builder sets in bold. Added: real-world evaluation as the principle (a Framing row
first), the buying simulation of the submitted system itself (eda/231_sim_nx11.py: 10.6% of the perfect-foresight profit lost, the agents'
rules 12.4%, R3 19.5%; an earlier five-member version, split differently, lost 10.8%), the NOAA temperature data (D155 / D157 / D159 / D159b)
and the ranges as a weakness. Figures as audited (workflow audit-nx11-buying-sim). User requests quoted in D431. Read by
eda/181v_final_nx14_2026.py, which fills the {placeholders} (see eda/181s_nx11_text.py for them and for the earlier sources)."""

MODEL_EXPLANATION = {
 "Your approach": (
   "In short: an ensemble of deep-learning and classical models reads each item's recent history, seasonality, temperature, fashion trends and "
   "look-alike items, campaigns and special events, then corrects itself on its own past errors.\n"
   "Real-world evaluation: the system is judged the way the decision is made. A backtest replays each month-end with only the data known then; "
   "the baselines are R3 and, above all, planning rules written by AI agents that, like a planner, saw only that data and never anything after "
   "the month-end (beyond whatever general knowledge their language-model pretraining holds, which cannot be audited); and the submitted system "
   "was run through a simulation of inventory buying, so it is judged by the money it would make as well as by its error. Each model and "
   "correction was kept only if it improved that backtest.\n"
   "Six models: each forecasts the July-October total relative to the item's recent rate: (1) planning rules written by AI agents, revised at "
   "each month-end with only the data up to it; (2) gradient-boosted trees (LightGBM), with a separate version for items under six months old "
   "that weighs launches more; (3) a neural network with a hierarchical item embedding (product type, style, colour) that reads the weekly sales"
   " curve; (4, 5) TabPFN, a tabular foundation model, one version with calibrated ranges; (6) a recommender-style model that forecasts each "
   "item at each place it sells (web regions, TikTok, each store and partner) and adds them up, at a quarter weight.\n"
   "Corrections and ranges: two small models trained on earlier forecast errors correct the six-model average; ranges are calibrated on the "
   "system's own past errors, last July-October and the latest March-June counting more.\n"
   "Recent history: sales rates over 7, 28 and 91 days, momentum and the weekly curve, with last year's same months as reference.\n"
   "Seasonality: like-for-like by product type, the item's own last year and the seasonality of the items bought with it.\n"
   "Temperature: actual monthly temperatures for every US state from NOAA (Hawaii and DC from Open-Meteo), observed up to each month-end and "
   "weighted by where each item sells, with 2014-2023 normals for the coming months.\n"
   "Fashion trends and relations between items: a co-purchase network, nearest neighbours by photo (OpenCLIP) and text embeddings, looks, and "
   "the item embeddings of the recommender and the network; a language model turned the sparse product text into 14 standard fashion "
   "descriptors.\n"
   "Launches: each item's place on the launch curve for its age and type.\n"
   "Campaigns and events: about 30 US events; the four planned sales that recur are learnt from last year, and the two new ones (Clearance 12-19"
   " August and a sale of up to 80% on 24-29 September) are added at half the measured lift of a past campaign day, by edikted.com share "
   "({campaign_pct}).\n"
   "Hand rules: only for the three World Cup items and the out-of-stock item.\n"
   "Data artefacts: a survivor panel that grows over time (trend read from like-for-like items only); a mass cancellation on 14-15 October 2025 "
   "(cancelled lines excluded); own stores growing from 1 to 16 (store shares, location model); partial launch months (rates per day on sale); "
   "sales before the publish date, stored in Israel time (start = first sale or NY publish date); discounts outside 0-1 clipped; possible "
   "stock-outs (0.4% of units) left alone except the out-of-stock target."),
 "How you tested it": (
   "Real-world evaluation first: the agents' rules stand in for a planner and the buying simulation for the decision itself; error metrics come "
   "second. The 'Metrics vs baselines' sheet sets every metric against R3, the seasonal naive and the agents' rules.\n"
   "Protocol: a leakage-free rolling backtest at 13 month-ends (February 2025 to February 2026): everything, corrections included, is trained "
   "only on data before the month-end and scored on the next four months, the task asked.\n"
   "Leakage checks: features rebuilt from data cut at two test month-ends (June and October 2025) are identical, a model trained on shuffled "
   "outcomes collapses, and one that never saw an item's style is only 1% worse.\n"
   "Baselines and abbreviations: SYS = the submitted system; R3 = the last three months x4; SN = the seasonal naive (R3 x last year's seasonal "
   "pattern of the product type); AR = the agents' rules. The figures below are on the nine recent month-ends, always in the order SYS, R3, SN, "
   "AR; intervals resample whole styles and whole month-ends.\n"
   "Volume-weighted error (WAPE): share of units missed SYS 27%, R3 70%, SN 44%, AR 34%, so SYS has 0.38 of R3's error, 0.61 of SN's and 0.78 of"
   " AR's; SYS on the five latest month-ends 0.29 of R3's, on last July-October 0.68.\n"
   "Per-item error: every item counting once, the median error is SYS 24%, R3 57%, SN 39%, AR 30%.\n"
   "Hit rate and win rate: items within +/-20% SYS 42%, R3 18%, SN 27%, AR 35%; SYS is closer than R3 for 76% of items, than SN for 68% and than"
   " AR for 61%.\n"
   "Bias: SYS -6%, R3 +15%, SN -7%, AR -6%; SYS -1% over all 13 month-ends.\n"
   "Variance explained: R2 on units SYS 0.85, R3 0.15, SN 0.64, AR 0.76; on log units SYS 0.92, R3 0.76, SN 0.84, AR 0.90.\n"
   "Ranges: pinball loss per unit SYS 10.5%, R3 28.1%, SN 16.7%, AR 13.2% (SYS 0.37 of R3's); about 80% of actuals inside P10-P90 and 50% inside"
   " P25-P75.\n"
   "Confidence: 81% of stated levels held on the five latest month-ends.\n"
   "Decision value (buying simulation): a daily simulation orders from the forecast (arrival after 21 days, at least 7 days between orders, "
   "safety stock from a calibrated spread, the same for every system) and compares profit with perfect foresight. The profit lost: SYS 10.6%, R3"
   " 19.5%, SN 14.5%, AR 12.4%; SYS was better than AR in 9 of 11 three-month windows and than R3 in all 11. The buying quantity sits about "
   "{buying_uplift}% above the forecast.\n"
   "Rigour: in the later rounds a change was kept only if it also held on month-ends not used to choose it, on the four earliest and with last "
   "year's season removed; earlier steps (TabPFN, the averaging) and the campaign uplift do worse on the four earliest and were kept on the "
   "recent evidence. Dozens of ideas, many in several variants, were rejected.\n"
   "Explainability: the 'Cascade' sheets show every item, style, product type and the total as a chain: recent rate x4, seasonal baseline, each "
   "model, their weighted average, the total after the corrections, the campaign uplift and the final number (the difference is the hand rules)."
   " For all 100 items: 473k, 594k, models 571-750k, average 651k, after corrections 645k, campaign +9k, World Cup rule -23k, final 630k. SHAP "
   "values of the tree model, grouped by a language model into 11 driver categories, explain each item's move away from its recent rate; a "
   "linear model on those 11 categories recovers 72% of the tree model's gain over R3 (85% on its 15 largest SHAP values)."),
 "Where it is weak": (
   "Breakouts: the largest misses are young launches that break out (sales growing 3-10x within the window), and every model misses them "
   "together. About half show within the first month, so a re-forecast after July catches them.\n"
   "Shared shocks: items bought together rise and fall together, and past sales do not predict that shared shock.\n"
   "Bias: the forecast ran about 6% below actual sales on the nine recent month-ends (2% over all 13), and 7-12% below for items under six "
   "months old.\n"
   "New items: for the {launch_items} targets under six months, the launch-weighted tree raises the six-model average by about {launch_lift}: "
   "the backtest supports the direction but never showed a lift that large, and for two launches (the Vikkey tank top and the Stephanie lace "
   "dress) the last June weeks were already below the level the forecast assumes for July. These are the first items to re-forecast.\n"
   "Campaign uplift: it rests on one assumption, half the lift of a past campaign day, because the plan gives only headline discounts.\n"
   "Discontinued items: the backtest only holds items still selling in 2025-26, so it never saw an item being discontinued.\n"
   "Optimism: the same month-ends were also used to choose between models, so the real error should be somewhat worse than 0.38.\n"
   "Early month-ends: on the four earliest, whose training data had no earlier year behind them (the data start in January 2024), AR did better "
   "than SYS (AR 0.59, SYS 0.68 of R3's error).\n"
   "Judgement: the World Cup and out-of-stock items rest on judgement rather than a model (confidence 3).\n"
   "Ranges: they are the weakest part of the uncertainty. About 80% of past actuals fell inside P10-P90 overall but only 70% last July-October, "
   "the same months as this forecast, and level 1 held for only 37% of items then; the total's P10-P90 is wide, 496k to 772k units.\n"
   "What would improve it, by measured value: a re-forecast after the first weeks of sales (the month-1 read was worth 0.05 of R3's error in the"
   " backtest; half of the breakouts show in July); Edikted's per-item plan for stores, partners, launches and discounts as known inputs "
   "(already built: 0.385 to 0.371 in the backtest); demand signals that come before sales (page views, add-to-cart, wishlists, TikTok views of "
   "a look) to foresee the shared shocks of looks and launches; pricing tests to measure price response causally; and a third season of history."
   " More compute ranks last: no tuning round transferred to later months."),
}

DECISIONS = [
 ("Framing", "Real-world evaluation: a planner's rules as the baseline and a buying simulation as the test",
  "A forecast is only worth what it does for the decision it serves, so the evaluation copies that decision rather than only scoring errors. "
  "The backtest replays each month-end with only the data known then. The main baseline is planning rules written by AI agents: at each "
  "month-end a fresh agent saw only that data and the previous rules with their track record, as a planner would, and never any data after the "
  "month-end (apart from whatever general knowledge its language-model pretraining holds, which cannot be audited), so beating them means "
  "beating a realistic rule-based plan, not just a naive formula. And a daily buying simulation turns each forecast into orders under a 21-day "
  "lead time and scores the profit against perfect foresight: the submitted system lost 10.6% of it, the agents' rules 12.4%, the seasonal "
  "naive 14.5%, R3 19.5%. Metrics that a buyer would not feel (a lower error from a change that does not move orders) were reported but did not"
  " decide on their own. The 'Metrics vs baselines' sheet sets every metric against each baseline."),
 ("Framing", "Forecast each item's 4-month total relative to its own recent rate (a log ratio); split by month afterwards",
  "Items range from tens to tens of thousands of units, and the recent rate is the strongest single predictor, so the models only learn how an "
  "item moves away from it. A ratio target also lets small and large items train one model."),
 ("Framing", "Robust losses (median and Huber) instead of mean-seeking ones",
  "A few breakout items dominate a mean. Squared error and Tweedie on units scored worse in the first rounds; Huber on the log ratio is the "
  "best tree loss tested."),
 ("Framing", "Forecast at item level; location level only as one member",
  "Item x location x month forecasts summed up were clearly worse (0.57 against 0.47 for the item-level model then): sparse location counts add"
  " noise that does not cancel. Reconciling to category totals did not help either."),
 ("Models", "Average six different models with simple weights",
  "Their errors differ: alone they score 0.42-0.49, their average 0.39. Weights fitted on recent windows, or per item, cluster and product type"
  " from each model's past correlation with actual sales, were level or worse on later months, so the weights stay equal apart from the sixth "
  "model."),
 ("Models", "1. Planning rules written by AI agents",
  "Recent daily rate x seasonal and event indices by product type x launch ramp and fade. At each of the 13 backtest month-ends a fresh agent "
  "saw only a data snapshot cut at that month-end (checked: no later date in any file; the agents never saw data after their month-end, and "
  "their only outside knowledge is the language model's own pretraining, which cannot be audited) and the previous month's rules with a "
  "scoreboard of their past forecasts, and revised them; no fitted models allowed. Transparent, anchored on level, and wrong in different "
  "places from the learned models (0.49 alone). The same rules are the rule-based baseline the system is compared with."),
 ("Models", "2. LightGBM on tabular features; a launch-weighted version for items under six months",
  "The best single model (0.42). Training rows of new items counted five times improve the forecast of launches; used for every item it hurt "
  "established items, so it is routed by age. On the previous five-model system this cut the error on five later month-ends from 0.295 to 0.293"
  " and passed the last-year test."),
 ("Models", "3. Neural network with an item embedding and a weekly-curve encoder",
  "Learns item identity hierarchically (product type, then style, then colour) and reads the raw weekly curve, which the tabular models only "
  "see through summary features (0.46 alone)."),
 ("Models", "4, 5. TabPFN v2, a tabular foundation model (two versions)",
  "Needs no tuning, takes the fashion attributes and scores directly, and one version supplies the calibrated ranges. Version 2 only: the newer"
  " 3.5 is under a non-commercial licence."),
 ("Models", "6. A location recommender at a quarter weight",
  "Forecasts each item at each of 31 selling places and adds them up. At full weight it gave the best backtest but failed the test with last "
  "year's season removed (0.433 against 0.411), so it leans too much on a repeat of 2025; at a quarter weight it passes."),
 ("Models", "Two small correction models on top",
  "Trained only on earlier forecast errors: bias by volume, age and discount group, and a breakout / fading model (weekly momentum, growth of "
  "look-alike items, promotion sensitivity, a recent spike) with one linear term added: how the item's look-alikes by photo and text ramped in "
  "the same months last year. Together they move the system from 0.39 to 0.38; at double strength they were worse. The look-alike term is the "
  "only fashion signal that passed every test (0.2926 to 0.2921 on five later month-ends, and it holds with last year's season removed)."),
 ("Models", "No tuned settings adopted",
  "All six models were tuned on the 2025 month-ends and judged on five later ones: none improved the combined forecast there. Settings chosen "
  "on one year did not transfer, so the defaults stay."),
 ("Models", "Monthly split from the rules' and the network's monthly shapes",
  "Scored against eight alternatives (either one alone, last year's product-type or item shape, blends, shrinking to flat): the lowest monthly "
  "error overall and on both seasons most like this one (0.31 against 0.43 for a flat split)."),
 ("Information", "Recent history",
  "Sales rates over the last 7, 28 and 91 days, monthly lags and trend go into every learned model (the agents' rules use the last 14 days, the"
  " last month and the last 91 days), with last year's same months as reference; the neural network also reads the weekly curve itself, and the"
  " breakout correction adds weekly momentum (the last four weeks against the four before, the last week against the last eight). Recent "
  "momentum is the largest group in the tree model's SHAP values (27% of the total contribution)."),
 ("Information", "Seasonality",
  "Like-for-like seasonal ratios by product type, group and occasion, the item's own pattern last year, and the seasonality of the items bought"
  " with it. The history holds each season only once or twice and the models lean on it (without last year's same season the tree alone is 0.21"
  " worse), so every later change had to hold with that season removed."),
 ("Information", "Cohorts and life cycle",
  "The 1,500 items are a survivor panel that grows over time, so seasonality comes from items alive in both periods, never from raw totals. "
  "Each item is placed on the typical launch curve for its age and product type; corrections are by age band; launches get their own tree."),
 ("Information", "Locations and channels",
  "The recommender forecasts web regions, TikTok, each own store and each partner separately; the other models see store and channel shares. "
  "Store openings are distribution steps, not demand. The 2026 roll-out plan was not in the data: in the backtest, knowing the placement, "
  "discount and launch plan ahead cut the error from 0.385 to about 0.371."),
 ("Information", "Fashion attributes",
  "The data has no long descriptions, so a language model scored every item on 14 fixed 0-10 fashion fields with anchored rubrics (warmth, "
  "coverage, formality, statement, staple ...), and product photos and texts were embedded (OpenCLIP for photos, a sentence model for text) to "
  "find look-alikes and looks. Tagger attributes and these scores go into TabPFN, and the neural network embeds occasion, collection and tagger"
  " attributes. In LightGBM they added nothing once sales history was in, so they were removed there; a retest with photo and text embeddings "
  "gave new items a small, non-significant gain and lost with last year's season removed. The fashion signal that does help sits in the "
  "correction models."),
 ("Information", "Fashion trends and relations between items",
  "Several parts read how items relate: the co-purchase network (basket neighbours lend their seasonality), nearest neighbours by photo and "
  "text embeddings (their ramp in the same months last year feeds the correction), looks (items bought together and looking alike), and the "
  "item embeddings of the recommender and the neural network. Items of a look rise and fall together within a window: if each product type and "
  "look total were known the error would fall from 0.40 to 0.26 of R3's, but past sales do not predict it."),
 ("Information", "Events and weather",
  "A calendar of about 30 US events and sales periods, and temperature where each item sells: actual monthly average and maximum temperatures "
  "for every US state from NOAA's Climate at a Glance (Hawaii and DC from Open-Meteo city data), 2014 to June 2026, weighted by each item's "
  "online sales by state over the past 12 months: the temperature observed over the three months up to the month-end, its anomaly, the "
  "2014-2023 normal of the target months and the item's temperature preference. On an earlier tree model, temperature alone cut the error from "
  "0.455 to 0.439 of R3's (0.433 to 0.426 within the three-model average, interval clear of zero) and, with the events, replaced the seasonal "
  "indices. Temperature and the events calendar carry largely the same seasonal timing: either alone gives most of the gain, and together they "
  "add little more (0.437 with events only, 0.436 with both). Knowing the realised temperature of the target months added nothing at a "
  "four-month horizon; Google Trends did not help."),
 ("Information", "Prices and promotions",
  "Discount level is a correction group and promotion sensitivity a breakout signal. The campaign plan enters through the explicit uplift "
  "(under Judgement)."),
 ("Judgement", "An explicit uplift for the two new site sales (Clearance 12-19 August and a sale of up to 80% on 24-29 September)",
  "A past campaign day sold about 1.55x a normal day on edikted.com, like for like, with no pull-forward. Half of that lift is applied (the "
  "plan gives headline depth only) to each item's August and September in proportion to its edikted.com share: {campaign_pct} in total. "
  "Backtested with the change in campaign days against the year before: the error fell from 0.3806 to 0.3796 on the nine recent month-ends and "
  "from 0.2926 to 0.2916 on the five latest (intervals exclude zero), but rose on the four earliest, where the extra campaign days were the "
  "whole site discounting deeper, not new sales. The four recurring sales are left to the models. Feeding the campaign calendar to the models "
  "as inputs instead was tested in three forms and lost: two seasons give too few distinct months to learn it from."),
 ("Judgement", "Hand rules for four items",
  "World Cup items: the models extend the June ramp past the 19 July final, which they cannot know. Out-of-stock item: no recent sales to learn"
  " from, and the brief says to assume stock. The raw model totals are in the optional column."),
 ("Metrics", "Volume-weighted error (WAPE), read against baselines",
  "Captures the errors on the large items, where most units are. Share of units missed on the nine recent month-ends: SYS 26.5%, R3 69.9%, SN "
  "43.6%, AR 34.1%, so SYS has 0.38 of R3's error, 0.61 of SN's and 0.78 of AR's. SYS as a share of R3's error: all 13 month-ends 0.44, the "
  "five latest 0.29, last July-October 0.68, the four earliest 0.68 (where AR did better, 0.59). Against R3, SN and AR averaged: 0.59; better "
  "than each at every recent month-end, with 95% intervals no higher than 0.82. A ratio to a baseline does not swing with the season as raw "
  "error does."),
 ("Metrics", "Per-item error, every item counting once",
  "Captures the many small items that WAPE underweights. Median absolute error per item: SYS 24.2%, R3 57.2%, SN 38.8%, AR 30.3%. SYS's error "
  "relative to R3's per item: median 0.46, mean 0.41."),
 ("Metrics", "Hit rate within +/-20% and win rate",
  "Captures how often a single item lands close, the threshold of the confidence scale. Items within +/-20%: SYS 42.3%, R3 18.1%, SN 27.1%, AR "
  "34.6%; by units SYS 49.1%, R3 19.2%, SN 29.9%, AR 37.9%; the 100 targets SYS 48%. SYS is closer than R3 for 76% of item forecasts, than SN "
  "for 68% and than AR for 61%."),
 ("Metrics", "Bias",
  "Captures the direction of the miss, which drives buying cost. Forecast against actual units: SYS -5.6%, R3 +15.4%, SN -6.7%, AR -6.4%; SYS "
  "-1% over all 13 month-ends, and new items run 7-12% under. Error and bias together ((MAE + |bias|) per unit): SYS 33.6%, R3 113.7%, SN "
  "50.4%, AR 45.0% (SYS 0.30 of R3's)."),
 ("Metrics", "Variance explained (R2)",
  "Captures whether the forecast ranks and scales items correctly across the catalogue. R2 on units: SYS 0.85, R3 0.15, SN 0.64, AR 0.76; on "
  "log units: SYS 0.92, R3 0.76, SN 0.84, AR 0.90. The item explorer also shows each item's correlation and R2 across month-ends."),
 ("Metrics", "Ranges: pinball loss and coverage",
  "Captures whether the uncertainty is honest. Pinball loss over P10 / P25 / P50 / P75 / P90 per unit (each forecast given the quantile model's"
  " spread, scored the same way for all four): SYS 10.5%, R3 28.1%, SN 16.7%, AR 13.2% (SYS 0.37 of R3's). Leaving each month-end out, about "
  "80% of actuals fell inside SYS's P10-P90 and 50% inside P25-P75 (70% inside P10-P90 last July-October)."),
 ("Metrics", "Confidence levels by the scale's own definitions",
  "Captures whether each stated level keeps its word (1 within +/-20%, 2 within +/-50%, 3 could miss by more): 81% of stated levels held on the"
  " five latest month-ends; about 60% of level-1 items landed within +/-20% and 84% of level-2 items within +/-50%."),
 ("Metrics", "Decision value: the buying simulation",
  "Captures what the forecast is worth in the decision it serves. Daily orders from the forecast, 21-day arrival, at least 7 days between "
  "orders, safety stock from a calibrated spread, the same for every forecast; profit compared with perfect foresight. Profit lost, simulated "
  "on the submitted system itself, over 11 three-month windows (March 2025 to January 2026): SYS 10.6%, R3 19.5%, SN 14.5%, AR 12.4%; SYS "
  "better than AR in 9 of 11 windows and than R3 in all 11. Demand filled: SYS 94%, R3 86%, AR 92%, with slightly more leftover stock for SYS "
  "than for AR (252k against 243k units). Without the campaign uplift SYS also loses 10.6%, so the uplift leaves buying level (its backtest "
  "uses the realised campaign calendar as the plan, an upper bound). An earlier five-member version, simulated with each member split by its "
  "own monthly profile, lost 10.8%. Each order uses only the latest month-end forecast. Forecasting too low costs more than too high, so the "
  "buying quantity sits about {buying_uplift}% above the forecast; under a 21-day lead time, chasing a July breakout does not pay."),
 ("Metrics", "How sure the scores are",
  "Intervals resample whole styles (colours of a style move together) and whole month-ends. SYS against AR: -0.12 to -0.10 of R3's error by "
  "style, -0.15 to -0.08 by month-end. Thirteen overlapping month-ends behave like about three independent tests, so gains under about 0.005 "
  "are not trusted."),
 ("Evaluation", "A rolling backtest at 13 month-ends",
  "Each month-end repeats the real task: train on data before it, forecast the next four months, score item totals. Thirteen overlapping "
  "windows behave like about three independent tests, so single-window gains are not trusted."),
 ("Evaluation", "Rules for accepting a change",
  "It must also hold on month-ends not used to choose it, on the four earliest month-ends, and with last year's same season removed, with "
  "intervals from resampling whole styles (colours of a style move together). Dozens of ideas, many in several variants, were rejected."),
 ("Evaluation", "Confidence levels by a tested rule",
  "Scored by the scale's own definitions: the share of stated levels that held rose from 75% to 81% on five later month-ends when level 3 was "
  "kept for items with less than even odds of landing within +/-50%. A 1 is not a promise: about 60% of level-1 items landed within +/-20%, 37%"
  " last July-October."),
 ("Evaluation", "Ranges calibrated on the system's own past errors",
  "Per age and volume group, each side separately, with last July-October and the latest March-June counting four times as much; leaving each "
  "month-end out, about 80% of actuals fell inside P10-P90 and 50% inside P25-P75."),
 ("Explainability", "A cascade for every item and group",
  "'Cascade (items)' shows each item's forecast as a chain: the recent rate x4 (R3), the product type's seasonal baseline, each of the six "
  "models, their weighted average, the total after the corrections (the raw model total), the edikted.com share and campaign uplift, and the "
  "final number, with the buying quantity, the spread between the highest and lowest model, confidence and P10-P90. 'Cascade (hierarchy)' adds "
  "the same chain up by style, product type and in total (all 100 items: R3 473k, seasonal 594k, the six models 571-750k, average 651k, after "
  "corrections 645k, campaign +9k, the World Cup rule -23k, final 630k), and 'Ranges (totals)' gives ranges for the total and for the product "
  "types and styles with three or more items, allowing for misses that move together. Each step's contribution to an item, a type or the total "
  "can be read off and challenged: which model pulled it up, how much the corrections moved it, what judgement added."),
 ("Explainability", "Explanations checked against the model",
  "Each forecast's move away from R3 is explained by SHAP values of the tree model, grouped into 11 categories (momentum, life cycle, events, "
  "seasonality, weather, regional climate, items bought with it, last year, style, scale, portfolio drift) by a language model, with three "
  "model judges as a second opinion. The explanation is tested, not assumed: a linear model on the 11 category sums recovers 72% of the model's"
  " gain over R3 (85% on the 15 largest SHAP values), and the categories explain the final forecast's deviation from R3 with R2 0.94; a linear "
  "model on the raw inputs fails, so the model's effects are genuinely non-linear."),
 ("Roadmap", "A re-forecast after the first weeks of sales",
  "The largest misses are launches that break out, and half of them show within a month: they run 1.47x the forecast in month 1, then about 2x."
  " Knowing month 1 was worth 0.048 of R3's error in the backtest (0.398 to 0.350), and rescaling months 2-4 by the month-1 pace cut their "
  "error from 0.326 to 0.303. Cheapest large gain: a weekly re-forecast for items under six months. Under the current 21-day lead time it pays "
  "only with faster replenishment or chase buying of breakouts."),
 ("Roadmap", "Edikted's per-item plan as known inputs",
  "Training the models with each item's store, partner, launch and discount plan for the window cut the error from 0.385 to 0.371, at all 13 "
  "month-ends; for launches that gained 5+ stores the under-forecast fell from 37% to 28%. 88% of the large under-forecast error is on items "
  "that gained 2+ stores. The inputs are built; the 2026 plan is the missing piece."),
 ("Roadmap", "Foresee the demand for looks and launches",
  "The misses are shared: knowing each product type x look total would cut the error from 0.40 to 0.26, and in 72% of the largest misses all "
  "five models are wrong in the same direction. Past sales cannot foresee this (breakouts are spotted at the month-end with AUC 0.66; analog "
  "launch curves lost). Signals that arrive before the sales can: product-page views, add-to-cart, wishlists, TikTok and social views of the "
  "look, look-level search interest, email and ad response. A hierarchical model would forecast look demand from them and from photo and "
  "attribute similarity, then share it out to items; a vision-language model reading the photos for trend attributes (silhouette, colour story,"
  " styling) extends this to brand-new launches."),
 ("Roadmap", "Pricing tests and a price-aware demand model",
  "A past campaign day sold about 1.55x a normal day, and the response grows sharply with depth, but discounts were not set at random: slow "
  "sellers are marked down, strong sellers are not, so history mixes price response with that choice (item discount and elasticity inputs did "
  "not help). Randomised or rotated discount tests by product type, age and channel (holdout items, alternating weeks or regions) would measure"
  " price response causally, replace the assumed half-strength campaign uplift with a measured one, and turn the forecast into demand at a "
  "given price, useful for markdown and campaign planning as well as buying."),
 ("Roadmap", "A third season of history",
  "With two seasons the calendar cannot be learned as an input, every change had to pass the test with last year's season removed, and 13 "
  "month-ends are about three independent tests. A third season would allow multi-year seasonality and campaign effects, and a re-test of "
  "weighting the youngest launches more, which helped on the latest month-ends but failed that test."),
 ("Roadmap", "Compute last",
  "Tuning all six models overnight never transferred to later month-ends. Compute is best spent on weekly nowcasting, photo-based trend "
  "extraction, averaging several runs of the neural models (two runs of the recommender's neural part differ by a median 13% per item) and, "
  "with a commercial licence, the newer TabPFN."),
]

# The "Metrics vs baselines" sheet (user: "add a table that shows the improvement of each metric over R3, seasonal naive, and agent rule based.
# emphasize the agents never saw any data into the future (except for their unknow pretraining)"). Values from data/metrics_table_nx12.csv
# (eda/232_metrics_table.py: scored exactly as data/scoreboard.json; buying loss from eda/231_sim_nx11.py and eda/231b_sim_naive.py).
METRICS_NOTE = ("Every metric of the submitted forecast next to the three baselines, on the nine most recent backtest month-ends (9,031 item forecasts), all scored the same way. "
                "The agents' rules are the toughest baseline and a time-honest one: at each month-end a fresh AI agent wrote and revised planning rules from a data snapshot cut at "
                "that month-end, so it never saw any later data (its only outside knowledge is its language model's pretraining, which cannot be audited). "
                "'Better than' is the share of the baseline's error removed for errors and losses, and the points gained for shares and R2. Abbreviations as in the other sheets: SYS = the submitted system, R3 = the last three months x 4, SN = the seasonal naive, AR = the agents' rules.")
# (key in the csv, label, what it captures, kind): kind loss = lower is better, shown as a share removed; share / r2 = higher is better, points
# gained; bias = closer to zero, points removed; win = only the comparison columns (share of item forecasts closer than the baseline)
METRICS_ROWS = [
 ("WAPE (error / units)", "Volume-weighted error (WAPE)", "Errors on the large items, where most units are", "loss"),
 ("Median error per item", "Per-item error (median)", "Every item counting once", "loss"),
 ("Items within +/-20%", "Hit rate: items within +/-20%", "How often a single item lands close", "share"),
 ("Units within +/-20%", "Hit rate: units within +/-20%", "The same, weighted by volume", "share"),
 ("Bias (forecast / actual - 1)", "Bias", "The direction of the miss, which drives buying cost", "bias"),
 ("(MAE + |bias|) / actual units", "Error plus bias", "Size and direction of the miss together, per month-end", "loss"),
 ("R2 on units", "Variance explained (R2, units)", "Ranking and scaling items across the catalogue", "r2"),
 ("R2 on log units", "Variance explained (R2, log units)", "The same, with small items counting more", "r2"),
 ("Pinball loss, P10-P90 (/ units)", "Pinball loss, P10 to P90", "Quality of the ranges (each forecast given the quantile model's spread)", "loss"),
 ("Buying loss (% of perfect-foresight profit)", "Buying loss (decision value)", "Profit lost against perfect foresight in the daily buying simulation", "loss"),
 ("System closer than the baseline (items)", "Win rate", "Share of item forecasts closer to the actual than the baseline's", "win"),
]

# One or two sentences per item: what drove the number, what the models said, whether it was adjusted. The builder fills, per item:
#   {apr} {may} {jun}  units in April / May / June 2026        {spring}  mean monthly units April-June 2026     {spring_ly}  the same in 2025
#   {last_year}  units in July-October 2025                    {lo} {hi}  lowest and highest of the six models    {raw}  the models' total
#   {jul} {oct}  the forecast for July / October               {lift}  what the launch-weighted tree adds to this item's forecast
#   {adjusted}  "Adjusted +x% for the two new site sales (August, September), to <final total>" or "Not adjusted"
EXPLANATIONS = {
 "S23794_BLACK": "Launched 30 April and already the largest item: {jun} units in June, with weekly sales peaking in mid-June. The six models range {lo} to {hi} and give {raw} (July above June, then fading), including a {lift} lift from the launch-weighted tree. {adjusted}; the last June week was below the peak, so it is the first item to re-forecast.",
 "S15132_WHITE": "Three years on sale and still rising: {apr} units in April, {jun} in June. Last year it sold {last_year} in July-October from a slightly lower spring rate; the models agree closely ({lo} to {hi}) and give {raw}, with the August back-to-school peak for jeans. {adjusted}; confidence 1.",
 "S17159_BLACK-WASHED": "Steady at about {spring} a month all spring. Last year it sold {last_year} in July-October from the same rate; the models ({lo} to {hi}) expect a smaller autumn step for an item now two years old and give {raw}, with October, its Halloween month, the highest. {adjusted}; if last year's step repeats, it will come in higher.",
 "S23965_BLACK": "Launched 12 April; {jun} units in June at a steady 1,000-1,300 a week. The models range {lo} to {hi} and give {raw} ({jul} in July, down to {oct} by October), including a {lift} lift from the launch-weighted tree. {adjusted}.",
 "S11233_BLACK": "Five years on sale and declining year on year: about {spring} a month this spring against {spring_ly} a year ago. Tagged for Halloween, it peaks in August and again in October, so the models ({lo} to {hi}) give {raw} ({oct} in October), well below last year's {last_year}. {adjusted}.",
 "S20850_BLACK-AND-WHITE": "A hoodie launched last summer; spring sales fell from {apr} in April to {jun} in June, as hoodies do. It sold {last_year} in July-October last year; the models give {raw}, rising to {oct} in October. {adjusted}; their wide range ({lo} to {hi}) is the doubt about a second autumn as strong as its first.",
 "S14091_BLUE": "Three years on sale; about {spring} a month this spring against {spring_ly} a year ago, and June fell to {jun}. The models agree ({lo} to {hi}) on {raw}, with the August peak for jeans, against {last_year} last year. {adjusted}.",
 "S23605_PINK": "On sale since February and in volume since April, at about {spring} a month. The models range {lo} to {hi} and give {raw}: a little above the spring rate in July and August, then {oct} in October, including a {lift} lift from the launch-weighted tree. {adjusted}.",
 "S18184_LIGHT-BLUE": "A bikini top tagged for the 4th of July, on sale since February; June was its best month ({jun} units). The holiday passes early in July and swimwear falls after it, so the models ({lo} to {hi}) give {raw}: {jul} in July, {oct} by October. {adjusted}; if the June rise was mostly holiday demand, July will come in lower.",
 "S15999_BLUE": "A summer short two years on sale, steady at about {spring} a month, half its rate of a year ago ({spring_ly}). The models agree ({lo} to {hi}) on {raw}: July and August a little above the current rate, then the usual drop after summer. {adjusted}; confidence 1.",
 "S21729_GRAY-AND-WHITE": "A sweater launched last autumn; spring sales fell to {jun} in June, as expected. Sweaters climb from August, so the models' forecast rises to {oct} in October, {raw} in total. {adjusted}; the models range widely ({lo} to {hi}) on how strong its second autumn will be.",
 "S21911_LIGHT-GRAY": "Small in spring (about {spring} a month), but June more than doubled to {jun}. It sells with the sweaters, so the models ({lo} to {hi}) give {raw}, more than three times the recent rate, on the autumn season and its June momentum. {adjusted}.",
 "S20878_WHITE": "A graduation dress: sales peaked in May ({may} units) and fell to {jun} in June once the season ended, with the last June week under 200. The models ({lo} to {hi}) give {raw}, about half the spring rate. {adjusted}; with graduation over, the late-June pace points lower: the clearest downside risk among the larger items.",
 "S21143_BLACK": "About a year on sale; June dipped to {jun} after {may} in May. It sold {last_year} in July-October last year, in its launch season; the models split ({lo} to {hi}) and their average gives {raw}. {adjusted}; one of the widest model ranges among the larger items.",
}
ADJUSTED = "Adjusted {pct} for the two new site sales (August, September), to {total}"
NOT_ADJUSTED = "Not adjusted"

# the hand-ruled items: {model} = the models' total, {p10} {p90} = the models' own range, {final} = the rule's total
WORLD_CUP_FACTS = {
 "S24840_LIGHT-GRAY-MELANGE": "World Cup and 4th of July tee launched 25 May; 6,685 units in June, about 306 a day in late June",
 "S25039_YELLOW": "World Cup tee launched 25 May; 2,584 units in June, about 120 a day at month end",
 "S24847_NAVY": "World Cup shorts launched 25 May, flat at about 30 a day in June (928 units)",
}
WORLD_CUP = ("{fact}. The models answer {model} (P10-P90 {p10}-{p90}), not knowing the tournament ends on 19 July. I set July to the late-June pace until the final and capped Aug-Oct at 75/50/45% of July, "
             "the post-event decay of past holiday items. Final {final}. Confidence 3: one season, event-driven demand.")
OUT_OF_STOCK = ("Out of stock: no sales since 10 April. Forecast as if back in stock, as the brief asks: its January-March rate (about 32 a month) x the seasonal pattern of tops at least a year old. "
                "No model covers it, so the number is a rule; confidence 3.")

# D433: the lead line of the coloured ratio table in the "Model explanation" sheet (Fork 2's wording, D419c)
RATIO_LEAD = ("Submitted system against the three baselines on the nine most recent backtest month-ends; ratio = SYS / baseline (for bias, of the absolute bias); "
              "green = SYS better, deeper green = more times better; grey = within 5%. SYS = the submitted system, R3 = the last three months x 4, "
              "SN = the seasonal naive, AR = the agents' rules.")
