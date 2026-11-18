"""Model explanation text and the 'Method and decisions' table of candidate nx7 (D389v), spliced into the builder by
eda/181n_make_nx7_builder.py. `_upct` (the buying uplift in percent) is defined in the builder before this block runs.
User (18 Nov): "the explanation should have an explanation for each major decision, both in model choice and feature types, evaluation
methods, methodology. usage of each of the axes of information (cohorts, locations, momentum and seasonality, fashion trend and looks, etc."
and "say the model improved significantly over R3, naive Seasonality and hand rules together" (figures: eda/226_vs_baselines.py).
The template's three boxes are 220 points high and 110 characters wide (about 1,600 characters each; "A page in total is plenty"), so the
boxes hold the short version (rows enlarged to fit) and every decision is justified row by row in the extra sheet.
Every figure is from the time-honest backtest; WAPE / R3 unless said otherwise; sources in log.md (decision ids in the comments)."""
ME = {
 "Your approach": (
   "Each item's July-October total is forecast relative to its own recent rate and then split by month. It is a weighted average of six models that see the problem differently: (1) hand-written planning rules "
   "(recent daily rate x seasonal and event indices by product type x launch ramp and fade); (2) LightGBM on tabular features, the best single model, with a launch-weighted version for items under six months old; "
   "(3) a neural network that learns an item embedding and reads the weekly sales curve; (4, 5) TabPFN v2, a tabular foundation model, one version with calibrated ranges; (6) a model that forecasts each item at each place "
   "it sells and adds them up, at a quarter weight because it leans most on last year. Two small models trained on earlier forecast errors then correct the average (biases by volume, age and discount; signs of breakouts "
   "and of fading items). Two hand rules replace the models where they cannot know the cause: the three World Cup items and the one out-of-stock item (forecast as if in stock, as the brief asks).\n"
   "The models use momentum (7, 28 and 91-day rates), seasonality (like-for-like by product type, the item's own last year, and the items it is bought with), each item's place on its launch curve, where it sells "
   "(stores, channels, regions), fashion attributes and looks (items bought together and looking alike), about 30 US events, and normal temperatures. Every major choice (target, losses, each model, each kind of "
   "information, what was tried and dropped) is justified with its evidence in the 'Method and decisions' sheet; the 'Cascade' sheets show every step per item.\n"
   "Data artefacts and treatment. The 1,500 items are a survivor panel that grows over time, so total units grow sixfold mostly from more items: trend and seasonality are read from like-for-like items, never from raw "
   "totals. 14-15 October 2025 had a mass cancellation (145k units): cancelled lines are excluded, as the target says. Own stores grew from 1 to 16: handled through store shares and the location model. The launch month "
   "is partial: rates are per day on sale. 85 items sold before their published date, which is stored at midnight Israel time: an item starts at its first sale or its New York publish date, whichever is earlier. "
   "Item discounts outside 0-1 are clipped. Possible stock-outs hide about 0.4% of units and were left alone, except for the out-of-stock target."),
 "How you tested it": (
   "A rolling backtest at 13 month-ends (February 2025 to February 2026): everything, including the corrections, is trained only on data before the month-end and scored on the next four months, exactly the task asked. "
   "Error (WAPE of item totals) is always read against simple baselines.\n"
   "Result: the model improved significantly on R3 (the last three months x 4), on a naive seasonal forecast (R3 x last year's seasonal pattern of the product type) and on the hand rules, alone and together. "
   "On the nine most recent month-ends its error is 0.38 of R3's, 0.61 of the seasonal naive's, 0.78 of the hand rules' and 0.59 of the three averaged together; it beats each at every one of those month-ends, and the "
   "95% intervals stay well below 1. On last July-October, the same months as this forecast, it had 0.69 of R3's error and 0.82 of the three together. About four in ten item totals land within +/-20%.\n"
   "In the later rounds a change was kept only if it also held on month-ends not used to tune it and with last year's same season removed from training; well over a hundred ideas failed and were dropped. "
   "Leakage checks passed (features rebuilt from data cut at the month-end are identical; a model trained on shuffled outcomes collapses).\n"
   "Confidence follows the scale's own definitions, checked in the same backtest: 1 only for items with the narrowest range that are also rising and sold in the same months last year (about 60% of such items landed "
   "within +/-20%, though only 37% last July-October); 3 for the four hand-ruled items; 2 for the rest (about 80% landed within +/-50%). 'Cascade (items)' also gives each item a P10-P90 and a P25-P75 range "
   f"(about 80% and 50% of past actuals fell inside) and a buying quantity about {_upct}% above the forecast, because forecasting too low costs more than too high."),
 "Where it is weak": ("The largest misses are young launches that break out (sales growing 3-10x within the window); every model misses them together. About half show within the first month, so a re-forecast after July "
   "catches them. Items bought together rise and fall together, and past sales do not predict that shared shock; the launch and markdown plan, early sell-through and the stock plan would help most. "
   "The forecast ran about 6% below actual sales on the nine recent month-ends (2% over all 13), and 7-12% below for items under six months old. For the 24 targets under six months, the launch-weighted tree raises the "
   "forecast by about 11%: the backtest supports the direction but never showed a lift that large, and for two launches (the Vikkey tank top and the Stephanie lace dress) the last June weeks were already below the level "
   "the forecast assumes for July. These are the first items to re-forecast. The backtest only holds items still selling in 2025-26, so it never saw an item being discontinued. The same month-ends were also used to choose "
   "between models, so the real error should be somewhat worse than 0.38. The World Cup and out-of-stock items rest on judgement rather than a model (confidence 3).")}
# (area, decision, why and evidence). Scores are the error (WAPE) relative to R3 on the nine recent month-ends unless said otherwise.
DEC = [
 ("Framing", "Forecast each item's 4-month total relative to its own recent rate (a log ratio); split by month afterwards",
  "Items range from tens to tens of thousands of units, and the recent rate is the strongest single predictor, so the models only learn how an item moves away from it. A ratio target also lets small and large items train one model."),   # D009-D012
 ("Framing", "Robust losses (median and Huber) instead of mean-seeking ones",
  "A few breakout items dominate a mean. Squared error and Tweedie on units scored worse in the first rounds; Huber on the log ratio is the best tree loss tested."),   # D011, D353
 ("Framing", "Forecast at item level; location level only as one member",
  "Item x location x month forecasts summed up were clearly worse (0.57 against 0.47 for the item-level model then): sparse location counts add noise that does not cancel. Reconciling to category totals did not help either."),   # D126, D022
 ("Models", "Average six different models with simple weights",
  "Their errors differ: alone they score 0.42-0.49, their average 0.39. Weights fitted on recent windows, or per item, cluster and product type from each model's past correlation with actual sales, were level or worse on later months, so the weights stay equal apart from the sixth model."),   # D389e, D389f, D115
 ("Models", "1. Hand-written planning rules",
  "Recent daily rate x seasonal and event indices by product type x launch ramp and fade, written and checked month by month. Transparent, anchored on level, and wrong in different places from the learned models (0.49 alone)."),
 ("Models", "2. LightGBM on tabular features; a launch-weighted version for items under six months",
  "The best single model (0.42). Training rows of new items counted five times improve the forecast of launches; used for every item it hurt established items, so it is routed by age. On the previous five-model system this cut the error on five later month-ends from 0.295 to 0.293 and passed the last-year test."),   # D389h
 ("Models", "3. Neural network with an item embedding and a weekly-curve encoder",
  "Learns item identity hierarchically (product type, then style, then colour) and reads the raw weekly curve, which the tabular models only see through summary features (0.46 alone)."),
 ("Models", "4, 5. TabPFN v2, a tabular foundation model (two versions)",
  "Needs no tuning, takes the fashion attributes and scores directly, and one version supplies the calibrated ranges. Version 2 only: the newer 3.5 is under a non-commercial licence."),   # D169, D204
 ("Models", "6. A location recommender at a quarter weight",
  "Forecasts each item at each of 31 selling places and adds them up. At full weight it gave the best backtest but failed the test with last year's season removed (0.433 against 0.411), so it leans too much on a repeat of 2025; at a quarter weight it passes."),   # D358
 ("Models", "Two small correction models on top",
  "Trained only on earlier forecast errors: bias by volume, age and discount group, and a breakout / fading model (weekly momentum, growth of look-alike items, promotion sensitivity, a recent spike). They move the system from 0.39 to 0.38; at double strength they were worse."),   # D245-D259, D389o
 ("Models", "No tuned settings adopted",
  "All six models were tuned on the 2025 month-ends and judged on five later ones: none improved the combined forecast there. Settings chosen on one year did not transfer, so the defaults stay."),   # D325, D326, D362, D363, D381, D389
 ("Models", "Hand rules for four items",
  "World Cup items: the models extend the June ramp past the 19 July final, which they cannot know. Out-of-stock item: no recent sales to learn from, and the brief says to assume stock. Their raw model totals are in the optional column."),   # D033, D034
 ("Models", "Monthly split from the rules' and the network's monthly shapes",
  "Scored against eight alternatives (either one alone, last year's product-type or item shape, blends, shrinking to flat): the lowest monthly error overall and on both seasons most like this one (0.31 against 0.43 for a flat split)."),   # D389j
 ("Information", "Momentum",
  "Sales rates over the last 7, 28 and 91 days, lags and trend go into every model; the breakout correction adds week-on-week momentum. It is the main driver for young items."),
 ("Information", "Seasonality",
  "Like-for-like seasonal ratios by product type, group and occasion, the item's own pattern last year, and the seasonality of the items bought with it. The history holds each season only once or twice and the models lean on it "
  "(without last year's same season the tree alone is 0.21 worse), so every later change had to hold with that season removed."),   # D315
 ("Information", "Cohorts and life cycle",
  "The 1,500 items are a survivor panel that grows over time, so seasonality comes from items alive in both periods, never from raw totals. Each item is placed on the typical launch curve for its age and product type; corrections are by age band; launches get their own tree."),   # D007, D253
 ("Information", "Locations and channels",
  "The recommender forecasts web regions, TikTok, each own store and each partner separately; the other models see store and channel shares. Store openings are distribution steps, not demand. The 2026 roll-out plan was not in the data: in the backtest, knowing the placement, discount and launch plan ahead cut the error from 0.385 to about 0.371."),   # D308-D310
 ("Information", "Fashion attributes",
  "Tagger attributes and scores derived from them (season, occasion, formality, statement or staple) go into TabPFN, and the neural network embeds occasion, collection and tagger attributes. In LightGBM they added nothing once sales history was in, so they were removed there."),   # D253
 ("Information", "Looks and items bought together",
  "Looks are groups of items bought together and looking alike (baskets, images, text). Growth of look-alike items feeds the breakout correction, and neighbours in the basket network lend their seasonality. Items of a look rise and fall together within a window: "
  "if the total of each product type and look were known the error would fall from 0.40 to 0.26, but past sales do not predict it."),   # D198, D247, line 1136
 ("Information", "Events and weather",
  "A calendar of about 30 US events and sales periods, and normal temperatures weighted by where each item sells. Actual weather added nothing at a four-month horizon; Google Trends did not help."),   # D157, D253
 ("Information", "Prices and promotions",
  "Discount level is a correction group and promotion sensitivity a breakout signal. The planned 2026 campaigns mostly repeat 2025's and are already in the seasonal pattern, so no separate campaign adjustment is applied."),   # D030
 ("Evaluation", "A rolling backtest at 13 month-ends",
  "Each month-end repeats the real task: train on data before it, forecast the next four months, score item totals. Thirteen overlapping windows behave like about three independent tests, so single-window gains are not trusted."),
 ("Evaluation", "Error measured against baselines",
  "Raw error swings with the season; a ratio to a baseline does not. The model is significantly better than R3, a naive seasonal forecast and the hand rules, alone and together: 0.38, 0.61, 0.78 and 0.59 of their error, "
  "better at every recent month-end, with 95% intervals (resampling styles and whole month-ends) no higher than 0.82. On last July-October: 0.69 of R3's error."),   # eda/226
 ("Evaluation", "Rules for accepting a change",
  "It must also hold on month-ends not used to choose it, on the four earliest month-ends, and with last year's same season removed, with intervals from resampling whole styles (colours of a style move together). Well over a hundred ideas failed."),   # D294, D315
 ("Evaluation", "Confidence levels by a tested rule",
  "Scored by the scale's own definitions: the share of stated levels that held rose from 75% to 81% on five later month-ends when level 3 was kept for items with less than even odds of landing within +/-50%. A 1 is not a promise: about 60% of level-1 items landed within +/-20%, 37% last July-October."),   # D389k
 ("Evaluation", "Ranges calibrated on the system's own past errors",
  "Per age and volume group, each side separately, with last July-October and the latest March-June counting four times as much; leaving each month-end out, about 80% of actuals fell inside P10-P90 and 50% inside P25-P75."),   # D389i
 ("Evaluation", "A buying simulation",
  f"Orders placed from the forecast and compared with perfect foresight: forecasting too low costs more than too high, so the buying quantity sits about {_upct}% above the forecast."),   # D067, D260
 ("Evaluation", "What would improve it most",
  "A re-forecast after one month of sales (0.395 to 0.369 in the backtest), the store and partner placement plan, and the launch, markdown and stock plans."),   # D283
]
