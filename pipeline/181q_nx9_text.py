"""The words of the workbook nx9 (D422; as nx8, D421, with the look-alike correction of D401 described): the three "Model explanation" boxes, the "Method and decisions" sheet and the item
explanations. Read by pipeline/181q_final_nx9_2026.py, which fills the {placeholders} from the numbers it computes, so text and numbers cannot
disagree.

Sources of the statements: the time-honest backtest of the submitted system (log.md; D389u for the audit that corrected the earlier text,
D389v for the first version of this text, D389v / D389x of Fork 2 for the campaign plan, pipeline/226_vs_baselines.py for the baseline
comparison, D400 / D401 for the fashion retest and the look-alike term). Scores are the error (WAPE) relative to R3 on the nine recent month-ends unless said otherwise.

The template's three boxes are 220 points high and 110 characters wide ("A page in total is plenty"), so the boxes hold a short version
(the builder enlarges the rows to fit) and every decision is justified row by row in the "Method and decisions" sheet.
Placeholders: {campaign_pct} = what the campaign uplift adds to the 100 items in total; {buying_uplift} = the median buying quantity over
the forecast, in percent; {launch_items} / {launch_lift} = the targets under six months and what the launch-weighted tree adds to their
six-model average."""

MODEL_EXPLANATION = {
 "Your approach": (
   "Each item's July-October total is forecast relative to its own recent rate and then split by month. It is a weighted average of six models that see the problem differently: (1) hand-written planning rules "
   "(recent daily rate x seasonal and event indices by product type x launch ramp and fade); (2) LightGBM on tabular features, the best single model, with a launch-weighted version for items under six months old; "
   "(3) a neural network that learns an item embedding and reads the weekly sales curve; (4, 5) TabPFN v2, a tabular foundation model, one version with calibrated ranges; (6) a model that forecasts each item at each place "
   "it sells and adds them up, at a quarter weight because it leans most on last year. Two small models trained on earlier forecast errors then correct the average (biases by volume, age and discount; signs of breakouts "
   "and of fading items, including how the item's look-alikes ramped last year).\n"
   "Judgement is added in two places; the 'Model raw total' column is the forecast before both. Campaign plan: four of the six planned site sales ran on the same dates in 2024 and 2025, so the models carry them; "
   "the two new ones (Clearance 12-19 August, 24-29 September) are added at half the measured lift of a past campaign day, by each item's edikted.com share ({campaign_pct} in total). "
   "Hand rules: the three World Cup items and the one out-of-stock item (forecast as if in stock, as the brief asks).\n"
   "The models use momentum, seasonality (like-for-like by product type, the item's own last year, the items it is bought with), each item's place on its launch curve, where it sells, fashion attributes and looks, "
   "about 30 US events, and normal temperatures. Each major choice is justified with its evidence in the 'Method and decisions' sheet; the 'Cascade' sheets show every step per item.\n"
   "Data artefacts and treatment. The 1,500 items are a survivor panel that grows over time, so total units grow sixfold mostly from more items: trend and seasonality are read from like-for-like items, never from raw "
   "totals. 14-15 October 2025 had a mass cancellation (145k units): cancelled lines are excluded, as the target says. Own stores grew from 1 to 16: handled through store shares and the location model. The launch month "
   "is partial: rates are per day on sale. 85 items sold before their published date, which is stored at midnight Israel time: an item starts at its first sale or its New York publish date, whichever is earlier. "
   "Item discounts outside 0-1 are clipped. Possible stock-outs hide about 0.4% of units and were left alone, except for the out-of-stock target."),
 "How you tested it": (
   "A rolling backtest at 13 month-ends (February 2025 to February 2026): everything, including the corrections, is trained only on data before the month-end and scored on the next four months, exactly the task asked. "
   "Error (WAPE of item totals) is always read against simple baselines.\n"
   "Result: the model improved significantly on R3 (the last three months x 4), on a naive seasonal forecast (R3 x last year's seasonal pattern of the product type) and on the hand rules, alone and together. "
   "On the nine most recent month-ends its error is 0.38 of R3's, 0.61 of the seasonal naive's, 0.78 of the hand rules' and 0.59 of the three averaged together; it beats each at every one of those month-ends, and the "
   "95% intervals stay well below 1. On last July-October, the same months as this forecast, it had 0.69 of R3's error and 0.82 of the three together. About four in ten item totals land within +/-20%. "
   "These figures are for the models before the campaign uplift, which was backtested separately.\n"
   "In the later rounds a change was kept only if it also held on month-ends not used to tune it and with last year's same season removed from training; well over a hundred ideas failed and were dropped. "
   "Leakage checks passed (features rebuilt from data cut at the month-end are identical; a model trained on shuffled outcomes collapses).\n"
   "Confidence follows the scale's own definitions, checked in the same backtest: 1 only for items with the narrowest range that are also rising and sold in the same months last year (about 60% of such items landed "
   "within +/-20%, though only 37% last July-October); 3 for the four hand-ruled items; 2 for the rest (about 80% landed within +/-50%). 'Cascade (items)' also gives each item a P10-P90 and a P25-P75 range "
   "(about 80% and 50% of past actuals fell inside) and a buying quantity about {buying_uplift}% above the forecast, because forecasting too low costs more than too high."),
 "Where it is weak": (
   "The largest misses are young launches that break out (sales growing 3-10x within the window); every model misses them together. About half show within the first month, so a re-forecast after July "
   "catches them. Items bought together rise and fall together, and past sales do not predict that shared shock; the launch and markdown plan, early sell-through and the stock plan would help most. "
   "The forecast ran about 6% below actual sales on the nine recent month-ends (2% over all 13), and 7-12% below for items under six months old. For the {launch_items} targets under six months, the launch-weighted tree raises the "
   "six-model average by about {launch_lift}: the backtest supports the direction but never showed a lift that large, and for two launches (the Vikkey tank top and the Stephanie lace dress) the last June weeks were already below the level "
   "the forecast assumes for July. These are the first items to re-forecast. The campaign uplift rests on one assumption, half the lift of a past campaign day, because the plan gives only headline discounts. "
   "The backtest only holds items still selling in 2025-26, so it never saw an item being discontinued. The same month-ends were also used to choose between models, so the real error should be somewhat worse than 0.38. "
   "The World Cup and out-of-stock items rest on judgement rather than a model (confidence 3)."),
}

# (area, decision, why and the evidence)
DECISIONS = [
 ("Framing", "Forecast each item's 4-month total relative to its own recent rate (a log ratio); split by month afterwards",
  "Items range from tens to tens of thousands of units, and the recent rate is the strongest single predictor, so the models only learn how an item moves away from it. A ratio target also lets small and large items train one model."),
 ("Framing", "Robust losses (median and Huber) instead of mean-seeking ones",
  "A few breakout items dominate a mean. Squared error and Tweedie on units scored worse in the first rounds; Huber on the log ratio is the best tree loss tested."),
 ("Framing", "Forecast at item level; location level only as one member",
  "Item x location x month forecasts summed up were clearly worse (0.57 against 0.47 for the item-level model then): sparse location counts add noise that does not cancel. Reconciling to category totals did not help either."),
 ("Models", "Average six different models with simple weights",
  "Their errors differ: alone they score 0.42-0.49, their average 0.39. Weights fitted on recent windows, or per item, cluster and product type from each model's past correlation with actual sales, were level or worse on later months, so the weights stay equal apart from the sixth model."),
 ("Models", "1. Hand-written planning rules",
  "Recent daily rate x seasonal and event indices by product type x launch ramp and fade, written and checked month by month. Transparent, anchored on level, and wrong in different places from the learned models (0.49 alone)."),
 ("Models", "2. LightGBM on tabular features; a launch-weighted version for items under six months",
  "The best single model (0.42). Training rows of new items counted five times improve the forecast of launches; used for every item it hurt established items, so it is routed by age. On the previous five-model system this cut the error on five later month-ends from 0.295 to 0.293 and passed the last-year test."),
 ("Models", "3. Neural network with an item embedding and a weekly-curve encoder",
  "Learns item identity hierarchically (product type, then style, then colour) and reads the raw weekly curve, which the tabular models only see through summary features (0.46 alone)."),
 ("Models", "4, 5. TabPFN v2, a tabular foundation model (two versions)",
  "Needs no tuning, takes the fashion attributes and scores directly, and one version supplies the calibrated ranges. Version 2 only: the newer 3.5 is under a non-commercial licence."),
 ("Models", "6. A location recommender at a quarter weight",
  "Forecasts each item at each of 31 selling places and adds them up. At full weight it gave the best backtest but failed the test with last year's season removed (0.433 against 0.411), so it leans too much on a repeat of 2025; at a quarter weight it passes."),
 ("Models", "Two small correction models on top",
  "Trained only on earlier forecast errors: bias by volume, age and discount group, and a breakout / fading model (weekly momentum, growth of look-alike items, promotion sensitivity, a recent spike) with one linear term added: "
  "how the item's look-alikes by photo and text ramped in the same months last year. Together they move the system from 0.39 to 0.38; at double strength they were worse. "
  "The look-alike term is the only fashion signal that passed every test (0.2926 to 0.2921 on five later month-ends, and it holds with last year's season removed)."),
 ("Models", "No tuned settings adopted",
  "All six models were tuned on the 2025 month-ends and judged on five later ones: none improved the combined forecast there. Settings chosen on one year did not transfer, so the defaults stay."),
 ("Models", "Monthly split from the rules' and the network's monthly shapes",
  "Scored against eight alternatives (either one alone, last year's product-type or item shape, blends, shrinking to flat): the lowest monthly error overall and on both seasons most like this one (0.31 against 0.43 for a flat split)."),
 ("Judgement", "An explicit uplift for the two new site sales (Clearance 12-19 August, 24-29 September)",
  "A past campaign day sold about 1.55x a normal day on edikted.com, like for like, with no pull-forward. Half of that lift is applied (the plan gives headline depth only) to each item's August and September in proportion to its edikted.com share: {campaign_pct} in total. "
  "Backtested with the change in campaign days against the year before: the error fell from 0.3806 to 0.3796 on the nine recent month-ends and from 0.2926 to 0.2916 on the five latest (intervals exclude zero), "
  "but rose on the four earliest, where the extra campaign days were the whole site discounting deeper, not new sales. The four recurring sales are left to the models. "
  "Feeding the campaign calendar to the models as inputs instead was tested in three forms and lost: two seasons give too few distinct months to learn it from."),
 ("Judgement", "Hand rules for four items",
  "World Cup items: the models extend the June ramp past the 19 July final, which they cannot know. Out-of-stock item: no recent sales to learn from, and the brief says to assume stock. The raw model totals are in the optional column."),
 ("Information", "Momentum",
  "Sales rates over the last 7, 28 and 91 days, lags and trend go into every model; the breakout correction adds week-on-week momentum. It is the main driver for young items."),
 ("Information", "Seasonality",
  "Like-for-like seasonal ratios by product type, group and occasion, the item's own pattern last year, and the seasonality of the items bought with it. The history holds each season only once or twice and the models lean on it "
  "(without last year's same season the tree alone is 0.21 worse), so every later change had to hold with that season removed."),
 ("Information", "Cohorts and life cycle",
  "The 1,500 items are a survivor panel that grows over time, so seasonality comes from items alive in both periods, never from raw totals. Each item is placed on the typical launch curve for its age and product type; corrections are by age band; launches get their own tree."),
 ("Information", "Locations and channels",
  "The recommender forecasts web regions, TikTok, each own store and each partner separately; the other models see store and channel shares. Store openings are distribution steps, not demand. The 2026 roll-out plan was not in the data: in the backtest, knowing the placement, discount and launch plan ahead cut the error from 0.385 to about 0.371."),
 ("Information", "Fashion attributes",
  "Tagger attributes and scores derived from them (season, occasion, formality, statement or staple) go into TabPFN, and the neural network embeds occasion, collection and tagger attributes. In LightGBM they added nothing once sales history was in, so they were removed there; "
  "a retest with photo and text embeddings gave new items a small, non-significant gain and lost with last year's season removed. The fashion signal that does help sits in the correction models."),
 ("Information", "Looks and items bought together",
  "Looks are groups of items bought together and looking alike (baskets, images, text). Growth of look-alike items and their ramp in the same months last year feed the breakout correction, and neighbours in the basket network lend their seasonality. Items of a look rise and fall together within a window: "
  "if the total of each product type and look were known the error would fall from 0.40 to 0.26, but past sales do not predict it."),
 ("Information", "Events and weather",
  "A calendar of about 30 US events and sales periods, and normal temperatures weighted by where each item sells. Actual weather added nothing at a four-month horizon; Google Trends did not help."),
 ("Information", "Prices and promotions",
  "Discount level is a correction group and promotion sensitivity a breakout signal. The campaign plan enters through the explicit uplift (under Judgement)."),
 ("Evaluation", "A rolling backtest at 13 month-ends",
  "Each month-end repeats the real task: train on data before it, forecast the next four months, score item totals. Thirteen overlapping windows behave like about three independent tests, so single-window gains are not trusted."),
 ("Evaluation", "Error measured against baselines",
  "Raw error swings with the season; a ratio to a baseline does not. The model is significantly better than R3, a naive seasonal forecast and the hand rules, alone and together: 0.38, 0.61, 0.78 and 0.59 of their error, "
  "better at every recent month-end, with 95% intervals (resampling styles and whole month-ends) no higher than 0.82. On last July-October: 0.69 of R3's error."),
 ("Evaluation", "Rules for accepting a change",
  "It must also hold on month-ends not used to choose it, on the four earliest month-ends, and with last year's same season removed, with intervals from resampling whole styles (colours of a style move together). Well over a hundred ideas failed."),
 ("Evaluation", "Confidence levels by a tested rule",
  "Scored by the scale's own definitions: the share of stated levels that held rose from 75% to 81% on five later month-ends when level 3 was kept for items with less than even odds of landing within +/-50%. A 1 is not a promise: about 60% of level-1 items landed within +/-20%, 37% last July-October."),
 ("Evaluation", "Ranges calibrated on the system's own past errors",
  "Per age and volume group, each side separately, with last July-October and the latest March-June counting four times as much; leaving each month-end out, about 80% of actuals fell inside P10-P90 and 50% inside P25-P75."),
 ("Evaluation", "A buying simulation",
  "Orders placed from the forecast and compared with perfect foresight: forecasting too low costs more than too high, so the buying quantity sits about {buying_uplift}% above the forecast."),
 ("Evaluation", "What would improve it most",
  "A re-forecast after one month of sales (0.395 to 0.369 in the backtest), the store and partner placement plan, and the launch, markdown and stock plans."),
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
