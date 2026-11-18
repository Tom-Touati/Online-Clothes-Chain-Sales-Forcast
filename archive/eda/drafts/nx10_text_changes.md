# Text changes for the next workbook (nx10): repairs + improvement roadmap

Drafted by Fork 2 (D413) on the user's request: "add this improvement roadmap to the model explanation, and repair with each of your suggestions".
Forecast numbers do not change. Base: the submitted nx9 text (eda/181q_nx9_text.py). Sources are log entries, cited in brackets for the builder; the cites
are not for the workbook.

## A. Repairs to existing text

### A1. "Your approach", model (1): the rules are written by agents, time-honestly
Replace
> (1) hand-written planning rules (recent daily rate x seasonal and event indices by product type x launch ramp and fade);

with
> (1) planning rules written by AI agents (recent daily rate x seasonal and event indices by product type x launch ramp and fade): at each backtest month-end a fresh agent saw only the data up to that date and the previous month's rules with their track record, and revised them, so the rules are both a time-honest model and the rule-based baseline;

[D103, D104]

### A2. "Method and decisions", row "1. Hand-written planning rules"
Decision: "1. Planning rules written by AI agents"
Why: "Recent daily rate x seasonal and event indices by product type x launch ramp and fade. At each of the 13 backtest month-ends a fresh agent saw only a data snapshot cut at that month-end (checked: no later date in any file) and the previous month's rules with a scoreboard of their past forecasts, and revised them; no fitted models allowed. Transparent, anchored on level, and wrong in different places from the learned models (0.49 alone). The same rules are the rule-based baseline the system is compared with." [D103, D104]

### A3. "How you tested it" and the Evaluation row "Error measured against baselines"
Where the text says "the hand rules", say "the agents' rules" (it is the same baseline: 0.78 of its error). [D389v / D422 baselines]

### A4. Evaluation: name the metrics (new sentence in "How you tested it", and in the Evaluation row "Error measured against baselines")
> Besides WAPE (volume-weighted), each system is scored per item so that small and large items count equally (geometric mean of each item's error relative to R3), on bias, on the share within +/-20%, on pinball loss and coverage for the ranges, and on whether each confidence level's claim holds.

[score_lib; D313 item-equal metric; D389k]

### A5. Evaluation row "A buying simulation": say what it does and what it found
> A daily buying simulation: each day an order is placed from the latest month-end forecast (no look-ahead), arriving 21 days later, with at least 7 days between orders of the same item and safety stock from the forecast's spread; profit is compared with perfect foresight. On earlier versions of the system it lost 11.8% of the ideal profit against 14.2% for a run-rate rule on July-September 2025, and was better in 9 of 11 three-month windows. It showed that forecasting too low costs more than too high, so the buying quantity sits about 8% above the forecast. Seeing a breakout in July does not pay off under a 21-day lead time, so faster replenishment matters as much as the forecast.

[D067, D068, D265]

### A6. New Evaluation row: explainability
Area: Evaluation. Decision: "Explanations checked against the model"
Why: "Each forecast's move away from R3 is explained by SHAP values of the tree model, grouped into 11 categories (momentum, life cycle, events, seasonality, weather, regional climate, items bought with it, last year, style, scale, portfolio drift) by a language model, with three model judges as a second opinion. The explanation is tested, not assumed: a linear model on the 11 category sums recovers 72% of the model's gain over R3 (85% on the 15 largest SHAP values), and the categories explain the final forecast's deviation from R3 with R2 0.94; a linear model on the raw inputs fails, so the model's effects are genuinely non-linear."

[D191, D194, D195, D210]

### A7. "Method and decisions", row "Fashion attributes": how the descriptors were made
Insert at the start:
> The data has no long descriptions, so a language model scored every item on 14 fixed 0-10 fashion fields with anchored rubrics (warmth, coverage, formality, statement, staple ...), and product photos and texts were embedded (OpenCLIP for photos, a sentence model for text) to find look-alikes and looks.

[D-line 709 LLM scoring; D171 content embeddings]

## B. Improvement roadmap

### B1. "Where it is weak": replace the last sentence's lead-in so the box ends with a short roadmap (about 600 characters)
> What would improve it, by measured value: a re-forecast after the first weeks of sales (the month-1 read was worth 0.05 of R3's error in the backtest; half of the breakouts show in July); Edikted's per-item plan for stores, partners, launches and discounts as known inputs (already built: 0.385 to 0.371 in the backtest); demand signals that come before sales (page views, add-to-cart, wishlists, TikTok views of a look) to foresee the shared shocks of looks and launches; pricing tests to measure price response causally; and a third season of history. More compute ranks last: no tuning round transferred to later months.

### B2. "Method and decisions": replace the row "What would improve it most" with an "Improvement roadmap" block (six rows, area "Roadmap")

1. Decision: "A re-forecast after the first weeks of sales"
   Why: "The largest misses are launches that break out, and half of them show within a month: they run 1.47x the forecast in month 1, then about 2x. Knowing month 1 was worth 0.048 of R3's error in the backtest (0.398 to 0.350), and rescaling months 2-4 by the month-1 pace cut their error from 0.326 to 0.303. Cheapest large gain: a weekly re-forecast for items under six months. Under the current 21-day lead time it pays only with faster replenishment or chase buying of breakouts."

2. Decision: "Edikted's per-item plan as known inputs"
   Why: "Training the models with each item's store, partner, launch and discount plan for the window cut the error from 0.385 to 0.371, at all 13 month-ends; for launches that gained 5+ stores the under-forecast fell from 37% to 28%. 88% of the large under-forecast error is on items that gained 2+ stores. The inputs are built; the 2026 plan is the missing piece."

3. Decision: "Foresee the demand for looks and launches"
   Why: "The misses are shared: knowing each product type x look total would cut the error from 0.40 to 0.26, and in 72% of the largest misses all five models are wrong in the same direction. Past sales cannot foresee this (breakouts are spotted at the month-end with AUC 0.66; analog launch curves lost). Signals that arrive before the sales can: product-page views, add-to-cart, wishlists, TikTok and social views of the look, look-level search interest, email and ad response. A hierarchical model would forecast look demand from them and from photo and attribute similarity, then share it out to items; a vision-language model reading the photos for trend attributes (silhouette, colour story, styling) extends this to brand-new launches."

4. Decision: "Pricing tests and a price-aware demand model"
   Why: "A past campaign day sold about 1.55x a normal day, and the response grows sharply with depth, but discounts were not set at random: slow sellers are marked down, strong sellers are not, so history mixes price response with that choice (item discount and elasticity inputs did not help). Randomised or rotated discount tests by product type, age and channel (holdout items, alternating weeks or regions) would measure price response causally, replace the assumed half-strength campaign uplift with a measured one, and turn the forecast into demand at a given price, useful for markdown and campaign planning as well as buying."

5. Decision: "A third season of history"
   Why: "With two seasons the calendar cannot be learned as an input, every change had to pass the test with last year's season removed, and 13 month-ends are about three independent tests. A third season would allow multi-year seasonality and campaign effects, and a re-test of weighting the youngest launches more, which helped on the latest month-ends but failed that test."

6. Decision: "Compute last"
   Why: "Tuning all six models overnight never transferred to later month-ends. Compute is best spent on weekly nowcasting, photo-based trend extraction, averaging several runs of the neural models (two runs of the recommender's neural part differ by a median 13% per item) and, with a commercial licence, the newer TabPFN."

[D248, D309, D244 shape, D247, D129, D014x, D079 / D173, D389l-s, D360]
