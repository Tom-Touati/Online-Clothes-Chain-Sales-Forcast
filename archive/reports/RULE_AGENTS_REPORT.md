# Rule-writing agents against our models (15 Nov 2026)

Decisions D103–D104 in `log.md`. Code: `eda/75_agent_snapshots.py` (snapshots), `eda/76_rules_harness.py` (run, validate, feedback), `eda/77_rules_eval.py` (scoring, output in `data/rules_eval_output.txt`). Every agent's rules, notes and research scripts are in `agent_eval/<origin>/`.

## Setup
- **13 monthly origins,** from the end of Feb 2025 to the end of Feb 2026. These are the same origins, items and four target months as our models' backtest: 626 to 1,262 items per origin.
- **One fresh agent per origin.** Each saw only a data snapshot cut at its origin's month end: daily and monthly item sales by channel, discount, site, channel and store series, the item master and the tagger attributes. No snapshot contains a date after its cutoff, and actual outcomes were never in an agent's folder.
- **What each agent did:**
  - explored the data;
  - wrote `rules.py`: explicit pandas/numpy logic built from levels, ratios, segment indices and hand-set multipliers, with no fitted ML or statistical models;
  - wrote notes with a changelog.
- **Monthly updates.** The first agent built rules from scratch. Each later agent received last month's rules and notes, plus a scoreboard of every earlier forecast against the actuals known at its own cutoff, and revised them. The result is 13 versions, v1 to v13.
- **Our system:** the production quantile model's median 4-month total, the monthly LightGBM ensemble, the embedding NN (6 seeds) and the LightGBM + NN average. All were retrained at each origin on the same history.
- **Score:** WAPE of each item's 4-month total. Rules minus system, item bootstrap within origins.

## Results

| 4-month item totals | Rules | Quantile model | LightGBM | NN | LightGBM + NN | Rules × quantile (geometric mean) |
|---|---|---|---|---|---|---|
| **All 13 origins:** WAPE | **0.340** | 0.382 | 0.360 | 0.368 | 0.349 | **0.328** |
| All 13: bias | −6.3% | −0.2% | −6.7% | +1.9% | −2.4% | −5.2% |
| All 13: within ±20% / ±50% | 34.5% / 73.3% | 33.0% / 68.0% | 33.2% / 70.7% | 33.6% / 69.4% | 35.0% / 71.4% | 35.2% / 73.4% |
| **Nine recent origins (Jun 2025 – Feb 2026):** WAPE | 0.341 | **0.329** | 0.349 | 0.338 | **0.328** | **0.313** |
| Recent: bias | −6.4% | −9.9% | −11.4% | −3.4% | −7.4% | −9.4% |

**Paired differences in WAPE.** Negative means the first system is better; the brackets are 95% item-bootstrap intervals.

| Comparison | All 13 origins | Nine recent origins |
|---|---|---|
| Rules − quantile model | −0.042 [−0.056, −0.029] | +0.012 [+0.002, +0.022] |
| Rules − LightGBM + NN | −0.008 [−0.019, +0.003] | +0.013 [+0.004, +0.023] |
| Rules × quantile − quantile model | −0.054 [−0.063, −0.047] | −0.015 [−0.021, −0.010] |
| Rules × quantile − LightGBM + NN | −0.021 [−0.027, −0.015] | −0.014 [−0.020, −0.009] |
| Rules × quantile − rules | −0.013 [−0.019, −0.006] | −0.028 [−0.034, −0.022] |

**What the numbers say**
1. **Over all 13 origins, the rules beat every one of our single models.** They are level with the LightGBM + NN average; the difference is not significant.
   - The lead comes from the first two origins (Feb and Mar 2025). There, the rules score 0.37 against 0.86 and 0.69 for the quantile model, which still had little history with known 4-month outcomes to train on.
   - An agent with the same thin history did better by reasoning explicitly: about last February's promotion spike, the July 2024 move to deep discounts, and store openings.
2. **On the nine recent origins, which our production model was chosen on, our system is better:**
   - The quantile model and the LightGBM + NN average beat the rules by about 0.013 WAPE, which is significant.
   - The NN and LightGBM alone are level with the rules.
   - The rules win at 2 of 9 recent origins against the quantile model, and at 3 of 9 against the average.
3. **The rules and the quantile model err differently.** Their 50/50 geometric mean, with no tuning, is significantly better than either alone in both windows.
   - Recent origins: 0.313, against 0.329 for the quantile model and 0.341 for the rules.
   - It also beats the quantile model at 10 of 13 origins and the LightGBM + NN average at 11 of 13.
4. **Monthly accuracy favours the rules.** Item-month WAPE is 0.385, against 0.415 for LightGBM and 0.414 for the NN. Their one-month-ahead site total is the most accurate (mean |log error| 0.127, against 0.185 and 0.140).
5. **Segments.**
   - The rules are far better on warm items (hoodies, sweaters, jackets, sweatpants): WAPE 0.313, against 0.435 for the LightGBM + NN average and 0.510 for the quantile model. They are also better on swimwear and accessories.
   - They are worse on occasion-tagged items (0.372 against 0.317) and on bottoms.
   - On the 100 targets the rules score 0.351, against 0.349 for the average and 0.391 for the quantile model.
6. **The June 2025 origin, the deliverable's exact situation, was the rules' worst:** WAPE 0.401 and bias −32%, against 0.334 for the quantile model.
   - v2–v5 had removed last July's back-to-school sale from the 2024 data, treating it as a one-off discount change. It recurred on the same dates in 2025.
   - The July agent found and fixed this.

## How the rules evolved (v1 → v13)
The final rule is: an item's recent daily rate × a site and channel plan × a damped like-for-like seasonal index by product type × young-item ramp and fade × occasion-tag and occasion-tier indices.

- **Site level.** The level was the agents' main miss at every stage, and the plan changed with it:
  - from last year's ratios (v1–v3);
  - to the item's drift relative to the whole site (v3);
  - to an explicit site plan in units per day with Black Friday and December (v6 onwards).
  - The scoreboard showed the item logic alone within about ±3% when the site level was right.
- **Seasonal amplitude:** damped and then measured from the current year. The winter/summer swing in 2025 was about half of 2024's.
- **Retail:** from the flagship store to all established stores; pop-up stores handled; a new-store uplift at full run-rate.
- **Occasion and tier indices:** re-measured each month from the same base month a year earlier.
- **Tested and rejected repeatedly:** regression toward the average item size, and uplifts for dropship-heavy items.

## Caveats
- **Isolation was enforced by instruction, not technically.** All 13 agents reported reading only their own folder, and their code reads only `data/`. The agents did see the user's memory note, which holds a logging rule and no data.
- **Some forward-looking knowledge is shared by both sides:** the item panel exists only because items survived into 2026, and item master fields are as of the dataset.
  - The agents also used general calendar knowledge (Black Friday dates, school seasons), as planned.
- **Agent-written rules are not fully reproducible:** a re-run would write different rules. These results describe one run of 13 agents.
- **What the comparison covers:** the production pipeline's hand rules (D051–D063) could not be scored per item, so "our system" here is the models before those rules.
- **Cost:** about 2.0 million agent tokens over about 2.2 hours.

## Recommendation
- **Model choice:** this does not argue for replacing the models with rules. On the recent origins the models are better.
- **Combination:** the rules are a strong, complementary second opinion. The untuned 50/50 combination of the rules and the quantile model is the best system measured in this project on 4-month item totals: recent-origin WAPE 0.313, against 0.329 for the quantile model alone. The NN re-centring of D095 reached 0.313–0.315 on its own representative-origin set; the two were not compared on identical items.
- **Next step (not done):** if wanted, test rules + quantile + NN together on identical item sets, and apply it to the June 2026 origin. That needs one more agent run on the June 2026 snapshot.

## Follow-up: June 2026 origin and a combined candidate (D105)
- **Agent v14.** One more agent ran at the end of June 2026 on 1,424 items: the eligible set plus the out-of-stock target. It used v13 and its scoreboard for Mar–Jun 2026.
  - v13 had scored 0.335 there with bias −3.5%. That bias was two errors cancelling: a web plan that was too high and item logic that was too low.
  - v14 keeps last year's back-to-school jump in the site plan, re-reads the Jul–Oct occasion indices and adds a launch-season index.
  - For the 100 targets it forecasts 620,269 units.
- **Adding the NN helps.** On identical items over the nine recent origins:

  | Centre (geometric mean) | Recent WAPE | All 13 |
  |---|---|---|
  | Quantile model alone | 0.329 | 0.382 |
  | Quantile × NN (NN candidate) | 0.319 | 0.359 |
  | Rules × quantile | 0.313 | 0.328 |
  | **Rules × quantile × NN, equal thirds** | **0.307** | **0.326** |

  - Three-way minus quantile × NN: −0.012 [−0.015, −0.009]. On the 100 targets alone the gain is small: 0.318 against 0.320.
- **Candidate workbook:** `Edikted_Forecasting_Exercise_candidate_rules.xlsx`.
  - The production chain is re-centred on the three-way median, and all hand rules run unchanged.
  - Representative-origin P50 WAPE: 0.302, against 0.313 for the NN candidate and 0.329 for the submitted method.
  - Total **615,937** (Jul 178,334; Aug 178,372; Sep 129,368; Oct 129,863). That is +6.0% against the submitted 580,968 and −1.8% against the NN candidate.
  - Confidence levels 1 / 2 / 3: 5 / 69 / 26 items.
  - The workbook passes the explanation-consistency check.
  - The submitted workbook is unchanged.

## Relative to a recency baseline (D109)
- **Baseline:** R3 = last 3 months × 4.
- **Metrics:**
  - rWAPE = system WAPE ÷ R3 WAPE on the same items (below 1 beats recency);
  - win = share of items where the system beats R3;
  - mdlog = median |log error|, every item weighted equally.
- Full output: `eda/relative_eval_output.txt`.

| Nine recent origins | rWAPE | win | mdlog |
|---|---|---|---|
| **Rules × quantile × NN** | **0.440** | **75.5%** | **0.277** |
| Quantile × NN | 0.457 | 74.3% | 0.284 |
| Quantile median | 0.471 | 73.2% | 0.297 |
| Rules | 0.488 | 71.1% | 0.310 |
| Last month × 4 | 0.936 | 57.4% | 0.525 |

- **Where recency was right after the fact** (within 20%, 18% of items): every system has about twice R3's error, with the three-way at 2.04. This band is chosen by R3 being right.
- **Where recency was badly wrong** (over 50%, 56% of items): the systems score 0.35–0.37 and win on more than 92% of items.
- **Split by recency's record at the previous window,** which is known in advance: the models beat recency in every band, including the "recency was good" band (three-way 0.457). No subset can be picked in advance where recency should be used instead.

## Metric suite (D110)
Nine recent origins. MdASE = median error ÷ the item's past recency error. relMAE = typical-item error ratio against a reference. Pinball covers P10–P90. Simulation = buying-simulation profit lost, as a share of perfect-foresight profit, over 11 windows.

| | MdASE | relMAE vs recency | relMAE vs rules | MAE + abs bias | Pinball | Sim loss |
|---|---|---|---|---|---|---|
| Recency (last 3 months × 4) | 0.679 | 1 | 2.06 | 1.137 | 0.281 | 19.5% |
| Rules | 0.299 | 0.485 | 1 | 0.450 | 0.132 | 12.4% |
| Quantile model | 0.288 | 0.478 | 0.986 | 0.420 | 0.123 | 12.8% |
| Quantile × NN | 0.280 | 0.471 | 0.971 | 0.400 | 0.120 | 12.2% |
| **Rules × quantile × NN** | **0.272** | **0.452** | **0.931** | **0.392** | **0.117** | **11.8%** |

- The three-way combination is best on every metric. The bootstrap differences against quantile × NN are significant.
- In the simulation it loses in the May–July 2025 windows, because of the rules' back-to-school miss.
- On the 100 targets it is level with quantile × NN rather than better.
