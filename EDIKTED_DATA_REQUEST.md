# What to ask Edikted for

Proposals 1–5 of the error study (decision D323). Each item says what to send, in which format, and what it was worth in the backtest (WAPE ÷ R3 on recent origins; the submitted system scores 0.3851). Every file is keyed by `style_color` as in the data we received. The tools that use each file are already in the repository.

## 1. The roll-out and markdown plan for July–October 2026 (most valuable)

**What:** for each item, and at least for the 100 targets:

- the number of own stores that will carry it;
- the number of wholesale partners;
- the planned change in its web discount;
- the new colours of its style that will launch.

**Format:** fill in `data/plan_template_2026.csv`. It already holds each target's current own stores, partners and web discount (the discount as a share of the list price, so 0.56 means 56% off on average). Leave a cell blank for "no change".

**Use:** `python eda/plan_engine_2026.py data/plan_template_2026.csv out.csv` returns each target's forecast with the plan applied. It multiplies the submitted forecast by the plan's effect, so the hand rules and corrections are kept.

**Worth:**
- Stores, partners, launches and the discount plan known together: 0.3706 against 0.3851, better at all 13 origins (D309, D310).
- Stores alone: 0.3752. The discount plan alone: 0.3818. With the store count off by two stores, the stores' gain falls from 0.010 to 0.005.
- For launches that gained five or more stores, the bias falls from −37% to −28%.

**Also, for the longer term:** receipts per store (date and units received). Sales alone cannot date a store's arrival, so per-store launch curves built from first sales added nothing (D279).

## 2. The wholesale order book

**What:** confirmed partner orders per item per month for July–October 2026: units, partner, ship month.

**Format:** `style_color, partner, ship_month (2026-07 … 2026-10), units`.

**Use:** the partner channel is taken from the orders, and the models forecast only direct demand (web, TikTok, own stores). Fork 2's per-channel model is built for this split.

**Worth:** partners take about 10% of units. Launches that gained two or more partners grew 3.7× R3 and were under-forecast by 24%. Partners added in the window alone moved the correction from 0.3876 to 0.3871 (D310); the confirmed units should be worth more, because they remove a known quantity from the unknown.

## 3. Stock on hand and inbound deliveries

**What:** units on hand per item on 30 June 2026 and planned inbound deliveries per month.

**Format:** `data/stock_2026.csv` with `style_color, on_hand, inbound_jul, inbound_aug, inbound_sep, inbound_oct` (Main, D296).

**Use:** the workbook build caps each month at stock plus inbound. The cap does nothing until this file exists.

**Worth:** small today (0.003), because only 2% of units had stock gaps at the forecast date. It is necessary once the forecast drives purchasing across the whole assortment.

## 4. TikTok Shop analytics

**What:** per item and week, since January 2024: number of creators posting it, videos, views and TikTok Shop orders.

**Format:** `style_color, week_start, creators, videos, views, orders`.

**Use:** a leading signal for launches, as an input to the TikTok channel of Fork 2's model.

**Worth:** cannot be measured from the order lines. TikTok carried 13% of the unexpected units in large under-forecasts, against 7% of those items' base, and its share of units doubled over the backtest.

## 5. The full catalogue, including discontinued items

**What:** the same order-line extract we received, for every item sold since January 2024, including items discontinued before June 2026.

**Use:** the backtest is rebuilt on every item that was active at each origin (Main).

**Worth:** no accuracy gain. It removes an optimistic bias of unknown size from every number we report: 1,431 of the 1,500 items we received are still selling in the last month of data.

## Once the data arrives

| File | Who runs it | What changes |
|---|---|---|
| Filled plan template | Fork: `eda/plan_engine_2026.py` | A plan-driven candidate workbook (Main), on the user's word |
| Order book | Fork 2 | The partner channel is taken from the orders |
| `data/stock_2026.csv` | Main: `eda/181h` / `181i` build | The stock cap |
| TikTok analytics | Fork 2 | TikTok channel inputs |
| Full catalogue | Main | The backtest is rebuilt; every score is re-stated |

July 2026 sales, once known, go through `eda/month1_update_2026.py`. It re-forecasts August–October (D322) and needs no new data beyond July's order lines.
