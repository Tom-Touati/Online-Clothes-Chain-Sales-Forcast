# SHAP context for defining explanation categories

Model: LightGBM median of log((4-month units + 1) / (4 x recent level + 1)), i.e. each item's deviation from recency (R3 = last 3 months x 4). SHAP values are in log units (0.1 = about +10%). 13 monthly origins Feb 2025 - Feb 2026; summaries below use the nine recent ones (Jun 2025 - Feb 2026). A fashion e-commerce brand (Gen-Z women's apparel, US).

## Feature dictionary (feature: group: meaning)

- `a_fabric_feel`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_length`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_occasion_tier`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_print_pattern`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_silhouette`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_sleeve`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `a_weight`: product type and attributes: tagger attributes (fabric, sleeve, length, silhouette, print, occasion tier)
- `bT_dT`: temperature / weather: temperature preference x seasonal swing
- `bT_ta`: temperature / weather: temperature preference x origin anomaly
- `back_to_school`: fashion descriptors (LLM scores): LLM item score 0-10
- `beta_T`: temperature / weather: item's temperature preference across states
- `coverage`: fashion descriptors (LLM scores): LLM item score 0-10
- `cp_nb_lvl`: co-purchase cluster seasonality: co-purchase neighbours' sales level
- `cp_nb_sea4`: co-purchase cluster seasonality: co-purchase neighbours' last-year ramp into the target months
- `dT_norm`: temperature / weather: seasonal temperature swing origin -> target
- `dT_real_from_org`: temperature / weather: target normal minus origin felt temperature
- `ev_bfcm_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_bfcm_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_bts_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_bts_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_easter_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_easter_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_festival_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_festival_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_graduation_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_graduation_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_halloween_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_halloween_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_holiday_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_holiday_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_july4_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_july4_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_labor_day_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_labor_day_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_memorial_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_memorial_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_mothers_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_mothers_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_nye_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_nye_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_prime_day_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_prime_day_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_prom_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_prom_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_spring_break_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_spring_break_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_valentine_o`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `ev_valentine_t`: special events / occasions: ev_<event>_t / _o: days of a US event window in the target / origin months
- `f_r28`: recent momentum: last 28 days' rate vs level
- `f_r7`: recent momentum: last 7 days' rate vs level
- `festival`: fashion descriptors (LLM scores): LLM item score 0-10
- `fit_tightness`: fashion descriptors (LLM scores): LLM item score 0-10
- `formality`: fashion descriptors (LLM scores): LLM item score 0-10
- `halloween`: fashion descriptors (LLM scores): LLM item score 0-10
- `lagr1`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `lagr2`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `lagr3`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `lagr4`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `lagr5`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `lagr6`: recent momentum: lagr1..6: log ratio of each of the last 6 months' units to the recent level (momentum)
- `layering`: fashion descriptors (LLM scores): LLM item score 0-10
- `lfl_macro4`: seasonality (history): macro group's seasonal ratio
- `lfl_occ4`: seasonality (history): occasion group's seasonal ratio
- `lfl_type4`: seasonality (history): product type's like-for-like seasonal ratio
- `lvl`: sales level: log recent level
- `occ_last`: special events / occasions: days since its last occasion
- `occ_n4`: special events / occasions: occasions in the 4 target months
- `occ_next`: special events / occasions: days to the item's next occasion
- `own_sea4`: seasonality (history): item's own last-year ratio for the 4 target months
- `party`: fashion descriptors (LLM scores): LLM item score 0-10
- `peak_r`: recent momentum: recent peak vs level
- `ptype`: product type and attributes: product type
- `staple`: fashion descriptors (LLM scores): LLM item score 0-10
- `statement`: fashion descriptors (LLM scores): LLM item score 0-10
- `summer`: fashion descriptors (LLM scores): LLM item score 0-10
- `t_org`: temperature / weather: temperature felt in the origin window (item's regional mix)
- `t_tgt_norm`: temperature / weather: normal temperature of the target months
- `ta_org`: temperature / weather: origin-window temperature anomaly
- `tmonth`: seasonality (history): first target month
- `trend91`: recent momentum: slope over 91 days
- `versatility`: fashion descriptors (LLM scores): LLM item score 0-10
- `warmth`: fashion descriptors (LLM scores): LLM score 0-1 (also coverage, formality, fit_tightness, statement, staple, summer, winter, back_to_school, party, halloween, festival, layering, versatility)
- `winter`: fashion descriptors (LLM scores): LLM item score 0-10
- `x_bts`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months
- `x_festival`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months
- `x_halloween`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months
- `x_party`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months
- `x_summer`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months
- `x_warm_dT`: temperature / weather: LLM warmth x seasonal swing
- `x_winter`: special events / occasions: x_<event>: LLM event-fit score x event days in the target months

## Groups: share of total |SHAP|, mean signed contribution, R^2 of the actual deviation on the group alone

- recent momentum: share 27.1%, mean -0.004, R^2 0.330
- special events / occasions: share 19.8%, mean -0.029, R^2 0.250
- base: share 16.2%, mean -0.128, R^2 0.061
- temperature / weather: share 13.0%, mean +0.005, R^2 0.274
- co-purchase cluster seasonality: share 12.1%, mean -0.067, R^2 0.259
- level offset (young items): share 8.1%, mean +0.056, R^2 0.135
- product type and attributes: share 5.9%, mean -0.002, R^2 0.034
- seasonality (history): share 5.1%, mean -0.030, R^2 0.169
- sales level: share 5.0%, mean +0.005, R^2 0.003
- fashion descriptors (LLM scores): share 3.9%, mean +0.005, R^2 0.071

## Top 40 features by mean |SHAP|

- `cp_nb_sea4` (co-purchase cluster seasonality): 0.0919
- `lagr1` (recent momentum): 0.0770
- `f_r7` (recent momentum): 0.0612
- `lagr6` (recent momentum): 0.0476
- `f_r28` (recent momentum): 0.0456
- `ptype` (product type and attributes): 0.0432
- `x_winter` (special events / occasions): 0.0422
- `x_warm_dT` (temperature / weather): 0.0416
- `lvl` (sales level): 0.0392
- `cp_nb_lvl` (co-purchase cluster seasonality): 0.0359
- `x_bts` (special events / occasions): 0.0327
- `x_summer` (special events / occasions): 0.0317
- `t_org` (temperature / weather): 0.0305
- `bT_dT` (temperature / weather): 0.0294
- `peak_r` (recent momentum): 0.0289
- `ev_spring_break_t` (special events / occasions): 0.0283
- `lagr5` (recent momentum): 0.0251
- `lagr3` (recent momentum): 0.0231
- `t_tgt_norm` (temperature / weather): 0.0224
- `occ_next` (special events / occasions): 0.0207
- `own_sea4` (seasonality (history)): 0.0202
- `lagr2` (recent momentum): 0.0175
- `x_halloween` (special events / occasions): 0.0169
- `occ_last` (special events / occasions): 0.0157
- `ev_prom_o` (special events / occasions): 0.0153
- `beta_T` (temperature / weather): 0.0139
- `winter` (fashion descriptors (LLM scores)): 0.0124
- `tmonth` (seasonality (history)): 0.0123
- `ev_mothers_o` (special events / occasions): 0.0121
- `lagr4` (recent momentum): 0.0114
- `versatility` (fashion descriptors (LLM scores)): 0.0111
- `staple` (fashion descriptors (LLM scores)): 0.0099
- `statement` (fashion descriptors (LLM scores)): 0.0095
- `ev_bfcm_o` (special events / occasions): 0.0085
- `ev_graduation_o` (special events / occasions): 0.0084
- `lfl_type4` (seasonality (history)): 0.0082
- `a_fabric_feel` (product type and attributes): 0.0081
- `lfl_macro4` (seasonality (history)): 0.0075
- `summer` (fashion descriptors (LLM scores)): 0.0074
- `x_festival` (special events / occasions): 0.0066

## Mean signed group contribution by origin month

```
om                                   01     02     06     07     08     09     10     11     12
co-purchase cluster seasonality  -0.111 -0.083 -0.030 -0.014 -0.011 -0.033 -0.042 -0.099 -0.132
fashion descriptors (LLM scores) -0.003 -0.000  0.005  0.008  0.010  0.011  0.014  0.004 -0.002
product type and attributes      -0.008 -0.007  0.008  0.005  0.005  0.003  0.002 -0.004 -0.009
recent momentum                  -0.178 -0.070  0.061  0.180  0.084 -0.095 -0.058  0.182 -0.043
sales level                       0.000  0.000  0.012  0.008  0.010  0.009  0.008  0.004  0.002
seasonality (history)            -0.049 -0.049  0.016  0.010  0.010 -0.005 -0.006 -0.051 -0.104
special events / occasions       -0.143 -0.089  0.269  0.131  0.061  0.031 -0.008 -0.156 -0.176
temperature / weather            -0.068 -0.133  0.103  0.029  0.051  0.131  0.141 -0.041 -0.066
level offset (young items)        0.041  0.104  0.042  0.046  0.047  0.064  0.047  0.060  0.040
```

## How well the model's deviation matches the truth

Correlation of actual vs model deviation from R3: 0.775; residual sd 0.563 log units. The residual (actual minus model) is what the features do not explain.

## Highest-gain trees (origins 2025-06, 2025-10, 2026-02; top 3 levels)

### 2025-06, tree 2 (total gain 315)
```
if f_r7 <= -0.040922779531457 (gain 90.3):
  if x_winter <= 6.000000000000001 (gain 49.0):
    if ev_prom_o <= 40.00000000000001 (gain 34.8):
      ...
    else:
      ...
  else:
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 36.9):
    if f_r7 <= 0.22426098590087387 (gain 6.8):
      ...
    else:
  else:
    if x_warm_dT <= -0.270369133299135 (gain 15.1):
      ...
    else:
      ...
```
### 2025-06, tree 0 (total gain 312)
```
if f_r28 <= 0.01889409178890956 (gain 99.8):
  if ev_prom_o <= 40.00000000000001 (gain 45.3):
    if x_warm_dT <= -6.007852828188766 (gain 33.9):
    else:
      ...
  else:
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 19.5):
    if winter <= 1.5000000000000002 (gain 10.2):
      ...
    else:
  else:
    if winter <= 1.5000000000000002 (gain 11.5):
    else:
      ...
```
### 2025-06, tree 3 (total gain 294)
```
if f_r7 <= -0.040922779531457 (gain 86.7):
  if x_winter <= 8.400000000000002 (gain 43.3):
    if ev_prom_o <= 40.00000000000001 (gain 29.9):
      ...
    else:
      ...
  else:
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 33.3):
    if summer <= 5.500000000000001 (gain 7.7):
    else:
      ...
  else:
    if x_summer <= 1.8000000000000003 (gain 14.6):
      ...
    else:
```
### 2025-10, tree 1 (total gain 452)
```
if f_r7 <= -0.1615509485612117 (gain 144.8):
  if ev_graduation_o <= 38.50000000000001 (gain 64.7):
    if x_winter <= 6.000000000000001 (gain 47.3):
      ...
    else:
  else:
    if winter <= 1.0000000180025095e-35 (gain 22.6):
    else:
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 56.8):
    if f_r7 <= 0.3006175844625569 (gain 21.9):
      ...
    else:
  else:
    if x_summer <= 2.2000000000000006 (gain 19.7):
      ...
    else:
      ...
```
### 2025-10, tree 2 (total gain 426)
```
if f_r7 <= -0.1657518395197047 (gain 134.8):
  if t_org <= 11.187493583181059 (gain 55.1):
    if winter <= 1.5000000000000002 (gain 22.5):
    else:
  else:
    if winter <= 1.5000000000000002 (gain 52.3):
      ...
    else:
      ...
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 48.9):
    if f_r7 <= 0.22589145495223645 (gain 20.2):
      ...
    else:
      ...
  else:
    if peak_r <= 1.3120776510799288 (gain 14.2):
      ...
    else:
```
### 2025-10, tree 0 (total gain 424)
```
if f_r28 <= -0.05937360450295547 (gain 141.4):
  if ev_memorial_o <= 1.0000000180025095e-35 (gain 84.0):
    if x_winter <= 6.000000000000001 (gain 27.5):
      ...
    else:
  else:
    if lagr5 <= -0.0925852645828156 (gain 15.1):
    else:
      ...
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 31.4):
    if cp_nb_sea4 <= 0.39230846699533056 (gain 16.4):
    else:
      ...
  else:
    if winter <= 1.5000000000000002 (gain 18.6):
      ...
    else:
      ...
```
### 2026-02, tree 1 (total gain 744)
```
if f_r7 <= -0.17338351839016705 (gain 249.6):
  if x_warm_dT <= -4.652958025312383 (gain 121.4):
  else:
    if ev_prom_o <= 40.00000000000001 (gain 76.1):
      ...
    else:
      ...
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 60.5):
    if cp_nb_sea4 <= 0.39567831737685466 (gain 62.6):
    else:
      ...
  else:
    if x_summer <= 2.2000000000000006 (gain 20.9):
    else:
```
### 2026-02, tree 0 (total gain 730)
```
if f_r28 <= 0.004555853137268949 (gain 238.9):
  if x_warm_dT <= -6.098849289062651 (gain 102.5):
  else:
    if ev_prom_o <= 40.00000000000001 (gain 117.7):
      ...
    else:
      ...
else:
  if cp_nb_sea4 <= 0.37596134215776417 (gain 54.0):
  else:
    if ev_spring_break_t <= 1.0000000180025095e-35 (gain 34.0):
      ...
    else:
```
### 2026-02, tree 2 (total gain 683)
```
if f_r7 <= -0.17338351839016705 (gain 230.3):
  if x_winter <= 6.000000000000001 (gain 112.4):
    if ev_memorial_o <= 1.0000000180025095e-35 (gain 67.2):
      ...
    else:
      ...
  else:
    if beta_T <= 0.021393636958563746 (gain 15.4):
    else:
else:
  if ev_spring_break_t <= 1.0000000180025095e-35 (gain 52.0):
    if cp_nb_sea4 <= 0.3000989849192413 (gain 55.0):
    else:
      ...
  else:
    if peak_r <= 1.2665213820009658 (gain 15.4):
      ...
    else:
```
