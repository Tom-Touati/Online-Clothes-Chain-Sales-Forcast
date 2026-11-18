# Item cards

Units are log ratios: 0.1 is about +10%. Deviation = log((x + 1) / (R3 + 1)); R3 = the item's last 3 months x 4.
SHAP values add up to the model deviation together with the base value and the level offset.

## C000: Frances Faux Leather Mini Skort [Skort, BROWN]
- origin 2025-12 (forecast covers the next 4 months); item age 13 months; LLM highlights: fit_tightness 7, party 7, halloween 6, summer 5
- R3 1503 | model 604 (deviation -0.91) | actual 662 (deviation -0.82) | residual +0.09
- base value -0.11; level offset -0.01; other features sum -0.24
- top SHAP: cp_nb_sea4=-0.491 -> -0.161; lagr6=-0.793 -> -0.069; f_r7=-0.424 -> -0.068; lagr1=-0.325 -> -0.049; ptype=Skort -> -0.045; f_r28=-0.572 -> -0.044; x_summer=0.0 -> -0.042; cp_nb_lvl=5.674 -> +0.031; ev_prom_o=0 -> -0.031; lfl_type4=-0.563 -> -0.028; bT_dT=-0.076 -> -0.027; tmonth=1 -> -0.026

## C001: Emerson Button V Neck Top [Top, DARK-GRAY]
- origin 2025-06 (forecast covers the next 4 months); item age 17 months; LLM highlights: fit_tightness 7, back_to_school 6, versatility 6, coverage 5
- R3 92 | model 220 (deviation +0.87) | actual 273 (deviation +1.08) | residual +0.22
- base value -0.15; level offset +0.00; other features sum +0.20
- top SHAP: lagr6=1.606 -> +0.209; x_bts=30.0 -> +0.161; lagr5=1.589 -> +0.096; ev_mothers_o=10 -> +0.076; ev_spring_break_t=0 -> +0.066; cp_nb_lvl=5.095 -> +0.065; t_tgt_norm=20.31 -> +0.064; peak_r=2.71 -> +0.064; ev_prom_o=50 -> +0.057; lagr1=-0.409 -> -0.057; ev_graduation_o=46 -> +0.048; f_r28=-0.395 -> -0.041

## C002: Chelsie Polka Dot Foldover Shorts [Shorts, BLACK-AND-WHITE]
- origin 2025-07 (forecast covers the next 4 months); item age 1 months; LLM highlights: summer 8, fit_tightness 7, festival 5, statement 4
- R3 1255 | model 4219 (deviation +1.21) | actual 8795 (deviation +1.95) | residual +0.73
- base value -0.17; level offset +0.80; other features sum -0.06
- top SHAP: lagr2=nan -> +0.227; f_r7=0.991 -> +0.128; lvl=6.551 -> +0.107; cp_nb_lvl=5.195 -> +0.079; lagr3=nan -> +0.062; x_summer=0.0 -> -0.049; ptype=Shorts -> +0.042; f_r28=0.028 -> +0.034; ev_spring_break_t=0 -> +0.028; summer=8 -> -0.021; t_tgt_norm=16.453 -> +0.021; occ_next=nan -> -0.020

## C003: Mob Wife Faux Fur Jacket [Jacket, BLACK-AND-WHITE]
- origin 2026-01 (forecast covers the next 4 months); item age 14 months; LLM highlights: warmth 10, winter 10, layering 10, coverage 9
- R3 3473 | model 1268 (deviation -1.01) | actual 646 (deviation -1.68) | residual -0.67
- base value -0.11; level offset +0.00; other features sum -0.30
- top SHAP: cp_nb_sea4=-0.703 -> -0.184; x_warm_dT=5.106 -> -0.096; ptype=Jacket -> +0.089; lagr6=-2.877 -> -0.080; lagr1=-0.266 -> -0.058; lagr5=-2.124 -> -0.044; t_org=4.524 -> -0.044; x_summer=0.0 -> -0.042; winter=10 -> -0.036; f_r28=-0.32 -> -0.035; x_bts=0.0 -> -0.033; ev_spring_break_t=31 -> -0.033

## C004: France Oversized Sweatshirt [Sweatshirt, NAVY]
- origin 2026-01 (forecast covers the next 4 months); item age 18 months; LLM highlights: coverage 9, back_to_school 9, winter 8, versatility 8
- R3 231 | model 85 (deviation -0.99) | actual 132 (deviation -0.55) | residual +0.43
- base value -0.11; level offset +0.00; other features sum -0.23
- top SHAP: cp_nb_sea4=-0.992 -> -0.199; own_sea4=-1.138 -> -0.185; f_r7=0.241 -> +0.071; lagr6=-0.897 -> -0.070; lagr5=-1.031 -> -0.046; lvl=4.075 -> -0.041; ev_spring_break_t=31 -> -0.032; t_org=4.628 -> -0.030; ptype=Sweatshirt -> -0.030; x_summer=0.4 -> -0.029; bT_dT=-0.064 -> -0.029; x_warm_dT=3.554 -> -0.028

## C005: Lockie Striped Knit Babydoll Top [Sweater, LIGHT-GRAY]
- origin 2026-01 (forecast covers the next 4 months); item age 4 months; LLM highlights: back_to_school 7, versatility 6, warmth 5, winter 5
- R3 14905 | model 4232 (deviation -1.26) | actual 3842 (deviation -1.36) | residual -0.10
- base value -0.11; level offset +0.00; other features sum -0.37
- top SHAP: ptype=Sweater -> -0.133; cp_nb_lvl=7.869 -> -0.094; lagr1=-0.298 -> -0.083; f_r7=-0.429 -> -0.083; t_org=3.954 -> -0.076; x_summer=0.8 -> -0.053; bT_dT=-0.068 -> -0.049; ev_bfcm_o=4 -> -0.048; lagr6=nan -> -0.046; ev_spring_break_t=31 -> -0.043; x_bts=0.0 -> -0.043; x_winter=0.0 -> -0.036

## C006: Frayed Vintage Washed Denim Shorts [Short Jeans, BLUE-WASHED]
- origin 2025-12 (forecast covers the next 4 months); item age 29 months; LLM highlights: summer 10, festival 8, back_to_school 5, versatility 5
- R3 2017 | model 2723 (deviation +0.30) | actual 3987 (deviation +0.68) | residual +0.38
- base value -0.11; level offset -0.01; other features sum -0.00
- top SHAP: lagr6=1.732 -> +0.208; cp_nb_sea4=-0.234 -> -0.199; lagr5=1.673 -> +0.183; x_festival=20.0 -> +0.120; ptype=Short Jeans -> +0.090; peak_r=1.732 -> +0.059; own_sea4=0.716 -> +0.041; lvl=6.219 -> +0.039; cp_nb_lvl=7.051 -> -0.036; tmonth=1 -> -0.035; x_summer=0.0 -> -0.030; bT_dT=-0.058 -> -0.025

## C007: Eira Sheer Lace Tank Top [Tank Top, BLACK]
- origin 2025-07 (forecast covers the next 4 months); item age 22 months; LLM highlights: fit_tightness 8, summer 8, party 6, formality 5
- R3 5256 | model 4701 (deviation -0.11) | actual 7828 (deviation +0.40) | residual +0.51
- base value -0.17; level offset -0.00; other features sum -0.07
- top SHAP: cp_nb_sea4=0.602 -> -0.067; f_r28=0.167 -> +0.058; x_halloween=9.3 -> +0.049; lvl=7.178 -> +0.042; peak_r=1.092 -> +0.040; x_summer=0.0 -> -0.038; cp_nb_lvl=7.085 -> -0.037; f_r7=0.23 -> +0.032; occ_last=167.0 -> +0.032; x_winter=0.0 -> -0.024; ev_spring_break_t=0 -> +0.023; ev_mothers_o=10 -> +0.022

## C008: Nairobia Textured Mesh Mini Dress [Dress, WHITE]
- origin 2026-02 (forecast covers the next 4 months); item age 1 months; LLM highlights: fit_tightness 9, party 8, summer 6, formality 6
- R3 355 | model 1285 (deviation +1.28) | actual 7628 (deviation +3.07) | residual +1.78
- base value -0.10; level offset +0.99; other features sum +0.02
- top SHAP: lagr2=nan -> +0.171; cp_nb_sea4=0.22 -> -0.113; occ_next=93.0 -> +0.106; lagr3=nan -> +0.089; f_r7=0.528 -> +0.081; t_org=1.842 -> -0.069; cp_nb_lvl=5.726 -> +0.057; occ_last=272.0 -> +0.057; x_summer=6.0 -> +0.037; f_r28=0.087 -> +0.031; bT_dT=-0.214 -> -0.030; ev_spring_break_t=31 -> -0.030

## C009: Keinan Low Rise Denim Shorts [Short Jeans, BLUE-WASHED]
- origin 2025-10 (forecast covers the next 4 months); item age 2 months; LLM highlights: summer 9, staple 6, versatility 6, festival 5
- R3 1495 | model 1614 (deviation +0.08) | actual 3490 (deviation +0.85) | residual +0.77
- base value -0.12; level offset +0.17; other features sum -0.03
- top SHAP: beta_T=0.108 -> +0.086; f_r7=-0.433 -> -0.066; lvl=6.096 -> +0.054; ptype=Short Jeans -> +0.048; x_summer=0.0 -> -0.044; staple=6 -> +0.032; x_bts=0.0 -> -0.030; bT_ta=-0.019 -> +0.028; summer=9 -> -0.027; cp_nb_lvl=6.652 -> -0.025; bT_dT=-1.512 -> +0.023; x_winter=0.0 -> -0.022

## C010: Twyla Polka Dot Halter Top [Top, CREAM]
- origin 2025-11 (forecast covers the next 4 months); item age 4 months; LLM highlights: fit_tightness 9, summer 7, party 4, statement 4
- R3 1477 | model 923 (deviation -0.47) | actual 594 (deviation -0.91) | residual -0.44
- base value -0.11; level offset +0.00; other features sum -0.26
- top SHAP: f_r7=1.062 -> +0.111; ptype=Top -> -0.068; cp_nb_lvl=6.989 -> -0.064; bT_dT=-0.655 -> -0.056; beta_T=0.068 -> +0.049; x_bts=0.0 -> -0.044; x_summer=0.0 -> -0.043; ev_spring_break_t=31 -> -0.040; lvl=5.918 -> +0.028; cp_nb_sea4=nan -> +0.028; lagr5=nan -> -0.027; winter=0 -> +0.025

## C011: Tadria Drawstring Baby Doll Top [Strapless, WHITE]
- origin 2025-06 (forecast covers the next 4 months); item age 14 months; LLM highlights: summer 8, fit_tightness 6, festival 5, statement 5
- R3 2525 | model 1981 (deviation -0.24) | actual 1272 (deviation -0.69) | residual -0.44
- base value -0.15; level offset +0.00; other features sum -0.06
- top SHAP: cp_nb_sea4=-0.1 -> -0.121; lagr1=-0.479 -> -0.080; ev_prom_o=50 -> +0.075; lagr6=-0.612 -> -0.062; ev_mothers_o=10 -> +0.060; f_r28=-0.489 -> -0.055; lvl=6.453 -> +0.048; f_r7=-0.79 -> -0.046; ev_graduation_o=46 -> +0.045; t_org=17.753 -> +0.035; x_warm_dT=0.385 -> +0.033; t_tgt_norm=21.092 -> +0.029

## C012: 80 Oversized T-Shirt [T-Shirt, NAVY]
- origin 2025-08 (forecast covers the next 4 months); item age 24 months; LLM highlights: back_to_school 8, coverage 7, versatility 6, staple 5
- R3 1920 | model 2878 (deviation +0.40) | actual 1973 (deviation +0.03) | residual -0.38
- base value -0.16; level offset -0.00; other features sum +0.00
- top SHAP: lagr1=0.32 -> +0.151; x_winter=7.2 -> +0.105; peak_r=2.161 -> +0.081; f_r28=0.241 -> +0.064; lvl=6.174 -> +0.059; x_warm_dT=-4.592 -> +0.050; cp_nb_sea4=0.561 -> -0.048; ev_spring_break_t=0 -> +0.037; t_org=23.201 -> +0.037; lagr6=0.703 -> +0.034; lagr3=-0.572 -> +0.034; f_r7=-0.169 -> -0.033

## C013: Gianni Square Sunglasses [Sunglasses, BROWN]
- origin 2025-08 (forecast covers the next 4 months); item age 2 months; LLM highlights: summer 10, versatility 9, staple 6, festival 6
- R3 301 | model 372 (deviation +0.21) | actual 666 (deviation +0.79) | residual +0.58
- base value -0.16; level offset +0.62; other features sum -0.02
- top SHAP: cp_nb_sea4=-0.57 -> -0.099; lvl=4.952 -> -0.080; ptype=Sunglasses -> -0.078; versatility=9 -> +0.057; x_winter=4.8 -> +0.050; lagr3=nan -> +0.046; x_summer=0.0 -> -0.041; bT_dT=-0.35 -> -0.030; peak_r=-0.046 -> -0.028; f_r28=-0.09 -> -0.027; ev_spring_break_t=0 -> +0.024; x_bts=3.5 -> -0.024

## C014: Isabetta Tie Front Top [Tank Top, WHITE]
- origin 2025-09 (forecast covers the next 4 months); item age 23 months; LLM highlights: fit_tightness 8, summer 8, festival 5, versatility 5
- R3 132 | model 73 (deviation -0.58) | actual 49 (deviation -0.98) | residual -0.40
- base value -0.15; level offset -0.03; other features sum -0.11
- top SHAP: cp_nb_sea4=-0.717 -> -0.119; peak_r=1.808 -> +0.074; f_r7=-0.858 -> -0.074; f_r28=-0.376 -> -0.056; lagr1=-0.168 -> -0.044; x_summer=0.0 -> -0.035; own_sea4=-0.643 -> -0.022; x_winter=0.0 -> -0.022; lagr5=0.627 -> +0.021; beta_T=0.021 -> +0.019; bT_dT=-0.32 -> -0.017; ev_mothers_o=0 -> -0.016

## C015: Raelynn Washed Low Rise Baggy Jeans [Jeans, BLUE]
- origin 2025-12 (forecast covers the next 4 months); item age 32 months; LLM highlights: coverage 9, back_to_school 9, versatility 8, winter 5
- R3 43292 | model 16286 (deviation -0.98) | actual 15131 (deviation -1.05) | residual -0.07
- base value -0.11; level offset -0.00; other features sum -0.22
- top SHAP: own_sea4=-0.768 -> -0.168; cp_nb_sea4=-0.57 -> -0.164; x_summer=0.0 -> -0.050; f_r7=-0.216 -> -0.047; tmonth=1 -> -0.046; lvl=9.289 -> -0.042; ev_spring_break_t=31 -> -0.036; statement=2 -> +0.035; t_tgt_norm=6.003 -> -0.033; lagr6=-0.072 -> -0.032; f_r28=-0.183 -> -0.032; cp_nb_lvl=7.109 -> -0.031

## C016: Sasha Bow Detail Hoodie [Hoodie, BURGUNDY]
- origin 2025-10 (forecast covers the next 4 months); item age 9 months; LLM highlights: coverage 8, winter 8, back_to_school 8, warmth 7
- R3 673 | model 1303 (deviation +0.66) | actual 1760 (deviation +0.96) | residual +0.30
- base value -0.12; level offset -0.01; other features sum -0.05
- top SHAP: x_warm_dT=-10.802 -> +0.330; x_winter=19.2 -> +0.212; lagr1=0.308 -> +0.111; bT_dT=0.748 -> +0.101; cp_nb_sea4=2.301 -> +0.100; f_r28=0.338 -> +0.070; t_org=18.661 -> +0.055; f_r7=-0.648 -> -0.049; lagr5=-1.082 -> -0.047; lvl=5.125 -> -0.037; beta_T=-0.048 -> -0.030; tmonth=11 -> +0.027

## C017: Nikki Nylon Shorts [Shorts, DARK-GREEN]
- origin 2025-07 (forecast covers the next 4 months); item age 16 months; LLM highlights: summer 9, fit_tightness 5, back_to_school 5, versatility 5
- R3 9797 | model 6754 (deviation -0.37) | actual 4428 (deviation -0.79) | residual -0.42
- base value -0.17; level offset -0.00; other features sum -0.09
- top SHAP: cp_nb_sea4=-0.098 -> -0.076; x_bts=19.0 -> +0.071; lagr6=-0.578 -> -0.069; cp_nb_lvl=7.254 -> -0.047; x_summer=0.0 -> -0.044; peak_r=1.071 -> +0.042; summer=9 -> -0.036; f_r28=0.091 -> +0.035; lvl=7.8 -> +0.034; statement=2 -> +0.033; x_winter=0.0 -> -0.025; lagr5=-0.113 -> -0.024

## C018: Basel Button Up Tank Top [Tank Top, WHITE]
- origin 2025-10 (forecast covers the next 4 months); item age 13 months; LLM highlights: summer 8, fit_tightness 6, versatility 5, coverage 5
- R3 4420 | model 3163 (deviation -0.33) | actual 3553 (deviation -0.22) | residual +0.12
- base value -0.12; level offset -0.01; other features sum -0.10
- top SHAP: f_r7=-0.958 -> -0.069; lagr1=-0.445 -> -0.055; lvl=6.994 -> +0.055; f_r28=-0.437 -> -0.054; ptype=Tank Top -> +0.054; t_org=19.473 -> +0.044; x_summer=0.0 -> -0.038; lagr3=0.457 -> -0.038; statement=2 -> +0.034; cp_nb_sea4=0.4 -> -0.032; a_fabric_feel=cotton / terry -> +0.031; x_bts=0.0 -> -0.026

## C019: Rae Denim Mini Skort [Skort, DARK-BLUE]
- origin 2025-07 (forecast covers the next 4 months); item age 7 months; LLM highlights: summer 7, back_to_school 6, coverage 5, versatility 5
- R3 4640 | model 3984 (deviation -0.15) | actual 5630 (deviation +0.19) | residual +0.35
- base value -0.17; level offset -0.00; other features sum -0.07
- top SHAP: lagr1=0.326 -> +0.115; x_bts=22.8 -> +0.113; ptype=Skort -> -0.109; lagr6=-1.672 -> -0.104; f_r28=0.318 -> +0.087; lvl=7.057 -> +0.067; lagr5=-1.619 -> -0.067; f_r7=0.408 -> +0.065; x_summer=0.0 -> -0.041; trend91=1.118 -> -0.037; cp_nb_lvl=6.993 -> -0.029; ev_mothers_o=10 -> +0.028

## C020: Flower Girl Cord Necklace [Necklace, MIX]
- origin 2026-01 (forecast covers the next 4 months); item age 29 months; LLM highlights: festival 7, summer 7, coverage 5, versatility 5
- R3 43 | model 19 (deviation -0.79) | actual 0 (deviation -3.78) | residual -2.98
- base value -0.11; level offset +0.00; other features sum -0.08
- top SHAP: lagr1=-2.46 -> -0.378; cp_nb_sea4=-0.072 -> -0.134; f_r7=-2.46 -> -0.127; ptype=Necklace -> +0.058; x_festival=17.5 -> +0.057; lagr3=0.536 -> -0.053; cp_nb_lvl=3.712 -> +0.042; f_r28=-2.46 -> -0.037; peak_r=2.104 -> +0.033; x_winter=0.0 -> -0.028; lvl=2.46 -> -0.025; ev_prom_o=0 -> -0.023

## C021: Massie Tailored Button Up Shirt [Shirt, WHITE]
- origin 2025-08 (forecast covers the next 4 months); item age 13 months; LLM highlights: coverage 7, versatility 7, staple 6, back_to_school 6
- R3 12953 | model 11254 (deviation -0.14) | actual 17631 (deviation +0.31) | residual +0.45
- base value -0.16; level offset -0.01; other features sum -0.06
- top SHAP: occ_next=61.0 -> +0.087; occ_last=304.0 -> +0.083; f_r7=-0.363 -> -0.076; cp_nb_lvl=6.74 -> -0.058; versatility=7 -> +0.045; lvl=8.075 -> -0.036; lagr1=0.04 -> +0.034; x_summer=0.0 -> -0.030; bT_dT=-0.231 -> -0.029; f_r28=-0.023 -> +0.026; lagr3=-0.17 -> +0.023; staple=6 -> +0.022

## C022: Vera Grommet Draped Mini Dress [Dress, BLACK]
- origin 2025-10 (forecast covers the next 4 months); item age 6 months; LLM highlights: fit_tightness 9, party 9, statement 7, formality 7
- R3 1404 | model 1074 (deviation -0.27) | actual 826 (deviation -0.53) | residual -0.26
- base value -0.12; level offset -0.01; other features sum -0.10
- top SHAP: beta_T=0.082 -> +0.098; occ_next=106.0 -> +0.081; f_r7=-1.116 -> -0.073; occ_last=259.0 -> +0.069; lagr6=-0.665 -> -0.063; f_r28=-0.423 -> -0.055; lagr1=-0.406 -> -0.053; cp_nb_sea4=0.356 -> -0.041; bT_ta=-0.02 -> +0.036; t_org=20.495 -> +0.034; ev_mothers_o=0 -> -0.033; ev_graduation_o=0 -> -0.030

## C023: Faux Leather Straight Leg Pants [Pants, BLACK]
- origin 2025-09 (forecast covers the next 4 months); item age 24 months; LLM highlights: coverage 9, party 6, formality 5, fit_tightness 5
- R3 533 | model 1097 (deviation +0.72) | actual 1956 (deviation +1.30) | residual +0.58
- base value -0.15; level offset +0.00; other features sum +0.02
- top SHAP: x_warm_dT=-7.843 -> +0.266; x_winter=12.0 -> +0.129; x_halloween=15.5 -> +0.088; ptype=Pants -> +0.079; peak_r=1.871 -> +0.074; lvl=4.901 -> -0.064; occ_next=31.0 -> +0.057; t_org=22.552 -> +0.053; lagr1=0.103 -> +0.051; bT_dT=0.275 -> +0.041; f_r7=0.402 -> +0.039; cp_nb_lvl=5.369 -> +0.039

## C024: Oversized Off Shoulder Polka Dot Sweater [Sweater, NAVY]
- origin 2025-11 (forecast covers the next 4 months); item age 4 months; LLM highlights: back_to_school 7, warmth 6, winter 6, coverage 6
- R3 3795 | model 3526 (deviation -0.07) | actual 3073 (deviation -0.21) | residual -0.14
- base value -0.11; level offset +0.00; other features sum -0.33
- top SHAP: ev_spring_break_t=31 -> -0.114; x_warm_dT=-6.27 -> +0.106; lagr1=0.389 -> +0.102; x_winter=14.4 -> +0.095; f_r7=1.299 -> +0.085; bT_dT=0.303 -> +0.081; f_r28=0.444 -> +0.069; t_tgt_norm=3.5 -> -0.067; lvl=6.86 -> +0.052; x_bts=0.0 -> -0.049; t_org=14.207 -> -0.036; cp_nb_sea4=nan -> +0.036

## C025: Petite Raelynn Washed Low Rise Baggy Jeans [Jeans, TINTED-DENIM]
- origin 2026-02 (forecast covers the next 4 months); item age 5 months; LLM highlights: coverage 9, back_to_school 8, versatility 6, warmth 5
- R3 2441 | model 1457 (deviation -0.52) | actual 1150 (deviation -0.75) | residual -0.24
- base value -0.10; level offset +0.05; other features sum -0.16
- top SHAP: x_warm_dT=5.402 -> -0.121; lagr1=-0.374 -> -0.101; bT_dT=0.252 -> +0.082; t_org=3.942 -> -0.079; f_r7=-0.337 -> -0.059; lvl=6.463 -> +0.057; f_r28=-0.29 -> -0.051; ptype=Jeans -> +0.044; cp_nb_sea4=nan -> +0.043; ev_spring_break_t=31 -> -0.039; lagr5=-1.521 -> -0.038; lagr3=0.255 -> -0.037

## C026: Sequin Leopard Printed Micro Shorts [Shorts, LEOPARD]
- origin 2025-12 (forecast covers the next 4 months); item age 18 months; LLM highlights: statement 10, fit_tightness 9, party 9, festival 8
- R3 11643 | model 4583 (deviation -0.93) | actual 4864 (deviation -0.87) | residual +0.06
- base value -0.11; level offset -0.01; other features sum -0.16
- top SHAP: cp_nb_sea4=-0.167 -> -0.189; lagr1=-0.775 -> -0.138; own_sea4=-0.727 -> -0.082; x_festival=20.0 -> +0.071; f_r7=-0.195 -> -0.046; lagr3=0.778 -> -0.045; ptype=Shorts -> -0.042; cp_nb_lvl=6.859 -> -0.041; f_r28=-0.784 -> -0.040; x_summer=0.0 -> -0.040; bT_dT=-0.06 -> -0.034; ev_prom_o=0 -> -0.032

## C027: Selena Lace Up Corset [Corset, RED]
- origin 2025-11 (forecast covers the next 4 months); item age 22 months; LLM highlights: fit_tightness 9, halloween 9, party 8, statement 7
- R3 2937 | model 1591 (deviation -0.61) | actual 1135 (deviation -0.95) | residual -0.34
- base value -0.11; level offset +0.00; other features sum -0.16
- top SHAP: own_sea4=-0.962 -> -0.170; cp_nb_sea4=-0.453 -> -0.157; occ_next=76.0 -> +0.094; lagr1=-0.342 -> -0.069; f_r7=0.767 -> +0.067; f_r28=-0.278 -> -0.041; x_summer=0.0 -> -0.039; lvl=6.604 -> +0.037; lagr6=-0.424 -> -0.034; t_tgt_norm=4.741 -> -0.033; ev_spring_break_t=31 -> -0.033; ptype=Corset -> +0.033

## C028: Melodi Faux Leather Micro Shorts [Shorts, GOLD]
- origin 2025-06 (forecast covers the next 4 months); item age 8 months; LLM highlights: party 8, statement 7, fit_tightness 7, formality 6
- R3 365 | model 636 (deviation +0.55) | actual 816 (deviation +0.80) | residual +0.25
- base value -0.15; level offset +0.00; other features sum +0.04
- top SHAP: x_halloween=18.6 -> +0.133; occ_last=242.0 -> +0.105; occ_next=123.0 -> +0.084; ev_mothers_o=10 -> +0.061; peak_r=1.294 -> +0.053; cp_nb_sea4=2.481 -> +0.052; ptype=Shorts -> +0.047; t_tgt_norm=22.381 -> +0.045; lagr6=0.455 -> +0.040; ev_spring_break_t=0 -> +0.035; ev_prom_o=50 -> +0.034; lvl=4.529 -> -0.032

## C029: Kyrah Oversized Knit Sweater [Sweater, LIGHT-BLUE]
- origin 2026-01 (forecast covers the next 4 months); item age 4 months; LLM highlights: coverage 8, back_to_school 8, winter 7, versatility 7
- R3 547 | model 131 (deviation -1.42) | actual 159 (deviation -1.23) | residual +0.19
- base value -0.11; level offset +0.00; other features sum -0.38
- top SHAP: lagr1=-0.977 -> -0.194; cp_nb_sea4=-1.228 -> -0.161; ptype=Sweater -> -0.117; f_r7=-1.48 -> -0.114; t_org=3.838 -> -0.075; lagr3=0.35 -> -0.052; cp_nb_lvl=7.138 -> -0.051; x_summer=0.8 -> -0.044; ev_prom_o=0 -> -0.034; f_r28=-0.955 -> -0.034; x_bts=0.0 -> -0.033; x_winter=0.0 -> -0.032

## C030: Tana Faux Leather Halter Top [Top, BLACK]
- origin 2025-12 (forecast covers the next 4 months); item age 14 months; LLM highlights: fit_tightness 9, party 8, statement 6, formality 6
- R3 4936 | model 2185 (deviation -0.81) | actual 2757 (deviation -0.58) | residual +0.23
- base value -0.11; level offset -0.00; other features sum -0.23
- top SHAP: cp_nb_sea4=-0.271 -> -0.177; ptype=Top -> -0.072; lagr6=-0.637 -> -0.062; x_summer=0.0 -> -0.048; f_r7=0.127 -> +0.039; ev_spring_break_t=31 -> -0.035; x_bts=0.0 -> -0.032; f_r28=-0.131 -> -0.032; lagr5=-0.643 -> -0.031; lvl=7.115 -> +0.031; tmonth=1 -> -0.031; bT_dT=-0.03 -> -0.025

## C031: Contrast Satin Effect Mini Dress [Dress, CREAM]
- origin 2025-12 (forecast covers the next 4 months); item age 8 months; LLM highlights: fit_tightness 9, party 9, formality 7, statement 6
- R3 8635 | model 4750 (deviation -0.60) | actual 4689 (deviation -0.61) | residual -0.01
- base value -0.11; level offset -0.01; other features sum -0.14
- top SHAP: cp_nb_sea4=0.141 -> -0.101; ptype=Dress -> +0.056; f_r28=-0.246 -> -0.046; ev_spring_break_t=31 -> -0.041; x_summer=0.0 -> -0.039; tmonth=1 -> -0.036; lagr6=-0.032 -> -0.029; ev_prom_o=0 -> -0.025; lfl_type4=-0.209 -> -0.023; x_bts=0.0 -> -0.023; t_tgt_norm=6.404 -> -0.020; lfl_macro4=-0.209 -> -0.018

## C032: Stephanie Lace Mini Dress [Dress, YELLOW]
- origin 2026-02 (forecast covers the next 4 months); item age 2 months; LLM highlights: summer 6, coverage 5, party 5, fit_tightness 4
- R3 539 | model 1083 (deviation +0.70) | actual 3997 (deviation +2.00) | residual +1.30
- base value -0.10; level offset +0.64; other features sum -0.14
- top SHAP: f_r7=0.693 -> +0.146; lagr3=nan -> +0.093; t_org=2.018 -> -0.059; ev_spring_break_t=31 -> -0.056; lagr1=0.123 -> +0.054; f_r28=0.207 -> +0.049; x_summer=6.0 -> +0.045; bT_dT=-0.127 -> -0.033; cp_nb_lvl=5.763 -> +0.033; lvl=5.55 -> +0.031; x_bts=0.0 -> -0.025; winter=0 -> +0.025

## C033: Magic 8 long sleeve T shirt [T-Shirt, GRAY-MELANGE]
- origin 2025-07 (forecast covers the next 4 months); item age 21 months; LLM highlights: coverage 7, back_to_school 7, fit_tightness 6, statement 5
- R3 1487 | model 3285 (deviation +0.79) | actual 1970 (deviation +0.28) | residual -0.51
- base value -0.17; level offset +0.00; other features sum +0.13
- top SHAP: x_bts=26.6 -> +0.195; lagr1=0.375 -> +0.120; f_r7=0.801 -> +0.119; cp_nb_sea4=0.391 -> -0.097; lagr6=1.159 -> +0.095; f_r28=0.417 -> +0.087; peak_r=1.887 -> +0.085; lvl=5.922 -> +0.065; ptype=T-Shirt -> +0.049; t_org=20.791 -> +0.039; ev_spring_break_t=0 -> +0.037; t_tgt_norm=15.83 -> +0.031

## C034: Roman Low Rise Baggy Jeans [Jeans, WHITE]
- origin 2025-09 (forecast covers the next 4 months); item age 24 months; LLM highlights: coverage 9, back_to_school 7, versatility 7, warmth 5
- R3 24723 | model 27096 (deviation +0.09) | actual 22288 (deviation -0.10) | residual -0.20
- base value -0.15; level offset -0.00; other features sum -0.03
- top SHAP: x_warm_dT=-7.57 -> +0.228; x_winter=9.6 -> +0.226; cp_nb_sea4=0.161 -> -0.060; f_r28=-0.296 -> -0.055; t_org=22.873 -> +0.054; lagr6=-0.656 -> -0.053; cp_nb_lvl=7.598 -> -0.045; ptype=Jeans -> +0.043; f_r7=-0.674 -> -0.042; versatility=7 -> +0.042; bT_dT=-0.264 -> -0.032; x_summer=0.0 -> -0.028

## C035: Bonney Bow Detail Hoodie [Hoodie, DARK-GRAY]
- origin 2025-08 (forecast covers the next 4 months); item age 12 months; LLM highlights: coverage 9, winter 9, back_to_school 8, warmth 8
- R3 14831 | model 34788 (deviation +0.85) | actual 51790 (deviation +1.25) | residual +0.40
- base value -0.16; level offset -0.01; other features sum +0.02
- top SHAP: x_warm_dT=-9.186 -> +0.243; x_winter=21.6 -> +0.203; lagr1=0.235 -> +0.089; lagr6=1.0 -> +0.088; peak_r=2.963 -> +0.076; cp_nb_sea4=2.365 -> +0.075; f_r28=0.194 -> +0.058; ev_spring_break_t=0 -> +0.042; t_org=23.296 -> +0.041; ptype=Hoodie -> +0.039; f_r7=0.175 -> +0.027; ev_july4_o=10 -> +0.024

## C036: Raelynn Washed Low Rise Baggy Jeans [Jeans, BLACK-WASHED]
- origin 2025-06 (forecast covers the next 4 months); item age 24 months; LLM highlights: coverage 9, back_to_school 8, versatility 7, winter 5
- R3 14879 | model 24537 (deviation +0.50) | actual 28987 (deviation +0.67) | residual +0.17
- base value -0.15; level offset +0.00; other features sum +0.15
- top SHAP: x_bts=40.0 -> +0.178; cp_nb_sea4=0.992 -> -0.121; peak_r=1.579 -> +0.102; lagr6=0.927 -> +0.080; ev_mothers_o=10 -> +0.077; ev_spring_break_t=0 -> +0.068; ev_prom_o=50 -> +0.062; t_tgt_norm=20.666 -> +0.053; lagr1=-0.303 -> -0.050; ptype=Jeans -> +0.048; f_r28=-0.288 -> -0.046; ev_graduation_o=46 -> +0.040

## C037: 80'S Babe Tank Top [Tank Top, BLUE]
- origin 2025-07 (forecast covers the next 4 months); item age 15 months; LLM highlights: summer 8, fit_tightness 7, back_to_school 5, versatility 5
- R3 10456 | model 6250 (deviation -0.51) | actual 6358 (deviation -0.50) | residual +0.02
- base value -0.17; level offset -0.00; other features sum -0.08
- top SHAP: lagr6=-0.932 -> -0.087; cp_nb_sea4=-0.137 -> -0.076; cp_nb_lvl=7.719 -> -0.058; x_bts=19.0 -> +0.051; x_summer=0.0 -> -0.049; lagr5=-0.664 -> -0.040; lagr4=-0.454 -> -0.028; x_winter=0.0 -> -0.027; f_r28=0.076 -> +0.027; summer=8 -> -0.024; lvl=7.869 -> +0.021; ev_mothers_o=10 -> +0.021

## C038: Contrast Satin Effect Mini Dress [Dress, CREAM]
- origin 2026-01 (forecast covers the next 4 months); item age 9 months; LLM highlights: fit_tightness 9, party 9, formality 7, statement 6
- R3 8555 | model 4362 (deviation -0.67) | actual 4459 (deviation -0.65) | residual +0.02
- base value -0.11; level offset +0.00; other features sum -0.16
- top SHAP: cp_nb_sea4=0.133 -> -0.084; f_r7=-0.456 -> -0.075; lagr1=-0.334 -> -0.071; lagr3=0.289 -> -0.038; ptype=Dress -> +0.034; x_winter=0.0 -> -0.032; ev_prom_o=0 -> -0.030; f_r28=-0.356 -> -0.026; ev_spring_break_t=31 -> -0.022; x_bts=0.0 -> -0.021; ev_bfcm_o=4 -> -0.020; lagr6=0.039 -> -0.018

## C039: Sunrise Babydoll Tank Top [Tank Top, NAVY]
- origin 2025-07 (forecast covers the next 4 months); item age 13 months; LLM highlights: summer 8, versatility 6, coverage 5, staple 5
- R3 14991 | model 12346 (deviation -0.19) | actual 9417 (deviation -0.46) | residual -0.27
- base value -0.17; level offset -0.00; other features sum -0.03
- top SHAP: cp_nb_sea4=-0.304 -> -0.073; x_bts=15.2 -> +0.065; lagr6=-0.534 -> -0.059; f_r28=0.161 -> +0.051; f_r7=0.311 -> +0.049; x_summer=0.0 -> -0.044; statement=2 -> +0.040; cp_nb_lvl=7.397 -> -0.033; ev_spring_break_t=0 -> +0.026; summer=8 -> -0.025; ev_mothers_o=10 -> +0.025; x_winter=0.0 -> -0.021

## C040: Low Rise Bermuda Denim Carpenter Shorts [Short Jeans, WHITE]
- origin 2025-07 (forecast covers the next 4 months); item age 17 months; LLM highlights: summer 9, coverage 6, festival 5, versatility 5
- R3 117 | model 85 (deviation -0.32) | actual 0 (deviation -4.77) | residual -4.45
- base value -0.17; level offset -0.05; other features sum -0.03
- top SHAP: lagr1=-3.359 -> -0.210; cp_nb_sea4=-0.275 -> -0.105; lagr5=1.173 -> +0.094; f_r28=-3.359 -> -0.092; peak_r=1.314 -> +0.072; cp_nb_lvl=4.994 -> +0.064; ptype=Short Jeans -> +0.062; ev_mothers_o=10 -> +0.058; f_r7=-3.359 -> -0.043; summer=9 -> -0.041; ev_graduation_o=46 -> +0.037; lagr4=1.314 -> +0.036

## C041: Studded Foldover Micro Shorts [Shorts, BLACK]
- origin 2025-09 (forecast covers the next 4 months); item age 8 months; LLM highlights: fit_tightness 8, festival 8, summer 7, party 7
- R3 6769 | model 3895 (deviation -0.55) | actual 4671 (deviation -0.37) | residual +0.18
- base value -0.15; level offset -0.00; other features sum -0.18
- top SHAP: bT_dT=-0.646 -> -0.070; x_summer=0.0 -> -0.050; x_halloween=12.4 -> +0.046; lvl=7.43 -> +0.046; f_r7=0.004 -> +0.042; cp_nb_lvl=7.138 -> -0.040; lagr6=-0.321 -> -0.039; t_tgt_norm=8.511 -> -0.037; x_bts=0.0 -> -0.033; lagr3=0.079 -> -0.029; t_org=23.26 -> -0.027; f_r28=-0.098 -> -0.023

## C042: Cami Sheer Lace Tank Top [Tank Top, RED]
- origin 2025-10 (forecast covers the next 4 months); item age 15 months; LLM highlights: fit_tightness 9, party 6, statement 5, formality 5
- R3 1556 | model 1078 (deviation -0.37) | actual 702 (deviation -0.80) | residual -0.43
- base value -0.12; level offset -0.01; other features sum -0.15
- top SHAP: cp_nb_sea4=-0.091 -> -0.095; bT_dT=0.666 -> +0.078; f_r7=-1.041 -> -0.072; occ_next=106.0 -> +0.070; occ_last=259.0 -> +0.068; lagr1=-0.307 -> -0.048; lagr3=0.514 -> -0.044; t_org=18.615 -> +0.037; f_r28=-0.286 -> -0.036; cp_nb_lvl=6.783 -> -0.036; lvl=5.952 -> +0.032; t_tgt_norm=3.016 -> -0.032

## C043: Max Suede Detail Denim Micro Shorts [Short Jeans, BLUE-WASHED]
- origin 2025-08 (forecast covers the next 4 months); item age 7 months; LLM highlights: summer 8, fit_tightness 6, festival 6, party 5
- R3 141 | model 106 (deviation -0.28) | actual 51 (deviation -1.01) | residual -0.72
- base value -0.16; level offset -0.01; other features sum -0.15
- top SHAP: cp_nb_sea4=2.316 -> +0.121; f_r7=-0.463 -> -0.095; ptype=Short Jeans -> +0.062; x_summer=0.0 -> -0.044; cp_nb_lvl=7.11 -> -0.035; f_r28=0.021 -> +0.031; x_winter=0.0 -> -0.031; lagr2=0.325 -> -0.031; beta_T=0.018 -> +0.029; lagr1=0.051 -> +0.028; lagr3=-0.542 -> +0.026; ev_graduation_o=15 -> -0.025

## C044: Lillia V Neck Wrap Sweater [Sweater, WHITE]
- origin 2025-12 (forecast covers the next 4 months); item age 2 months; LLM highlights: fit_tightness 6, back_to_school 6, versatility 6, winter 5
- R3 3751 | model 4417 (deviation +0.16) | actual 7822 (deviation +0.73) | residual +0.57
- base value -0.11; level offset +0.62; other features sum -0.33
- top SHAP: f_r7=1.063 -> +0.155; ptype=Sweater -> -0.108; ev_spring_break_t=31 -> -0.097; lagr2=-1.316 -> +0.084; tmonth=1 -> -0.083; f_r28=0.377 -> +0.079; lagr3=nan -> +0.077; lagr1=0.294 -> +0.067; cp_nb_sea4=0.55 -> -0.064; cp_nb_lvl=6.725 -> -0.050; t_tgt_norm=6.038 -> -0.045; x_summer=0.0 -> -0.042

## C045: Bonney Bow Detail Hoodie [Hoodie, CREAM]
- origin 2025-11 (forecast covers the next 4 months); item age 13 months; LLM highlights: coverage 9, winter 9, back_to_school 8, warmth 8
- R3 3185 | model 3079 (deviation -0.03) | actual 4217 (deviation +0.28) | residual +0.31
- base value -0.11; level offset +0.00; other features sum -0.26
- top SHAP: x_warm_dT=-8.322 -> +0.198; lagr1=0.593 -> +0.157; cp_nb_sea4=0.269 -> -0.116; f_r28=0.66 -> +0.098; lagr6=-1.3 -> -0.094; f_r7=1.514 -> +0.088; x_winter=21.6 -> +0.083; ev_spring_break_t=31 -> -0.078; t_tgt_norm=3.667 -> -0.062; lvl=6.685 -> +0.055; bT_dT=0.204 -> +0.047; cp_nb_lvl=7.094 -> -0.047

## C046: Pamper Waffle Micro Shorts [Shorts, WHITE]
- origin 2026-01 (forecast covers the next 4 months); item age 34 months; LLM highlights: summer 10, fit_tightness 7, festival 7, statement 4
- R3 381 | model 152 (deviation -0.91) | actual 216 (deviation -0.57) | residual +0.35
- base value -0.11; level offset +0.00; other features sum -0.25
- top SHAP: cp_nb_sea4=0.002 -> -0.190; lagr1=-0.882 -> -0.169; own_sea4=-0.819 -> -0.144; f_r7=-0.994 -> -0.104; cp_nb_lvl=5.063 -> +0.060; lagr6=0.686 -> +0.051; x_summer=4.0 -> +0.039; lvl=4.571 -> -0.037; lagr3=0.296 -> -0.034; t_org=4.168 -> -0.033; f_r28=-0.852 -> -0.031; peak_r=2.151 -> +0.029

## C047: Amaris Striped Knit Shorts [Shorts, LIGHT-BLUE]
- origin 2026-02 (forecast covers the next 4 months); item age 13 months; LLM highlights: summer 8, fit_tightness 6, versatility 5, back_to_school 4
- R3 9004 | model 5634 (deviation -0.47) | actual 5859 (deviation -0.43) | residual +0.04
- base value -0.10; level offset +0.04; other features sum -0.10
- top SHAP: cp_nb_sea4=0.104 -> -0.112; x_summer=8.0 -> +0.088; occ_next=126.0 -> +0.070; lagr6=-0.637 -> -0.064; t_org=2.285 -> -0.057; lagr1=-0.3 -> -0.055; f_r28=-0.216 -> -0.053; lagr3=0.374 -> -0.046; f_r7=-0.34 -> -0.045; occ_last=239.0 -> +0.039; cp_nb_lvl=6.651 -> -0.039; lagr5=-0.681 -> -0.025

## C048: Dalia Oversized Denim Jacket [Jacket, DARK-BLUE]
- origin 2025-06 (forecast covers the next 4 months); item age 22 months; LLM highlights: layering 10, coverage 9, versatility 9, back_to_school 8
- R3 1260 | model 2139 (deviation +0.53) | actual 2992 (deviation +0.86) | residual +0.34
- base value -0.15; level offset +0.00; other features sum +0.21
- top SHAP: x_bts=40.0 -> +0.197; cp_nb_sea4=0.919 -> -0.122; ev_mothers_o=10 -> +0.079; ev_prom_o=50 -> +0.076; ev_spring_break_t=0 -> +0.066; lagr6=0.485 -> +0.053; ptype=Jacket -> +0.051; lvl=5.759 -> +0.049; versatility=9 -> +0.048; t_tgt_norm=20.304 -> +0.040; f_r7=-0.413 -> -0.039; f_r28=-0.256 -> -0.034

## C049: Chunky Leopard Scarf [Scarf, MIX]
- origin 2025-08 (forecast covers the next 4 months); item age 10 months; LLM highlights: winter 10, warmth 8, statement 6, coverage 6
- R3 357 | model 760 (deviation +0.75) | actual 6584 (deviation +2.91) | residual +2.16
- base value -0.16; level offset +0.00; other features sum +0.04
- top SHAP: x_warm_dT=-9.134 -> +0.234; x_winter=24.0 -> +0.181; lagr1=0.568 -> +0.163; f_r28=0.56 -> +0.114; f_r7=0.752 -> +0.109; peak_r=2.801 -> +0.091; lagr6=1.073 -> +0.077; ptype=Scarf -> -0.075; lvl=4.507 -> -0.071; t_org=23.065 -> +0.046; ev_spring_break_t=0 -> +0.045; winter=10 -> -0.043

## C050: Alexa Linen Blend Pants [Pants, WHITE]
- origin 2025-12 (forecast covers the next 4 months); item age 15 months; LLM highlights: coverage 9, versatility 7, summer 6, staple 6
- R3 1477 | model 962 (deviation -0.43) | actual 1436 (deviation -0.03) | residual +0.40
- base value -0.11; level offset -0.00; other features sum -0.17
- top SHAP: cp_nb_sea4=-0.041 -> -0.176; statement=1 -> +0.054; ptype=Pants -> +0.053; tmonth=1 -> -0.050; x_summer=0.0 -> -0.048; lvl=5.913 -> +0.048; lagr6=0.529 -> +0.046; f_r7=0.164 -> +0.039; cp_nb_lvl=6.784 -> -0.036; x_bts=0.0 -> -0.031; lfl_type4=-0.795 -> -0.030; f_r28=-0.007 -> -0.024

## C051: Oval Buckle Faux Leather Belt [Belt, BROWN]
- origin 2025-10 (forecast covers the next 4 months); item age 7 months; LLM highlights: versatility 8, staple 7, winter 5, back_to_school 5
- R3 3788 | model 5596 (deviation +0.39) | actual 2415 (deviation -0.45) | residual -0.84
- base value -0.12; level offset -0.02; other features sum +0.01
- top SHAP: x_winter=12.0 -> +0.190; ptype=Belt -> +0.183; cp_nb_sea4=5.563 -> +0.143; versatility=8 -> +0.065; f_r7=-0.877 -> -0.064; lvl=6.838 -> +0.060; lagr6=-0.772 -> -0.058; t_org=19.212 -> +0.055; staple=7 -> +0.032; x_summer=0.0 -> -0.029; x_bts=0.0 -> -0.029; cp_nb_lvl=6.789 -> -0.025

## C052: Only Edikted Straight Leg Sweatpants [Sweatpants, AQUA]
- origin 2025-12 (forecast covers the next 4 months); item age 4 months; LLM highlights: coverage 9, staple 7, back_to_school 7, versatility 6
- R3 477 | model 207 (deviation -0.83) | actual 356 (deviation -0.29) | residual +0.54
- base value -0.11; level offset +0.00; other features sum -0.26
- top SHAP: ev_spring_break_t=31 -> -0.081; tmonth=1 -> -0.067; cp_nb_lvl=7.562 -> -0.065; lvl=4.794 -> -0.050; x_summer=0.0 -> -0.047; t_tgt_norm=5.679 -> -0.038; lagr6=nan -> -0.037; x_bts=0.0 -> -0.035; ev_bfcm_o=4 -> -0.034; f_r7=0.085 -> +0.033; beta_T=-0.024 -> -0.026; winter=5 -> -0.025

## C053: Sunrise Babydoll Tank Top [Tank Top, NAVY]
- origin 2025-12 (forecast covers the next 4 months); item age 18 months; LLM highlights: summer 8, versatility 6, coverage 5, staple 5
- R3 5992 | model 3791 (deviation -0.46) | actual 3070 (deviation -0.67) | residual -0.21
- base value -0.11; level offset -0.00; other features sum -0.14
- top SHAP: cp_nb_sea4=-0.229 -> -0.192; lagr6=1.067 -> +0.124; f_r7=-0.45 -> -0.074; lagr5=1.178 -> +0.047; statement=2 -> +0.045; lfl_type4=-0.495 -> -0.041; x_summer=0.0 -> -0.038; tmonth=1 -> -0.036; lvl=7.31 -> +0.032; own_sea4=-0.377 -> -0.028; f_r28=-0.212 -> -0.026; cp_nb_lvl=6.795 -> -0.026

## C054: Keli Studded Mini Skort [Skort, CREAM]
- origin 2026-02 (forecast covers the next 4 months); item age 10 months; LLM highlights: fit_tightness 7, party 7, summer 6, formality 5
- R3 200 | model 184 (deviation -0.08) | actual 325 (deviation +0.48) | residual +0.56
- base value -0.10; level offset +0.07; other features sum -0.15
- top SHAP: lagr6=1.431 -> +0.102; ptype=Skort -> -0.098; cp_nb_sea4=0.308 -> -0.073; x_summer=6.0 -> +0.070; lagr1=-0.361 -> -0.068; f_r7=0.374 -> +0.065; bT_dT=0.743 -> +0.064; lvl=3.999 -> -0.057; cp_nb_lvl=4.601 -> +0.052; peak_r=1.859 -> +0.052; f_r28=-0.28 -> -0.045; bT_ta=-0.04 -> +0.042

## C055: Cecily Strapless Knit Top [Strapless, CREAM]
- origin 2025-09 (forecast covers the next 4 months); item age 18 months; LLM highlights: fit_tightness 9, summer 7, versatility 5, staple 4
- R3 36 | model 23 (deviation -0.45) | actual 8 (deviation -1.41) | residual -0.96
- base value -0.15; level offset +0.00; other features sum -0.11
- top SHAP: lvl=2.306 -> +0.211; lagr1=-0.696 -> -0.123; peak_r=2.227 -> +0.083; f_r7=-2.306 -> -0.082; f_r28=-0.856 -> -0.080; own_sea4=-1.265 -> -0.078; cp_nb_sea4=-0.268 -> -0.076; ptype=Strapless -> -0.041; lagr6=0.092 -> -0.038; x_warm_dT=-3.052 -> +0.031; x_summer=0.0 -> -0.031; cp_nb_lvl=5.274 -> +0.029

## C056: 80 Oversized T-Shirt [T-Shirt, NAVY]
- origin 2025-06 (forecast covers the next 4 months); item age 22 months; LLM highlights: back_to_school 8, coverage 7, versatility 6, staple 5
- R3 1347 | model 2827 (deviation +0.74) | actual 1783 (deviation +0.28) | residual -0.46
- base value -0.15; level offset +0.00; other features sum +0.09
- top SHAP: x_bts=40.0 -> +0.199; lagr6=1.513 -> +0.177; cp_nb_sea4=0.836 -> -0.125; peak_r=2.509 -> +0.101; ev_spring_break_t=0 -> +0.074; lagr5=1.218 -> +0.065; ev_mothers_o=10 -> +0.062; lvl=5.826 -> +0.060; t_tgt_norm=20.532 -> +0.059; ev_prom_o=50 -> +0.049; ptype=T-Shirt -> +0.040; t_org=17.213 -> +0.035

## C057: Eleanor Bra Detail Sheer Lace Top [Tank Top, WHITE]
- origin 2025-06 (forecast covers the next 4 months); item age 18 months; LLM highlights: fit_tightness 8, summer 8, party 6, festival 5
- R3 2004 | model 2215 (deviation +0.10) | actual 1198 (deviation -0.51) | residual -0.61
- base value -0.15; level offset +0.00; other features sum +0.07
- top SHAP: cp_nb_sea4=0.644 -> -0.078; peak_r=1.399 -> +0.066; t_tgt_norm=20.473 -> +0.060; lvl=6.222 -> +0.053; ptype=Tank Top -> +0.040; t_org=17.216 -> +0.034; x_halloween=9.3 -> +0.034; winter=0 -> -0.032; cp_nb_lvl=6.895 -> -0.032; ev_spring_break_t=0 -> +0.031; x_summer=3.2 -> +0.027; beta_T=-0.022 -> -0.025

## C058: Contrast Satin Effect Mini Dress [Dress, CREAM]
- origin 2026-02 (forecast covers the next 4 months); item age 10 months; LLM highlights: fit_tightness 9, party 9, formality 7, statement 6
- R3 5971 | model 3457 (deviation -0.55) | actual 4110 (deviation -0.37) | residual +0.17
- base value -0.10; level offset +0.04; other features sum -0.16
- top SHAP: lagr1=-0.526 -> -0.112; cp_nb_sea4=0.177 -> -0.088; f_r7=-0.572 -> -0.078; t_org=2.798 -> -0.077; f_r28=-0.443 -> -0.051; lagr6=0.647 -> +0.048; cp_nb_lvl=5.437 -> +0.043; ptype=Dress -> +0.042; x_winter=0.0 -> -0.030; x_summer=4.0 -> +0.027; lagr3=0.264 -> -0.027; x_warm_dT=2.261 -> -0.026

## C059: Sequin Fringe Knit Poncho [Knitted Top, PINK]
- origin 2025-10 (forecast covers the next 4 months); item age 6 months; LLM highlights: statement 8, party 8, festival 8, formality 6
- R3 4460 | model 2861 (deviation -0.44) | actual 2457 (deviation -0.60) | residual -0.15
- base value -0.12; level offset -0.02; other features sum +0.04
- top SHAP: lagr1=-0.614 -> -0.150; lagr6=-1.228 -> -0.095; f_r7=-0.97 -> -0.076; occ_last=259.0 -> +0.068; f_r28=-0.634 -> -0.065; occ_next=61.0 -> +0.062; ptype=Knitted Top -> -0.060; t_org=20.134 -> +0.056; lagr3=0.403 -> -0.046; beta_T=0.065 -> +0.046; cp_nb_lvl=6.944 -> -0.039; ev_mothers_o=0 -> -0.035

## C060: Me Time Oversized Waffle Top [Top, GRAY-MELANGE]
- origin 2025-12 (forecast covers the next 4 months); item age 25 months; LLM highlights: coverage 8, staple 8, versatility 8, back_to_school 7
- R3 9611 | model 2997 (deviation -1.17) | actual 3966 (deviation -0.88) | residual +0.28
- base value -0.11; level offset -0.00; other features sum -0.34
- top SHAP: cp_nb_sea4=-0.926 -> -0.175; own_sea4=-1.053 -> -0.136; ptype=Top -> -0.070; lagr6=-0.752 -> -0.067; f_r7=-0.298 -> -0.059; tmonth=1 -> -0.053; x_summer=0.0 -> -0.044; cp_nb_lvl=7.069 -> -0.043; t_tgt_norm=5.634 -> -0.037; statement=1 -> +0.037; x_bts=0.0 -> -0.033; ev_spring_break_t=31 -> -0.032

## C061: Rhian Lacey Pleated Babydoll Top [Tank Top, BLACK]
- origin 2025-10 (forecast covers the next 4 months); item age 2 months; LLM highlights: party 7, summer 6, statement 5, formality 5
- R3 2235 | model 4684 (deviation +0.74) | actual 25525 (deviation +2.44) | residual +1.70
- base value -0.12; level offset +0.51; other features sum -0.07
- top SHAP: lagr1=0.319 -> +0.093; lagr3=nan -> +0.092; occ_next=106.0 -> +0.079; occ_last=259.0 -> +0.079; f_r28=0.398 -> +0.073; f_r7=-0.406 -> -0.062; t_tgt_norm=3.521 -> -0.058; lvl=6.834 -> +0.052; lagr2=-0.842 -> +0.042; t_org=19.101 -> +0.035; ptype=Tank Top -> +0.031; x_bts=0.0 -> -0.031

## C062: Brie Cut Out Crochet Mini Dress [Dress, CREAM]
- origin 2025-11 (forecast covers the next 4 months); item age 21 months; LLM highlights: festival 8, summer 7, statement 6, party 5
- R3 1784 | model 2118 (deviation +0.17) | actual 5478 (deviation +1.12) | residual +0.95
- base value -0.11; level offset +0.00; other features sum -0.14
- top SHAP: lagr1=0.49 -> +0.125; f_r7=1.567 -> +0.106; cp_nb_lvl=5.05 -> +0.105; cp_nb_sea4=0.344 -> -0.081; f_r28=0.57 -> +0.073; ptype=Dress -> +0.064; lagr5=0.822 -> +0.059; t_tgt_norm=4.801 -> -0.038; x_bts=0.0 -> -0.037; lvl=6.106 -> +0.036; lagr6=0.725 -> +0.036; ev_spring_break_t=31 -> -0.034

## C063: Billiard Oversized Hoodie [Hoodie, GRAY-MELANGE]
- origin 2025-12 (forecast covers the next 4 months); item age 16 months; LLM highlights: back_to_school 8, winter 8, warmth 7, coverage 7
- R3 1965 | model 845 (deviation -0.84) | actual 718 (deviation -1.01) | residual -0.16
- base value -0.11; level offset -0.00; other features sum -0.36
- top SHAP: cp_nb_sea4=-0.627 -> -0.169; own_sea4=-0.66 -> -0.096; lagr6=-0.892 -> -0.071; beta_T=-0.046 -> +0.054; cp_nb_lvl=6.768 -> -0.049; tmonth=1 -> -0.048; x_summer=0.0 -> -0.046; f_r7=0.03 -> +0.044; lvl=6.195 -> +0.038; ev_spring_break_t=31 -> -0.034; peak_r=1.347 -> +0.033; x_bts=0.0 -> -0.025

## C064: Emerson Button V Neck Top [Top, DARK-GRAY]
- origin 2026-01 (forecast covers the next 4 months); item age 25 months; LLM highlights: fit_tightness 7, back_to_school 6, versatility 6, coverage 5
- R3 215 | model 63 (deviation -1.21) | actual 66 (deviation -1.17) | residual +0.04
- base value -0.11; level offset +0.00; other features sum -0.34
- top SHAP: cp_nb_sea4=-1.217 -> -0.213; own_sea4=-1.423 -> -0.191; f_r7=-0.705 -> -0.099; lagr1=-0.539 -> -0.092; t_org=3.968 -> -0.062; ptype=Top -> -0.061; lagr6=0.6 -> +0.059; lagr3=0.377 -> -0.046; peak_r=1.887 -> +0.039; x_summer=1.2 -> -0.036; bT_dT=-0.141 -> -0.034; x_winter=0.0 -> -0.032

## C065: 80 Oversized T-Shirt [T-Shirt, NAVY]
- origin 2025-07 (forecast covers the next 4 months); item age 23 months; LLM highlights: back_to_school 8, coverage 7, versatility 6, staple 5
- R3 1529 | model 3412 (deviation +0.80) | actual 2031 (deviation +0.28) | residual -0.52
- base value -0.17; level offset -0.00; other features sum +0.14
- top SHAP: x_bts=30.4 -> +0.247; lagr1=0.291 -> +0.113; lagr6=1.098 -> +0.093; peak_r=2.39 -> +0.085; f_r28=0.312 -> +0.082; cp_nb_sea4=0.702 -> -0.080; lvl=5.945 -> +0.074; f_r7=0.409 -> +0.057; lagr5=0.932 -> +0.054; t_org=21.189 -> +0.039; ptype=T-Shirt -> +0.039; ev_spring_break_t=0 -> +0.032

## C066: Gretta Striped Ribbed Tank Top [Tank Top, NAVY-AND-WHITE]
- origin 2025-08 (forecast covers the next 4 months); item age 18 months; LLM highlights: summer 8, fit_tightness 7, versatility 6, back_to_school 5
- R3 10636 | model 6910 (deviation -0.43) | actual 5538 (deviation -0.65) | residual -0.22
- base value -0.16; level offset -0.01; other features sum -0.05
- top SHAP: cp_nb_sea4=-0.298 -> -0.077; f_r7=-0.546 -> -0.068; bT_dT=0.266 -> +0.062; peak_r=1.303 -> +0.054; x_summer=0.0 -> -0.054; cp_nb_lvl=7.607 -> -0.048; lagr1=-0.098 -> -0.039; f_r28=-0.171 -> -0.036; x_winter=0.0 -> -0.032; statement=2 -> +0.031; lvl=7.872 -> +0.029; own_sea4=-0.721 -> -0.028

## C067: Striped Henley Tank Top [Tank Top, BLUE]
- origin 2025-10 (forecast covers the next 4 months); item age 13 months; LLM highlights: summer 8, fit_tightness 6, versatility 6, back_to_school 5
- R3 7141 | model 4408 (deviation -0.48) | actual 11094 (deviation +0.44) | residual +0.92
- base value -0.12; level offset -0.02; other features sum -0.04
- top SHAP: lagr1=-0.801 -> -0.156; cp_nb_sea4=-0.184 -> -0.081; f_r7=-1.615 -> -0.078; f_r28=-0.786 -> -0.071; bT_dT=0.233 -> +0.056; lagr6=-0.458 -> -0.056; lvl=7.47 -> +0.041; t_org=19.042 -> +0.040; ptype=Tank Top -> +0.037; lagr3=0.691 -> -0.035; x_summer=0.0 -> -0.035; a_fabric_feel=cotton / terry -> +0.032

## C068: Laila Textured Lacey Top [Top, CREAM]
- origin 2025-07 (forecast covers the next 4 months); item age 8 months; LLM highlights: coverage 6, fit_tightness 6, festival 6, versatility 5
- R3 207 | model 483 (deviation +0.85) | actual 418 (deviation +0.70) | residual -0.14
- base value -0.17; level offset -0.01; other features sum +0.13
- top SHAP: lagr6=1.621 -> +0.238; lagr5=1.61 -> +0.147; cp_nb_sea4=2.62 -> +0.116; peak_r=1.785 -> +0.110; f_r7=0.617 -> +0.077; cp_nb_lvl=5.203 -> +0.070; t_org=21.051 -> +0.046; lvl=3.955 -> -0.041; ev_spring_break_t=0 -> +0.039; ev_mothers_o=10 -> +0.036; f_r28=0.098 -> +0.032; t_tgt_norm=16.069 -> +0.025

## C069: 09 Raglan Long Sleeve T Shirt [T-Shirt, PINK]
- origin 2025-09 (forecast covers the next 4 months); item age 12 months; LLM highlights: back_to_school 7, coverage 6, versatility 6, fit_tightness 6
- R3 679 | model 652 (deviation -0.04) | actual 828 (deviation +0.20) | residual +0.24
- base value -0.15; level offset -0.01; other features sum -0.06
- top SHAP: x_winter=9.6 -> +0.213; x_warm_dT=-6.348 -> +0.164; lagr1=-0.576 -> -0.130; f_r28=-0.77 -> -0.092; t_org=22.157 -> +0.073; bT_dT=0.216 -> +0.050; lagr6=-0.652 -> -0.047; f_r7=-0.652 -> -0.044; cp_nb_sea4=0.494 -> -0.039; cp_nb_lvl=5.574 -> +0.034; ptype=T-Shirt -> +0.027; lvl=5.129 -> -0.025

## C070: Bows 4 Days Low Rise Baggy Jeans [Jeans, LIGHT-BLUE]
- origin 2026-01 (forecast covers the next 4 months); item age 25 months; LLM highlights: coverage 9, back_to_school 6, statement 5, warmth 5
- R3 149 | model 53 (deviation -1.03) | actual 58 (deviation -0.94) | residual +0.10
- base value -0.11; level offset -0.01; other features sum -0.24
- top SHAP: cp_nb_sea4=-0.846 -> -0.176; own_sea4=-0.743 -> -0.173; lagr1=-0.383 -> -0.075; lagr3=0.533 -> -0.057; lvl=3.641 -> -0.048; x_summer=0.8 -> -0.044; f_r28=-0.342 -> -0.031; lagr6=-0.274 -> -0.028; x_winter=0.0 -> -0.028; ev_prom_o=0 -> -0.025; peak_r=1.506 -> +0.022; lagr5=-0.506 -> -0.021

## C071: Saphi Striped Oversized Knit Sweater [Sweater, GRAY-AND-WHITE]
- origin 2025-09 (forecast covers the next 4 months); item age 1 months; LLM highlights: back_to_school 8, warmth 6, winter 6, versatility 6
- R3 557 | model 4109 (deviation +2.00) | actual 38112 (deviation +4.22) | residual +2.23
- base value -0.15; level offset +0.80; other features sum +0.07
- top SHAP: x_warm_dT=-9.212 -> +0.342; lagr2=nan -> +0.221; x_winter=14.4 -> +0.172; lagr1=0.295 -> +0.121; f_r28=0.378 -> +0.094; lvl=5.741 -> +0.094; lagr3=nan -> +0.074; cp_nb_lvl=4.408 -> +0.062; t_org=22.905 -> +0.061; ev_spring_break_t=0 -> +0.035; cp_nb_sea4=nan -> +0.030; x_bts=0.0 -> -0.027

## C072: Contrast Satin Effect Mini Dress [Dress, CREAM]
- origin 2025-11 (forecast covers the next 4 months); item age 7 months; LLM highlights: fit_tightness 9, party 9, formality 7, statement 6
- R3 9896 | model 5788 (deviation -0.54) | actual 5561 (deviation -0.58) | residual -0.04
- base value -0.11; level offset +0.00; other features sum -0.14
- top SHAP: cp_nb_sea4=-0.049 -> -0.142; f_r7=0.959 -> +0.074; ev_spring_break_t=31 -> -0.064; lagr6=-0.515 -> -0.050; t_tgt_norm=4.463 -> -0.044; f_r28=0.203 -> +0.041; bT_dT=-0.107 -> -0.038; x_bts=0.0 -> -0.035; lagr1=0.14 -> +0.035; x_summer=0.0 -> -0.031; lfl_macro4=-0.253 -> -0.021; lagr5=-0.179 -> -0.020

## C073: Lex Satin Effect & Lace Corset [Corset, LIGHT-PINK]
- origin 2025-06 (forecast covers the next 4 months); item age 16 months; LLM highlights: fit_tightness 9, party 7, formality 6, statement 5
- R3 340 | model 383 (deviation +0.12) | actual 440 (deviation +0.26) | residual +0.14
- base value -0.15; level offset +0.00; other features sum +0.07
- top SHAP: ev_spring_break_t=0 -> +0.076; lvl=4.458 -> -0.072; ptype=Corset -> -0.070; t_tgt_norm=20.743 -> +0.058; lagr1=0.243 -> +0.057; cp_nb_sea4=0.264 -> -0.055; f_r7=0.421 -> +0.049; f_r28=0.288 -> +0.048; occ_last=136.0 -> +0.027; ev_prom_o=50 -> +0.027; ev_mothers_o=10 -> +0.024; x_halloween=9.3 -> +0.024

## C074: Wide Leg Contrast Foldover Pants [Pants, NAVY]
- origin 2025-11 (forecast covers the next 4 months); item age 13 months; LLM highlights: coverage 9, back_to_school 5, versatility 5, summer 4
- R3 7265 | model 6213 (deviation -0.16) | actual 5903 (deviation -0.21) | residual -0.05
- base value -0.11; level offset +0.00; other features sum -0.07
- top SHAP: lagr1=0.612 -> +0.168; f_r28=0.677 -> +0.103; cp_nb_sea4=0.226 -> -0.095; t_tgt_norm=3.339 -> -0.093; f_r7=1.591 -> +0.092; ev_spring_break_t=31 -> -0.084; lagr6=-1.375 -> -0.083; bT_dT=0.356 -> +0.079; ptype=Pants -> +0.051; cp_nb_lvl=7.433 -> -0.042; x_bts=0.0 -> -0.042; x_summer=0.0 -> -0.038

## C075: Julianna Ruffle Shorts [Shorts, BLACK]
- origin 2025-11 (forecast covers the next 4 months); item age 27 months; LLM highlights: summer 9, fit_tightness 8, festival 7, party 5
- R3 3617 | model 1415 (deviation -0.94) | actual 1382 (deviation -0.96) | residual -0.02
- base value -0.11; level offset +0.00; other features sum -0.27
- top SHAP: cp_nb_sea4=-0.378 -> -0.163; own_sea4=-0.767 -> -0.138; f_r7=0.824 -> +0.083; lagr6=-0.743 -> -0.074; lagr1=-0.403 -> -0.061; f_r28=-0.326 -> -0.054; bT_dT=-0.403 -> -0.045; x_summer=0.0 -> -0.044; cp_nb_lvl=6.924 -> -0.042; lvl=6.812 -> +0.032; ptype=Shorts -> -0.029; lagr5=-0.413 -> -0.028

## C076: Polka Dot Backless Chiffon Top [Top, RED-AND-WHITE]
- origin 2025-09 (forecast covers the next 4 months); item age 6 months; LLM highlights: summer 7, fit_tightness 6, party 6, formality 5
- R3 8196 | model 3112 (deviation -0.97) | actual 4050 (deviation -0.70) | residual +0.26
- base value -0.15; level offset -0.02; other features sum -0.14
- top SHAP: cp_nb_sea4=-1.07 -> -0.112; lagr6=-1.259 -> -0.098; lagr1=-0.516 -> -0.083; f_r28=-0.615 -> -0.082; f_r7=-0.709 -> -0.082; ptype=Top -> -0.054; cp_nb_lvl=7.158 -> -0.047; x_summer=0.0 -> -0.038; occ_next=137.0 -> +0.037; lagr3=0.323 -> -0.035; bT_dT=-0.124 -> -0.035; t_tgt_norm=7.386 -> -0.027

## C077: Puppy Love Embroidered Sweater [Sweater, BLACK]
- origin 2025-06 (forecast covers the next 4 months); item age 5 months; LLM highlights: coverage 8, back_to_school 7, fit_tightness 6, warmth 6
- R3 103 | model 188 (deviation +0.60) | actual 395 (deviation +1.34) | residual +0.74
- base value -0.15; level offset +0.00; other features sum +0.09
- top SHAP: x_bts=35.0 -> +0.212; peak_r=1.611 -> +0.077; ev_mothers_o=10 -> +0.076; lagr1=-0.514 -> -0.068; lagr5=1.132 -> +0.067; ev_spring_break_t=0 -> +0.066; ptype=Sweater -> +0.066; cp_nb_lvl=4.221 -> +0.060; t_tgt_norm=20.388 -> +0.059; f_r28=-0.644 -> -0.058; ev_prom_o=50 -> +0.053; ev_graduation_o=46 -> +0.048

## C078: Frayed Vintage Washed Denim Shorts [Short Jeans, BLUE-WASHED]
- origin 2026-02 (forecast covers the next 4 months); item age 30 months; LLM highlights: summer 10, festival 8, back_to_school 5, versatility 5
- R3 2227 | model 4414 (deviation +0.68) | actual 5926 (deviation +0.98) | residual +0.29
- base value -0.10; level offset +0.03; other features sum -0.01
- top SHAP: x_festival=20.0 -> +0.146; f_r7=0.665 -> +0.144; lagr1=0.269 -> +0.107; x_summer=10.0 -> +0.097; ptype=Short Jeans -> +0.087; own_sea4=1.595 -> +0.074; f_r28=0.353 -> +0.067; lvl=6.351 -> +0.055; cp_nb_sea4=0.593 -> -0.047; bT_dT=0.201 -> +0.044; peak_r=1.601 -> +0.032; lagr5=-0.576 -> -0.029

## C079: Seashells Graphic T Shirt [T-Shirt, WHITE]
- origin 2025-07 (forecast covers the next 4 months); item age 5 months; LLM highlights: back_to_school 7, summer 7, coverage 6, fit_tightness 6
- R3 6484 | model 6051 (deviation -0.07) | actual 2630 (deviation -0.90) | residual -0.83
- base value -0.17; level offset -0.00; other features sum -0.03
- top SHAP: x_bts=26.6 -> +0.224; ev_mothers_o=10 -> +0.078; lvl=7.387 -> +0.057; cp_nb_lvl=7.273 -> -0.046; lagr5=-1.036 -> -0.043; lagr1=-0.124 -> -0.037; x_summer=0.0 -> -0.035; t_org=21.281 -> +0.029; bT_dT=0.042 -> -0.025; f_r7=-0.133 -> -0.022; x_winter=0.0 -> -0.021; peak_r=0.235 -> -0.020

## C080: Studded Built In Bra Tank Top [Tank Top, BLACK]
- origin 2026-02 (forecast covers the next 4 months); item age 4 months; LLM highlights: fit_tightness 7, summer 7, party 6, statement 6
- R3 4281 | model 2799 (deviation -0.42) | actual 5364 (deviation +0.23) | residual +0.65
- base value -0.10; level offset +0.02; other features sum -0.15
- top SHAP: cp_nb_sea4=-1.306 -> -0.254; x_summer=7.0 -> +0.086; f_r7=0.306 -> +0.060; lvl=6.999 -> +0.043; cp_nb_lvl=7.1 -> -0.034; lagr1=-0.153 -> -0.022; x_bts=0.0 -> -0.021; x_winter=0.0 -> -0.020; winter=0 -> +0.019; t_org=3.508 -> -0.018; x_warm_dT=1.099 -> -0.016; lfl_macro4=-0.271 -> -0.016

## C081: Camelia Denim Mini Skort [Skort, LIGHT-BLUE]
- origin 2026-01 (forecast covers the next 4 months); item age 7 months; LLM highlights: fit_tightness 7, summer 7, party 6, statement 5
- R3 4888 | model 2313 (deviation -0.75) | actual 4936 (deviation +0.01) | residual +0.76
- base value -0.11; level offset -0.00; other features sum -0.18
- top SHAP: cp_nb_sea4=0.003 -> -0.143; lagr1=-0.584 -> -0.138; f_r7=-0.728 -> -0.093; bT_dT=0.216 -> +0.059; lagr3=0.327 -> -0.045; lagr6=-0.587 -> -0.043; f_r28=-0.639 -> -0.030; x_summer=2.8 -> +0.029; lvl=7.104 -> +0.027; x_winter=0.0 -> -0.027; cp_nb_lvl=7.037 -> -0.026; ptype=Skort -> -0.025

## C082: Wide Leg Foldover Pants [Pants, LIGHT-GRAY-MELANGE]
- origin 2025-11 (forecast covers the next 4 months); item age 17 months; LLM highlights: coverage 9, back_to_school 5, versatility 5, summer 4
- R3 13167 | model 11039 (deviation -0.18) | actual 11895 (deviation -0.10) | residual +0.07
- base value -0.11; level offset +0.00; other features sum -0.15
- top SHAP: lagr1=0.751 -> +0.218; f_r28=0.824 -> +0.130; lagr6=-2.189 -> -0.100; f_r7=1.828 -> +0.093; t_tgt_norm=3.3 -> -0.090; ev_spring_break_t=31 -> -0.090; cp_nb_sea4=0.223 -> -0.087; bT_dT=0.334 -> +0.083; ptype=Pants -> +0.053; lagr5=-1.34 -> -0.047; cp_nb_lvl=7.503 -> -0.045; x_summer=0.0 -> -0.037

## C083: Romie V Neck Cable Knit Sweater [Sweater, NAVY]
- origin 2026-01 (forecast covers the next 4 months); item age 27 months; LLM highlights: coverage 9, winter 8, back_to_school 8, versatility 7
- R3 1225 | model 288 (deviation -1.45) | actual 187 (deviation -1.88) | residual -0.43
- base value -0.11; level offset -0.01; other features sum -0.32
- top SHAP: lagr1=-0.944 -> -0.197; cp_nb_sea4=-1.43 -> -0.167; own_sea4=-1.986 -> -0.134; ptype=Sweater -> -0.101; f_r7=-0.691 -> -0.094; t_org=3.453 -> -0.078; lagr6=-0.552 -> -0.056; x_summer=0.4 -> -0.046; bT_dT=-0.154 -> -0.037; x_warm_dT=3.618 -> -0.034; ev_prom_o=0 -> -0.034; f_r28=-0.976 -> -0.033

## C084: Kris Ribbed Knit Sweatpants [Pants, BLUE]
- origin 2025-07 (forecast covers the next 4 months); item age 6 months; LLM highlights: coverage 10, back_to_school 7, staple 6, warmth 6
- R3 277 | model 485 (deviation +0.56) | actual 892 (deviation +1.17) | residual +0.61
- base value -0.17; level offset -0.01; other features sum +0.07
- top SHAP: x_bts=26.6 -> +0.179; lagr6=0.893 -> +0.112; ev_mothers_o=10 -> +0.084; ptype=Pants -> +0.075; cp_nb_lvl=4.856 -> +0.074; lvl=4.242 -> -0.061; t_org=21.312 -> +0.061; lagr5=0.839 -> +0.055; x_warm_dT=-2.538 -> +0.047; peak_r=0.893 -> +0.042; lagr1=-0.099 -> -0.034; cp_nb_sea4=nan -> +0.026

## C085: Michaela Off Shoulder Knit Top [Knitted Top, WHITE]
- origin 2025-10 (forecast covers the next 4 months); item age 7 months; LLM highlights: fit_tightness 7, coverage 6, warmth 5, versatility 5
- R3 11823 | model 14814 (deviation +0.23) | actual 6841 (deviation -0.55) | residual -0.77
- base value -0.12; level offset -0.01; other features sum +0.02
- top SHAP: x_winter=12.0 -> +0.217; x_warm_dT=-7.419 -> +0.206; cp_nb_sea4=2.905 -> +0.132; t_org=19.479 -> +0.072; f_r7=-1.298 -> -0.057; ptype=Knitted Top -> -0.057; bT_dT=-0.222 -> -0.046; cp_nb_lvl=7.216 -> -0.028; f_r28=-0.26 -> -0.026; x_summer=0.0 -> -0.024; x_bts=0.0 -> -0.024; lagr3=0.231 -> -0.023

## C086: Romey Beaded Chiffon Mini Dress [Dress, BLACK]
- origin 2026-02 (forecast covers the next 4 months); item age 11 months; LLM highlights: fit_tightness 10, party 10, statement 8, formality 7
- R3 1103 | model 841 (deviation -0.27) | actual 953 (deviation -0.15) | residual +0.12
- base value -0.10; level offset +0.03; other features sum -0.15
- top SHAP: cp_nb_sea4=0.21 -> -0.096; lagr1=-0.441 -> -0.089; bT_dT=0.28 -> +0.067; t_org=4.168 -> -0.059; f_r28=-0.358 -> -0.046; ptype=Dress -> +0.045; cp_nb_lvl=5.482 -> +0.041; f_r7=-0.008 -> +0.041; lagr3=0.528 -> -0.038; lagr6=0.469 -> +0.037; x_summer=4.0 -> +0.027; lvl=5.656 -> +0.023

## C087: Cowl Neck Backless Knit Mini Dress [Dress, TURQUOISE]
- origin 2026-02 (forecast covers the next 4 months); item age 9 months; LLM highlights: party 9, statement 8, fit_tightness 8, formality 7
- R3 305 | model 419 (deviation +0.31) | actual 369 (deviation +0.19) | residual -0.13
- base value -0.10; level offset +0.03; other features sum -0.11
- top SHAP: f_r7=0.678 -> +0.128; cp_nb_sea4=1.605 -> +0.112; lagr6=0.951 -> +0.092; lagr1=0.204 -> +0.073; ev_spring_break_t=31 -> -0.061; f_r28=0.286 -> +0.059; ptype=Dress -> +0.044; bT_dT=0.391 -> +0.039; lagr5=0.907 -> +0.039; cp_nb_lvl=5.747 -> +0.035; lvl=4.381 -> -0.033; t_org=4.178 -> -0.029

## C088: Contrast Satin Effect Mini Dress [Dress, CREAM]
- origin 2025-08 (forecast covers the next 4 months); item age 4 months; LLM highlights: fit_tightness 9, party 9, formality 7, statement 6
- R3 7712 | model 9268 (deviation +0.18) | actual 9449 (deviation +0.20) | residual +0.02
- base value -0.16; level offset -0.01; other features sum +0.08
- top SHAP: occ_next=31.0 -> +0.099; cp_nb_lvl=5.553 -> +0.062; x_winter=4.8 -> +0.046; occ_last=334.0 -> +0.044; lagr1=0.149 -> +0.042; f_r28=0.139 -> +0.039; ptype=Dress -> -0.039; ev_spring_break_t=0 -> +0.035; t_org=23.605 -> +0.030; lagr4=-1.152 -> -0.028; bT_dT=-0.364 -> -0.027; x_bts=0.0 -> -0.024

## C089: Eve Off Shoulder Knit Mini Dress [Dress, BLACK]
- origin 2026-01 (forecast covers the next 4 months); item age 16 months; LLM highlights: fit_tightness 8, party 6, winter 5, formality 5
- R3 7577 | model 2500 (deviation -1.11) | actual 2520 (deviation -1.10) | residual +0.01
- base value -0.11; level offset -0.00; other features sum -0.29
- top SHAP: own_sea4=-1.145 -> -0.184; cp_nb_sea4=-1.167 -> -0.174; lagr1=-0.444 -> -0.083; lagr6=-2.172 -> -0.079; f_r7=-0.343 -> -0.070; occ_next=14.0 -> +0.056; lagr5=-1.403 -> -0.032; ev_prom_o=0 -> -0.031; x_summer=1.2 -> -0.030; x_winter=0.0 -> -0.030; f_r28=-0.448 -> -0.025; lagr3=0.267 -> -0.025

## C090: Roman Low Rise Baggy Jeans [Jeans, WHITE]
- origin 2025-07 (forecast covers the next 4 months); item age 22 months; LLM highlights: coverage 9, back_to_school 7, versatility 7, warmth 5
- R3 18596 | model 27555 (deviation +0.39) | actual 24572 (deviation +0.28) | residual -0.11
- base value -0.17; level offset -0.00; other features sum +0.06
- top SHAP: x_bts=26.6 -> +0.204; f_r28=0.246 -> +0.084; f_r7=0.534 -> +0.075; ptype=Jeans -> +0.070; cp_nb_sea4=0.712 -> -0.064; lagr1=0.228 -> +0.054; ev_spring_break_t=0 -> +0.048; lagr6=-0.438 -> -0.041; statement=2 -> +0.039; versatility=7 -> +0.034; ev_mothers_o=10 -> +0.031; x_summer=0.0 -> -0.029

## C091: Romey Beaded Chiffon Mini Dress [Dress, PINK]
- origin 2025-09 (forecast covers the next 4 months); item age 4 months; LLM highlights: fit_tightness 10, party 10, statement 8, formality 7
- R3 2052 | model 2201 (deviation +0.07) | actual 1328 (deviation -0.43) | residual -0.50
- base value -0.15; level offset -0.00; other features sum +0.00
- top SHAP: lagr1=0.514 -> +0.139; f_r28=0.499 -> +0.093; lvl=6.239 -> +0.046; occ_last=228.0 -> +0.046; ptype=Dress -> -0.035; x_bts=0.0 -> -0.034; f_r7=0.146 -> +0.032; lagr3=-0.796 -> +0.029; t_tgt_norm=7.612 -> -0.027; cp_nb_sea4=0.756 -> -0.025; x_summer=0.0 -> -0.023; bT_dT=-0.294 -> -0.022

## C092: Wilde Lace Up Faux Leather Corset [Corset, BLACK]
- origin 2025-12 (forecast covers the next 4 months); item age 22 months; LLM highlights: fit_tightness 9, party 7, halloween 7, formality 6
- R3 193 | model 84 (deviation -0.83) | actual 84 (deviation -0.83) | residual +0.00
- base value -0.11; level offset -0.00; other features sum -0.20
- top SHAP: cp_nb_sea4=-0.039 -> -0.157; lagr1=-0.717 -> -0.132; own_sea4=-0.616 -> -0.085; f_r7=-0.447 -> -0.078; f_r28=-0.82 -> -0.044; beta_T=0.049 -> +0.036; x_summer=0.0 -> -0.035; ev_prom_o=0 -> -0.031; ptype=Corset -> +0.029; lagr3=0.669 -> -0.028; peak_r=1.547 -> +0.028; lvl=3.895 -> -0.027

## C093: French Terry Straight Leg Sweatpants [Sweatpants, BLUE]
- origin 2025-10 (forecast covers the next 4 months); item age 9 months; LLM highlights: coverage 10, staple 8, back_to_school 8, warmth 6
- R3 291 | model 374 (deviation +0.25) | actual 393 (deviation +0.30) | residual +0.05
- base value -0.12; level offset -0.01; other features sum -0.02
- top SHAP: x_warm_dT=-9.282 -> +0.305; x_winter=12.0 -> +0.191; lvl=4.289 -> -0.099; bT_dT=0.444 -> +0.085; f_r7=-0.99 -> -0.063; t_org=18.783 -> +0.056; lagr3=0.411 -> -0.043; lagr5=-0.626 -> -0.028; lagr6=-0.439 -> -0.026; tmonth=11 -> +0.026; staple=8 -> +0.025; f_r28=-0.2 -> -0.024

## C094: Sequin Fringe Knit Poncho [Knitted Top, PINK]
- origin 2025-11 (forecast covers the next 4 months); item age 7 months; LLM highlights: statement 8, party 8, festival 8, formality 6
- R3 2985 | model 2596 (deviation -0.14) | actual 2750 (deviation -0.08) | residual +0.06
- base value -0.11; level offset +0.00; other features sum -0.27
- top SHAP: ptype=Knitted Top -> -0.110; beta_T=0.062 -> +0.098; lagr5=1.341 -> +0.085; ev_spring_break_t=31 -> -0.072; occ_next=31.0 -> +0.070; occ_last=289.0 -> +0.070; lagr1=-0.351 -> -0.064; peak_r=1.341 -> +0.058; f_r7=0.46 -> +0.054; x_warm_dT=-2.916 -> +0.046; cp_nb_lvl=6.611 -> -0.042; lagr6=0.506 -> +0.041

## C095: Tez Vintage Washed Low Rise Bootcut Jeans [Jeans, BLUE]
- origin 2026-02 (forecast covers the next 4 months); item age 3 months; LLM highlights: coverage 9, back_to_school 8, versatility 7, winter 5
- R3 723 | model 329 (deviation -0.79) | actual 363 (deviation -0.69) | residual +0.10
- base value -0.10; level offset +0.04; other features sum -0.04
- top SHAP: cp_nb_sea4=-1.693 -> -0.192; x_warm_dT=5.54 -> -0.098; lagr1=-0.363 -> -0.090; t_org=2.869 -> -0.078; f_r7=-0.464 -> -0.072; f_r28=-0.28 -> -0.045; ptype=Jeans -> +0.029; ev_spring_break_t=31 -> -0.028; x_bts=0.0 -> -0.027; x_winter=0.0 -> -0.027; lagr3=0.221 -> -0.025; ev_prom_o=0 -> -0.024

## C096: Lorenzo Polka Dot T Shirt [T-Shirt, LIGHT-PINK]
- origin 2025-12 (forecast covers the next 4 months); item age 6 months; LLM highlights: fit_tightness 7, summer 6, back_to_school 6, versatility 5
- R3 1424 | model 790 (deviation -0.59) | actual 1393 (deviation -0.02) | residual +0.57
- base value -0.11; level offset -0.00; other features sum -0.24
- top SHAP: f_r7=-0.614 -> -0.069; lagr5=0.806 -> +0.058; ptype=T-Shirt -> -0.051; x_summer=0.0 -> -0.043; f_r28=-0.302 -> -0.042; cp_nb_sea4=nan -> +0.041; lfl_type4=-0.638 -> -0.036; ev_prom_o=0 -> -0.033; bT_dT=-0.073 -> -0.031; x_bts=0.0 -> -0.030; tmonth=1 -> -0.029; lvl=5.874 -> +0.023

## C097: Glendon Babydoll Top [Tank Top, BURGUNDY]
- origin 2026-01 (forecast covers the next 4 months); item age 5 months; LLM highlights: fit_tightness 6, summer 6, party 5, coverage 5
- R3 15411 | model 5346 (deviation -1.06) | actual 10234 (deviation -0.41) | residual +0.65
- base value -0.11; level offset +0.00; other features sum -0.17
- top SHAP: lagr1=-0.929 -> -0.215; cp_nb_sea4=-0.94 -> -0.196; f_r7=-1.2 -> -0.111; lagr6=-2.389 -> -0.078; lagr5=-1.117 -> -0.057; lagr3=0.398 -> -0.056; t_org=4.261 -> -0.034; f_r28=-1.04 -> -0.033; cp_nb_lvl=7.645 -> -0.032; ptype=Tank Top -> +0.030; x_winter=0.0 -> -0.029; a_fabric_feel=cotton / terry -> +0.027

## C098: Layered Bra Cowl Neck Top [Tank Top, GREEN]
- origin 2025-12 (forecast covers the next 4 months); item age 12 months; LLM highlights: fit_tightness 8, party 8, summer 6, formality 6
- R3 4449 | model 3204 (deviation -0.33) | actual 2747 (deviation -0.48) | residual -0.15
- base value -0.11; level offset -0.01; other features sum -0.16
- top SHAP: cp_nb_sea4=-0.191 -> -0.186; f_r7=0.45 -> +0.083; cp_nb_lvl=7.287 -> -0.075; occ_next=76.0 -> +0.069; occ_last=289.0 -> +0.059; peak_r=1.521 -> +0.048; lagr6=0.711 -> +0.041; tmonth=1 -> -0.039; x_summer=0.0 -> -0.032; lfl_type4=-0.495 -> -0.032; lagr1=0.145 -> +0.031; ev_spring_break_t=31 -> -0.022

## C099: Aisha Low Rise Mini Skirt [Skirt, WHITE]
- origin 2025-08 (forecast covers the next 4 months); item age 43 months; LLM highlights: summer 7, fit_tightness 6, versatility 5, party 5
- R3 7679 | model 6739 (deviation -0.13) | actual 8467 (deviation +0.10) | residual +0.23
- base value -0.16; level offset -0.00; other features sum +0.02
- top SHAP: occ_last=304.0 -> +0.107; occ_next=61.0 -> +0.103; cp_nb_sea4=0.096 -> -0.067; cp_nb_lvl=6.926 -> -0.052; lagr6=-0.422 -> -0.042; lagr1=0.113 -> +0.039; ptype=Skirt -> -0.036; f_r28=0.081 -> +0.033; bT_dT=-0.14 -> -0.033; x_summer=0.0 -> -0.030; lagr5=-0.806 -> -0.029; statement=2 -> +0.029

## C100: Fabia Asymmetric Textured Lace Top [Top, LIGHT-BLUE]
- origin 2026-02 (forecast covers the next 4 months); item age 15 months; LLM highlights: fit_tightness 6, party 6, statement 5, formality 5
- R3 56 | model 21 (deviation -0.94) | actual 36 (deviation -0.43) | residual +0.51
- base value -0.10; level offset +0.00; other features sum -0.23
- top SHAP: cp_nb_sea4=-0.994 -> -0.246; own_sea4=-0.993 -> -0.207; t_org=2.403 -> -0.068; lvl=2.711 -> -0.061; cp_nb_lvl=4.644 -> +0.059; lagr6=-0.514 -> -0.054; x_warm_dT=3.45 -> -0.044; ptype=Top -> -0.036; lagr5=0.754 -> +0.027; x_summer=4.0 -> +0.025; peak_r=1.551 -> +0.024; ev_spring_break_t=31 -> -0.023

## C101: Frayed Vintage Washed Denim Shorts [Short Jeans, BLUE-WASHED]
- origin 2025-11 (forecast covers the next 4 months); item age 27 months; LLM highlights: summer 10, festival 8, back_to_school 5, versatility 5
- R3 2227 | model 3408 (deviation +0.43) | actual 3043 (deviation +0.31) | residual -0.11
- base value -0.11; level offset +0.00; other features sum -0.09
- top SHAP: lagr5=1.624 -> +0.215; cp_nb_sea4=-0.162 -> -0.173; lagr6=1.492 -> +0.141; f_r7=1.081 -> +0.097; ptype=Short Jeans -> +0.089; lagr1=0.225 -> +0.068; f_r28=0.294 -> +0.054; peak_r=1.624 -> +0.051; trend91=-1.554 -> +0.041; lagr4=1.566 -> +0.038; cp_nb_lvl=7.003 -> -0.037; lvl=6.327 -> +0.036

## C102: Moon Sign Faux Leather Mini Dress [Dress, BLACK]
- origin 2026-01 (forecast covers the next 4 months); item age 27 months; LLM highlights: fit_tightness 9, party 9, halloween 6, formality 6
- R3 77 | model 50 (deviation -0.43) | actual 38 (deviation -0.70) | residual -0.27
- base value -0.11; level offset +0.00; other features sum -0.18
- top SHAP: cp_nb_sea4=0.008 -> -0.151; cp_nb_lvl=4.802 -> +0.080; ptype=Dress -> +0.074; f_r7=0.284 -> +0.062; own_sea4=-0.398 -> -0.053; t_org=3.132 -> -0.047; lvl=3.016 -> -0.038; lagr6=-0.243 -> -0.028; bT_dT=-0.179 -> -0.025; ev_spring_break_t=31 -> -0.023; lagr1=-0.02 -> +0.023; x_winter=0.0 -> -0.021

## C103: Studded Foldover Micro Shorts [Shorts, BLACK]
- origin 2025-10 (forecast covers the next 4 months); item age 9 months; LLM highlights: fit_tightness 8, festival 8, summer 7, party 7
- R3 6353 | model 3431 (deviation -0.62) | actual 3843 (deviation -0.50) | residual +0.11
- base value -0.12; level offset -0.01; other features sum -0.12
- top SHAP: cp_nb_sea4=-0.188 -> -0.089; f_r7=-0.774 -> -0.074; bT_dT=-0.642 -> -0.048; x_summer=0.0 -> -0.045; lvl=7.364 -> +0.034; t_tgt_norm=5.849 -> -0.034; x_bts=0.0 -> -0.032; cp_nb_lvl=7.276 -> -0.029; t_org=20.059 -> +0.029; ptype=Shorts -> -0.027; lagr6=-0.304 -> -0.026; x_winter=0.0 -> -0.023

## C104: Reeva Polka Dot Foldover Pants [Pants, BLACK]
- origin 2026-01 (forecast covers the next 4 months); item age 7 months; LLM highlights: coverage 10, summer 5, versatility 5, back_to_school 4
- R3 41704 | model 14049 (deviation -1.09) | actual 10365 (deviation -1.39) | residual -0.30
- base value -0.11; level offset +0.00; other features sum -0.30
- top SHAP: lagr1=-0.554 -> -0.147; f_r7=-0.768 -> -0.110; lagr6=-1.974 -> -0.078; t_org=3.949 -> -0.062; lvl=9.254 -> -0.062; cp_nb_lvl=8.347 -> -0.060; lagr5=-1.48 -> -0.053; cp_nb_sea4=0.331 -> -0.043; bT_dT=-0.081 -> -0.040; ptype=Pants -> +0.037; x_winter=0.0 -> -0.033; f_r28=-0.575 -> -0.031

## C105: Marcia Chiffon Lacey Mini Dress [Dress, BLACK]
- origin 2025-11 (forecast covers the next 4 months); item age 5 months; LLM highlights: party 8, formality 6, statement 5, halloween 5
- R3 4592 | model 3092 (deviation -0.40) | actual 2730 (deviation -0.52) | residual -0.12
- base value -0.11; level offset +0.00; other features sum -0.18
- top SHAP: cp_nb_sea4=-0.072 -> -0.147; occ_next=76.0 -> +0.093; f_r7=0.67 -> +0.070; bT_dT=-0.397 -> -0.045; lagr5=-1.33 -> -0.040; t_tgt_norm=5.492 -> -0.033; lagr1=0.062 -> +0.031; x_bts=0.0 -> -0.031; ev_spring_break_t=31 -> -0.030; x_summer=0.0 -> -0.029; ptype=Dress -> +0.028; winter=1 -> +0.026

## C106: Striped Henley Tank Top [Tank Top, BLUE]
- origin 2025-07 (forecast covers the next 4 months); item age 10 months; LLM highlights: summer 8, fit_tightness 6, versatility 6, back_to_school 5
- R3 7277 | model 6987 (deviation -0.04) | actual 9523 (deviation +0.27) | residual +0.31
- base value -0.17; level offset +0.00; other features sum -0.03
- top SHAP: lagr1=0.481 -> +0.153; f_r28=0.513 -> +0.113; lagr6=-1.462 -> -0.109; f_r7=0.655 -> +0.102; x_bts=19.0 -> +0.063; lagr5=-1.213 -> -0.057; cp_nb_sea4=0.388 -> -0.057; x_summer=0.0 -> -0.042; trend91=1.236 -> -0.041; lvl=7.507 -> +0.037; cp_nb_lvl=7.229 -> -0.034; ev_spring_break_t=0 -> +0.033

## C107: Kait Pants [Pants, LIGHT-BLUE]
- origin 2025-07 (forecast covers the next 4 months); item age 7 months; LLM highlights: coverage 9, versatility 6, back_to_school 5, staple 5
- R3 67 | model 114 (deviation +0.53) | actual 70 (deviation +0.05) | residual -0.49
- base value -0.17; level offset -0.02; other features sum +0.04
- top SHAP: lagr6=0.951 -> +0.124; ptype=Pants -> +0.093; peak_r=1.187 -> +0.068; cp_nb_lvl=5.114 -> +0.064; f_r28=0.168 -> +0.061; x_bts=19.0 -> +0.060; f_r7=0.443 -> +0.054; t_org=21.104 -> +0.040; ev_spring_break_t=0 -> +0.035; lagr5=0.699 -> +0.029; statement=1 -> +0.028; ev_mothers_o=10 -> +0.027

## C108: Gaia Gathered Mini Dress [Dress, BLUE-AND-WHITE]
- origin 2026-02 (forecast covers the next 4 months); item age 33 months; LLM highlights: summer 9, fit_tightness 8, staple 4, versatility 4
- R3 1475 | model 1553 (deviation +0.05) | actual 1960 (deviation +0.28) | residual +0.23
- base value -0.10; level offset +0.02; other features sum -0.10
- top SHAP: cp_nb_sea4=0.189 -> -0.137; cp_nb_lvl=4.772 -> +0.104; f_r7=0.588 -> +0.103; x_summer=9.0 -> +0.061; t_org=1.937 -> -0.060; peak_r=2.39 -> +0.060; lvl=5.933 -> +0.050; ptype=Dress -> +0.049; lagr1=0.119 -> +0.043; own_sea4=-0.168 -> -0.036; f_r28=0.202 -> +0.027; bT_dT=-0.07 -> -0.025

## C109: Studded Faux Suede Micro Shorts [Shorts, BROWN]
- origin 2025-09 (forecast covers the next 4 months); item age 11 months; LLM highlights: statement 7, party 7, festival 6, halloween 5
- R3 3895 | model 5215 (deviation +0.29) | actual 4819 (deviation +0.21) | residual -0.08
- base value -0.15; level offset -0.00; other features sum -0.09
- top SHAP: lagr1=0.273 -> +0.136; x_halloween=15.5 -> +0.110; f_r7=0.495 -> +0.086; f_r28=0.27 -> +0.082; occ_next=31.0 -> +0.070; cp_nb_lvl=7.018 -> -0.060; occ_last=334.0 -> +0.053; lvl=6.881 -> +0.050; bT_dT=-0.749 -> -0.046; beta_T=0.05 -> +0.046; ptype=Shorts -> +0.031; lagr5=-0.679 -> -0.027

## C110: Sheer Chiffon Asymmetric Poncho [Top, BLACK]
- origin 2025-11 (forecast covers the next 4 months); item age 3 months; LLM highlights: statement 5, formality 5, party 5, festival 5
- R3 3527 | model 2747 (deviation -0.25) | actual 5102 (deviation +0.37) | residual +0.62
- base value -0.11; level offset +0.00; other features sum -0.12
- top SHAP: f_r7=0.845 -> +0.094; occ_next=76.0 -> +0.092; cp_nb_lvl=6.753 -> -0.084; occ_last=289.0 -> +0.079; ev_spring_break_t=31 -> -0.063; ptype=Top -> -0.060; f_r28=0.257 -> +0.052; bT_dT=-0.421 -> -0.051; x_bts=0.0 -> -0.045; t_tgt_norm=5.562 -> -0.038; x_summer=0.0 -> -0.036; lagr1=0.204 -> +0.035

## C111: Glennon Striped Shorts [Shorts, NAVY]
- origin 2025-12 (forecast covers the next 4 months); item age 5 months; LLM highlights: summer 8, versatility 5, staple 4, fit_tightness 4
- R3 881 | model 475 (deviation -0.62) | actual 828 (deviation -0.06) | residual +0.55
- base value -0.11; level offset -0.01; other features sum -0.14
- top SHAP: cp_nb_sea4=0.327 -> -0.091; lagr5=0.836 -> +0.054; f_r28=-0.531 -> -0.053; x_summer=0.0 -> -0.048; lagr1=-0.237 -> -0.037; ptype=Shorts -> -0.037; bT_dT=-0.112 -> -0.029; ev_prom_o=0 -> -0.026; lfl_type4=-0.426 -> -0.024; tmonth=1 -> -0.023; x_bts=0.0 -> -0.023; t_tgt_norm=6.974 -> -0.019

## C112: Basel Button Up Tank Top [Tank Top, RED]
- origin 2025-06 (forecast covers the next 4 months); item age 7 months; LLM highlights: summer 8, fit_tightness 6, versatility 5, coverage 5
- R3 6148 | model 6736 (deviation +0.09) | actual 8227 (deviation +0.29) | residual +0.20
- base value -0.15; level offset +0.00; other features sum +0.04
- top SHAP: lagr6=-0.788 -> -0.082; lvl=7.342 -> +0.075; t_tgt_norm=20.761 -> +0.055; t_org=17.585 -> +0.037; x_summer=3.2 -> +0.037; x_bts=20.0 -> +0.036; lagr5=-0.661 -> -0.036; winter=0 -> -0.034; ptype=Tank Top -> +0.033; ev_spring_break_t=0 -> +0.030; lagr1=0.096 -> +0.024; statement=2 -> +0.024

## C113: Nikki Nylon Shorts [Shorts, BLUE]
- origin 2026-01 (forecast covers the next 4 months); item age 19 months; LLM highlights: summer 9, fit_tightness 5, back_to_school 5, versatility 5
- R3 10205 | model 5029 (deviation -0.71) | actual 9663 (deviation -0.05) | residual +0.65
- base value -0.11; level offset +0.00; other features sum -0.16
- top SHAP: lagr1=-0.955 -> -0.211; cp_nb_sea4=0.018 -> -0.164; lagr6=1.217 -> +0.113; f_r7=-0.815 -> -0.101; lagr3=0.508 -> -0.046; t_org=3.992 -> -0.042; x_summer=3.6 -> +0.036; f_r28=-0.984 -> -0.030; statement=2 -> +0.030; x_winter=0.0 -> -0.026; cp_nb_lvl=6.707 -> -0.024; own_sea4=0.283 -> +0.023

## C114: Cecily Strapless Knit Top [Strapless, CREAM]
- origin 2025-06 (forecast covers the next 4 months); item age 15 months; LLM highlights: fit_tightness 9, summer 7, versatility 5, staple 4
- R3 41 | model 30 (deviation -0.32) | actual 30 (deviation -0.31) | residual +0.01
- base value -0.15; level offset +0.00; other features sum -0.03
- top SHAP: lagr1=-1.045 -> -0.156; cp_nb_sea4=0.958 -> -0.082; ev_mothers_o=10 -> +0.075; peak_r=2.102 -> +0.066; lagr6=-0.485 -> -0.064; f_r28=-0.981 -> -0.062; ev_prom_o=50 -> +0.061; ptype=Strapless -> -0.060; ev_graduation_o=46 -> +0.046; f_r7=-0.754 -> -0.044; cp_nb_lvl=5.576 -> +0.039; t_org=17.868 -> +0.036

## C115: Layered Bra Cowl Neck Top [Tank Top, GREEN]
- origin 2026-02 (forecast covers the next 4 months); item age 14 months; LLM highlights: fit_tightness 8, party 8, summer 6, formality 6
- R3 2937 | model 2440 (deviation -0.19) | actual 3333 (deviation +0.13) | residual +0.31
- base value -0.10; level offset +0.04; other features sum -0.12
- top SHAP: cp_nb_sea4=0.114 -> -0.122; x_summer=6.0 -> +0.083; occ_next=17.0 -> +0.067; lagr1=-0.317 -> -0.058; f_r7=0.209 -> +0.057; f_r28=-0.234 -> -0.051; peak_r=1.888 -> +0.049; lagr3=0.512 -> -0.040; lvl=6.642 -> +0.035; cp_nb_lvl=6.618 -> -0.025; lagr6=0.535 -> +0.024; t_org=2.896 -> -0.024

## C116: Petal Princess Socks [Socks, CREAM]
- origin 2025-07 (forecast covers the next 4 months); item age 13 months; LLM highlights: coverage 7, fit_tightness 4, versatility 4, winter 4
- R3 432 | model 585 (deviation +0.30) | actual 443 (deviation +0.03) | residual -0.28
- base value -0.17; level offset -0.01; other features sum +0.06
- top SHAP: ptype=Socks -> +0.205; cp_nb_sea4=0.459 -> -0.089; cp_nb_lvl=4.921 -> +0.086; peak_r=1.397 -> +0.080; ev_mothers_o=10 -> +0.063; lagr5=0.92 -> +0.059; lagr6=0.499 -> +0.046; lvl=4.683 -> -0.044; t_org=21.322 -> +0.038; x_winter=0.0 -> -0.021; ev_spring_break_t=0 -> +0.020; x_summer=0.0 -> -0.020

## C117: Double Pleated Mini Skort [Skort, CREAM]
- origin 2025-09 (forecast covers the next 4 months); item age 15 months; LLM highlights: summer 7, back_to_school 6, versatility 5, coverage 4
- R3 6316 | model 3542 (deviation -0.58) | actual 5515 (deviation -0.14) | residual +0.44
- base value -0.15; level offset -0.00; other features sum -0.07
- top SHAP: f_r7=-0.356 -> -0.070; lagr6=-0.824 -> -0.066; occ_last=334.0 -> +0.065; occ_next=31.0 -> +0.055; bT_dT=-0.707 -> -0.054; f_r28=-0.316 -> -0.052; cp_nb_lvl=6.78 -> -0.046; ptype=Skort -> -0.043; x_summer=0.0 -> -0.039; lagr3=0.127 -> -0.037; lagr1=-0.262 -> -0.033; lagr5=-0.661 -> -0.030

## C118: Contrast Lacey Long Sleeve Top [Top, BLACK]
- origin 2026-02 (forecast covers the next 4 months); item age 6 months; LLM highlights: coverage 7, fit_tightness 7, party 6, formality 5
- R3 243 | model 69 (deviation -1.24) | actual 29 (deviation -2.09) | residual -0.85
- base value -0.10; level offset +0.05; other features sum -0.30
- top SHAP: cp_nb_sea4=-1.187 -> -0.210; lagr1=-0.617 -> -0.157; t_org=1.669 -> -0.110; f_r7=-1.048 -> -0.088; ptype=Top -> -0.067; bT_dT=-0.474 -> -0.055; x_warm_dT=3.423 -> -0.052; f_r28=-0.536 -> -0.048; x_winter=0.0 -> -0.029; lvl=4.173 -> -0.028; ev_spring_break_t=31 -> -0.023; x_bts=0.0 -> -0.023

## C119: Wide Leg Contrast Foldover Pants [Pants, NAVY]
- origin 2025-07 (forecast covers the next 4 months); item age 9 months; LLM highlights: coverage 9, back_to_school 5, versatility 5, summer 4
- R3 2839 | model 6253 (deviation +0.79) | actual 7094 (deviation +0.92) | residual +0.13
- base value -0.17; level offset -0.00; other features sum +0.17
- top SHAP: lagr1=0.383 -> +0.137; f_r7=0.698 -> +0.105; lvl=6.563 -> +0.096; ptype=Pants -> +0.087; f_r28=0.403 -> +0.085; x_bts=19.0 -> +0.080; cp_nb_sea4=2.351 -> +0.069; t_org=21.246 -> +0.045; ev_spring_break_t=0 -> +0.045; lagr2=-0.429 -> +0.042; x_summer=0.0 -> -0.034; statement=2 -> +0.033
