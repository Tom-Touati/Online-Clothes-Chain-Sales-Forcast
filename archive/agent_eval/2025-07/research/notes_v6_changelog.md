- **v6 (2025-07-31 snapshot)**:
  1. *Site plan in absolute units/day* (`SITE_WEB_PD`, replaces `SITE_WEB_IDX` / `SITE_WEB`): Aug 13.5k, Sep
     12.3k, Oct 11.2k, **Nov 25.5k (Black Friday)**; actuals for past months so backtests use the real level.
     Evidence: the July miss (−35%) was all site level (web ×1.57 on June, back-to-school event from Jul 20 on
     2024's dates); v5's index had no November and would have put Nov web at ~0.65 × July instead of ~1.2.
  2. *Other channels = panel drift × other-channel plan* (`OTHER_PD`, `OTHER_PATH`), replacing ×0.9/month.
     Evidence: other +71%/day in July (Nordstrom Anniversary, TikTok, Pacsun), 2024 marketplaces ×1.5 in Nov;
     backtest 0.2745 → 0.2728.
  3. *Occasion index from a July base*: Halloween stronger (Oct 1.9 → 2.3, Aug 1.18, Nov 1.15; LY Oct ×3.7 vs
     segment, July-25 Halloween ×1.15 under), Nov entries for Homecoming / 4th of July / Graduation, New Year's
     Eve added (not on Homecoming items, `OCC_EXCLUDE`).
  4. `DRIFT_ADJ` (off, 1.0) and `ANCHOR_WIN` (2) added as tested switches. Unchanged after testing: base blend,
     ramp, K, SEAS_POW / SEAS_ANCHOR, RET_POW, POW off. Pop-up rule kept (NYR reopened in July; not foreseeable).
