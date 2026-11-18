"""Assemble notes.md = v7 head + previous notes (renamed to history) + v7 changelog entry. Run from the folder."""
prev = open("prev/notes.md", encoding="utf-8").read()
head = open("research/notes_head.md", encoding="utf-8").read()
prev = prev.replace("# Forecast notes: Aug–Nov 2025 per style_color (snapshot 2025-07-31)",
                    "# History: v6 notes (snapshot 2025-07-31), kept for reference", 1)
v7 = """- **v7 (2025-08-31 snapshot)**:
  1. *Site plan re-based on August and extended to December* (`SITE_WEB_PD` Aug 15.64k actual, Sep 14.0k, Oct
     12.6k, Nov 28.5k, Dec 25.0k; `OTHER_PD` Aug 3.13k, Sep 3.1k, Oct 2.75k, Nov 4.2k, Dec 5.5k). Evidence: v6's
     Aug -11.9% was the site (web +16% vs plan: the event ran to Aug 20 and late August stayed at 1.6-1.75x LY);
     YoY 1.65 in the first regime-matched month; 2024's Sep-Dec path from August.
  2. *Milder seasonal amplitude*: `SEAS_POW` 0.9 -> 0.8. Evidence: WINTER over / SUMMER under in Aug at three
     origins (0.73-0.87 / 1.09-1.12); Jun->Aug-25 group moves about LY's^0.45-0.57.
  3. *Anchoring uses the level gap since January* (`ANCHOR_LONG`, geometric mean with the 2-month gap). Evidence:
     the 2-month gap flipped from "winter fell less" (spring) to "winter rose less" (summer) and would add WINTER
     x1.18 for Sep-Dec; the gap since January says TY winter is already relatively high (0.90). Backtest
     0.2754 -> 0.2762 (noise), WINTER balance at Jun / Jul origins 0.93 / 0.87 -> 0.95 / 0.92.
  4. `DRIFT_ADJ` 1.0 -> 1.01: oracle-site backtest bias -9% -> -7% at equal WAPE.
  5. *Occasion index from an August base, December added*: Halloween Aug 1.18 -> 1.1 (Aug-25 rel. 0.95),
     Homecoming Sep index 2.1 -> 2.6 (LY x2.94 from Aug), NYE Dec 1.8, 4th of July / Graduation Dec entries.
  6. Fix: anchoring switch tested `> 0`, now `!= 0` (negative a = momentum; tested and rejected). Rejected after
     testing: level-neutral anchoring, momentum, ANCHOR_WIN 1/3/6, SEAS_POW 0.7 / 0.6, RAMP1 1.6, W14 0.3, POW.
     Unchanged: retail rule (the new-store run-rate handled BOCA's Aug-7 opening, x0.979), base blend, ramp, K.
"""
open("notes.md", "w", encoding="utf-8").write(head + "\n" + prev.rstrip("\n") + "\n" + v7)
print("notes.md written")
