# Judge brief (explainability step c)

You are one of several independent judges. For each item card, judge why the forecasting model's deviation from recency (R3) is what it is, and what the remaining miss most plausibly is.

**Read first:**
- `data/fk_categories.json`: the explanation categories. For each one, read its definition, evidence and sign notes, and its observations; `feature_map` tells you which features belong to which category.
- `data/fk_categoriser_context.md`: the feature dictionary, importances and the highest-gain trees.
- `data/fk_judge_cards.md`: 120 item cards.

**For each card,** using its SHAP values, its feature values, the item itself and the top trees:

1. **Explained categories.** For every category with `kind` = `explained`, give a signed integer from −10 to +10: how strongly, and in which direction, that category pushes this item's model deviation (− = below R3, + = above). Use 0 when the category plays no part. Judge the evidence; do not simply add SHAP numbers. Apply the categoriser's caveats. For example, event SHAP in a month with no event in the window is really a calendar effect: score it under the category that truly explains it.
2. **The residual.** The residual is the actual deviation minus the model's.
   - Give `residual_primary`: the one `unexplained` category id that most plausibly explains it, or `none` when |residual| < 0.15.
   - For every unexplained category, give a plausibility score from 0 to 10.
3. **Explanation.** One short sentence a buyer could read, in plain words.

**Work card by card.** Read each card and decide. Do not write or run any script that produces or adjusts scores. Work in chunks of about 20 cards, appending each chunk to your output file.

**Output:** a CSV with this exact header (the column order is the explained ids, then the unexplained ids, as they appear in `fk_categories.json`):

```
card,<explained id 1>,...,<explained id 11>,residual_primary,<unexplained id 1>,...,<unexplained id 5>,explanation
```

Quote the explanation field. Write one row per card, C000 to C119 in order.
