# Item scoring rubric (D151)

Score each item from its text only (product name, type, colour and tagger attributes). Every field is an integer 0–10. Use the anchors so that scores are comparable across items and across scorers.

Judge only what the garment *is*. Do not use any knowledge of how well an item, style or trend sold, or how trends developed after early 2024. A score must not depend on hindsight.

| Field | 0 | 5 | 10 |
|---|---|---|---|
| warmth | swimwear, mesh, very light summer piece | transitional: denim, light knit, long sleeve tee | heavy winter: puffer, thick fleece, chunky knit, faux fur |
| coverage | very revealing: micro length, cut-outs, sheer, backless | moderate | full coverage |
| formality | loungewear, sleep, gym | casual everyday is 3; going out / night out is 6 | formal, gown, black tie |
| fit_tightness | very oversized or baggy | regular fit | bodycon or skin-tight |
| statement | plain basic | some detail | bold statement: sequins, loud print, unusual cut |
| staple | micro-trend piece, very of-the-moment | | timeless wardrobe staple: basic tee, straight jeans, plain hoodie |
| summer | not worn in summer | could be worn | core summer item: shorts, swim, sundress, tank |
| winter | not worn in winter | could be worn | core winter item: coats, sweaters, fleece, boots-weather pieces |
| back_to_school | unsuitable for campus or school | | core back-to-school: denim, basics, hoodies, everyday tops |
| party | unsuitable for parties | | core party / night out / holiday party / New Year's piece |
| halloween | no costume use | could be part of a costume | an obvious costume piece: corset, cat-suit, devil or angel look, character colour, faux leather or latex, lingerie-like |
| festival | no festival use | | core music-festival look: crochet, fringe, mesh, sequins, boho, micro shorts |
| layering | worn alone | | a layering piece: cardigan, shrug, open shirt, vest, jacket |
| versatility | one narrow use | | goes with many outfits and occasions |

## Output format

A CSV file with exactly this header and one row per input item, in input order:

```
style_color,warmth,coverage,formality,fit_tightness,statement,staple,summer,winter,back_to_school,party,halloween,festival,layering,versatility
```
