# Freight Rate Prediction — Report

## 1. How the training data was split and validated

### Why a time-based split

The training file covers January through October 2025. The scoring file covers November and December 2025 — in other words, the future. The question the model is really being judged on is "can it price a load one month from now?", not "can it price a random load from any month?".

If I had shuffled all the rows and done a normal train/test split, October rows would end up in the training folds and (hypothetically) November and December rows in the test folds. That would make the model look better than it really is, because a model that's seen rows from after a given date has an unfair advantage. A real dispatcher quoting next week's loads doesn't have that luxury.

So the split is strictly by date, never by random shuffle.

### The split

| Set | Date range | Rows | Role |
|---|---|---|---|
| Fit | 2025-01-01 to 2025-10-01 | ~42,900 | Model training |
| Holdout | 2025-10-02 to 2025-10-31 | ~4,600 | Honest check before submission |
| Scoring | 2025-11-01 to 2025-12-31 | 12,000 | Final submission |

The holdout is 30 days long on purpose — it matches the gap between training and the real scoring set. The model has to predict a month it has never seen, exactly like the real submission. A shorter holdout (say the last week of October) would be too easy; a longer one would waste training signal.

### What was decided using the holdout, and what wasn't

Three choices were made against the holdout:

- Which models to blend together.
- The blend weights.
- Tree-model hyperparameters (via an inner 10% check inside the fit set — the holdout was never touched during training).

Nothing was tuned against the scoring file. Once the holdout numbers looked right, both models were retrained on the full cleaned training set (47,494 rows after dropping 506 bad ones) before producing the submitted predictions.

### Holdout results

The submitted model is a blend: **75% gradient-boosted trees + 25% extra-trees**. Ridge regression was included in the search as a sanity check for whether the problem is linear — it got zero weight, so it isn't.

| Model | Mean error | Percent error |
|---|---|---|
| Gradient-boosted trees | $59.30 | 2.46% |
| Extra-trees | $77.07 | 3.27% |
| Ridge (sanity check) | $431.54 | 21.41% |
| **Blend (0.75 + 0.25)** | **$60.53** | **2.49%** |

The average posted rate in the training set is around $2,370, so a $60 mean error means the submitted predictions should be off by roughly 2.5% in dollars on genuinely unseen rows.

## 2. The fixed December chart

The scorer produces this chart from `data/december_chart_inputs.csv`. The route is fixed: **Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb**. Only the date changes across the 31 days of December.

![December chart](scorer_results/candidate_december.png)

### How the chart was built

Every input to the model except the date is identical across the 31 rows. If I fed those rows in as-is, the prediction would be the same number 31 times and the chart would be a flat line. The curve comes from how the date-sensitive inputs are set up.

Two things made the chart work:

1. **Dataset coordinates, not real geographic coordinates.** The latitude and longitude values in the dataset are not the real coordinates for each city — they're the dataset's own fixed values. For the fixed route I pull them straight from the training file so the distance-related features line up with what the model learned from. Using real Google-maps coordinates put December into the wrong part of the model and dropped the baseline price by around $40.
2. **Market and quote values projected per day.** Both `market_index` and `quote_signal` have a clear weekly pattern in training (higher Thursday and Friday, lower on weekends) and a gentle upward drift across autumn. Rather than feed the same monthly average to all 31 days, each date gets its own projected value: the average of the last 60 training days, plus the trend, plus the day-of-week offset from the same window.

### What the chart shows

- Baseline around **$845**, close to the training average for the Lexington → Fort Wayne lane ($856 across 32 historical loads).
- Clear weekly cycle — peaks Thursday / Friday, troughs on weekends.
- Gentle upward drift through the month, driven by the autumn market trend rather than any extrapolation by the model itself.
- Overall range $825 to $866 (spread about $41), in line with normal month-over-month variation on comparable lanes in training.
