# Freight Rate Prediction — Report

## 1. Approach to validating the train/test split

### Why a time-based split is the only honest option here

The labeled file (`train_test.csv`) covers 2025-01-01 through 2025-10-31. The scoring file (`validation.csv`) covers 2025-11-01 through 2025-12-31 — **strictly in the future** relative to training. The real question the model is being evaluated on is *"can it price a load one month from now?"*, not *"can it price a load drawn from the same distribution?"*.

Random k-fold cross-validation would leak information backward in time: October rows would end up in training folds, and November and December rows (if we had them) would end up in holdout folds. That inflates the apparent score. A production dispatcher making next-week quotes doesn't have that luxury — they're always forecasting forward.

So the entire validation regime is built around **time**, not row shuffling.

### The split

| Set | Date range | Rows | Role |
|---|---|---|---|
| Fit | 2025-01-01 → 2025-10-01 | ~42,900 | Model training |
| Holdout | 2025-10-02 → 2025-10-31 | ~4,600 | Model and blend selection |
| Scoring | 2025-11-01 → 2025-12-31 | 12,000 | Final submission (`validation.csv`) |

The 30-day holdout window was chosen because it **mirrors the gap between training and the scoring set** — one month into the future, same lanes, same equipment, same seasonality boundary. Validating on a shorter window (say, the last week of October) would be optimistic; validating on the full last three months would eat too much of the training signal.

### What was tuned on the holdout, and what was not

Only three things were decided against the holdout:
- Which model families to include in the ensemble.
- The blend weights across those models.
- Hyperparameters for the two tree models (via early stopping on an inner 10% split of the fit set — never looking at the holdout during fitting).

Nothing was tuned against `validation.csv`. The submitted predictions use models retrained on the **entire** cleaned training set (47,494 rows after dropping 506 inconsistent rows for impossible road-to-haversine ratios and extreme rate-per-mile outliers).

### Holdout numbers

Ensemble: 75% `HistGradientBoostingRegressor` + 25% `ExtraTreesRegressor`. Ridge regression was included in the blend weight search as a non-linearity sanity check and received zero weight — confirming the problem is strongly non-linear.

| Model | MAE | MAPE |
|---|---|---|
| HistGradientBoosting (log target, absolute-error loss) | $59.30 | 2.46% |
| ExtraTrees | $77.07 | 3.27% |
| Ridge (sanity check) | $431.54 | 21.41% |
| **Ensemble (0.75 HGB + 0.25 ET)** | **$60.53** | **2.49%** |

Interpretation: the ensemble's mean absolute error of ~$60 against an average posted rate of ~$2,370 means the submitted predictions should be accurate to roughly 2.5% in dollar terms on genuinely out-of-sample rows.

## 2. Fixed December prediction chart

Produced by `score.py` from `data/december_chart_inputs.csv`. Fixed route: **Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb** — only the date changes across the 31 days of December.

![December chart](scorer_results/candidate_december.png)

### How the chart was built

Every input except the date is constant across the 31 rows. A tree model on identical inputs returns an identical prediction, so the fixed-route chart depends entirely on how the date-sensitive features — `dayofweek`, `market_index`, and `quote_signal` — are set up for the forecast window.

Two decisions matter for interpreting the chart:

1. **Dataset lat/lon, not real geographic coordinates.** The dataset's `pickup_lat`/`pickup_lon` for each city are the dataset's own fixed synthetic values, not Google-maps coords. For the fixed December route the lat/lon are pulled straight from `train_test.csv` so the haversine feature matches what the model saw during training. Using geographic coords put December in the wrong leaf and dropped the baseline by ~$40.
2. **`market_index` and `quote_signal` projected per day.** Both series exhibit a clear weekly cycle in training (Thu/Fri highs, weekend lows, ~0.15 swing) and a mild upward drift across autumn. Rather than feed the same averaged value to all 31 days, each date gets its own projection: last-60-day baseline + a linear trend extrapolated from the same window + a per-dayofweek residual.

### What the chart shows

- **Baseline ~$845**, close to the Lexington → Fort Wayne training average of $856 (n=32 historical loads on this exact lane).
- **Weekly cycle** — peaks on Thursdays and Fridays, troughs on weekends — consistent with the market_index pattern in the training tail.
- **Gentle upward drift** through the month, reflecting the autumn trend in the market features rather than any extrapolation by the tree model itself (trees can't extrapolate past training; the drift comes from the feature projection).
- **Range $825–$866**, spread ~$41 — in line with typical month-over-month rate variation observed on comparable lanes in training.
