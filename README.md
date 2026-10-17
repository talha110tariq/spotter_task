# Freight Rate Prediction

Spotter ML assessment — predict `posted_rate` for truckload freight.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Train the model and write both prediction files:

```bash
python -m src.train
python -m src.predict
```

Then validate with the provided scorer:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

Outputs:
- `validation_predictions.csv` — 12,000 rows (`load_id,predicted_rate`)
- `data/december_chart_inputs.csv` — filled `predicted_rate` column
- `scorer_results/candidate_december.png` — fixed December chart
- `models/model.pkl` — trained model (gitignored)

## Approach

See `report.pdf` for the write-up. Short version: time-based holdout (last 30 days of training as validation), sklearn HistGradientBoosting on log-rate with city/equipment as native categoricals, cleaned negative/missing weights and imputed market index. Holdout MAE ≈ $101 / 4.9% MAPE.

## Layout

```
data/                raw CSVs (train_test, validation, templates)
src/                 features, training, prediction
score.py             provided scorer (untouched)
report.pdf           write-up with December chart
```
