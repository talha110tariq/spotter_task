from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from .features import CATEGORICAL, FEATURES, build

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
TARGET = "posted_rate"


def time_split(df: pd.DataFrame, holdout_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def _fit(X: pd.DataFrame, y: np.ndarray, **kwargs) -> HistGradientBoostingRegressor:
    cat_mask = [c in CATEGORICAL for c in FEATURES]
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=1200,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=42,
        **kwargs,
    )
    model.fit(X, y)
    return model


def train():
    MODELS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "train_test.csv")
    df, stats = build(raw)

    train_df, val_df = time_split(df)
    print(f"train rows: {len(train_df):,}   holdout rows: {len(val_df):,}")

    y_train = np.log1p(train_df[TARGET].values)
    y_val = np.log1p(val_df[TARGET].values)

    model = _fit(train_df[FEATURES], y_train,
                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=30)

    pred = np.expm1(model.predict(val_df[FEATURES]))
    actual = val_df[TARGET].values
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    print(f"holdout MAE:  ${mae:,.2f}")
    print(f"holdout MAPE: {mape * 100:.2f}%")

    # final fit on everything
    full = _fit(df[FEATURES], np.log1p(df[TARGET].values))

    with open(MODELS / "model.pkl", "wb") as f:
        pickle.dump({"model": full, "stats": stats, "features": FEATURES,
                     "holdout_mae": mae, "holdout_mape": mape}, f)
    print(f"saved {MODELS / 'model.pkl'}")


if __name__ == "__main__":
    train()
