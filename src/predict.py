from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL, build

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load():
    with open(ROOT / "models" / "model.pkl", "rb") as f:
        return pickle.load(f)


def _numeric(X: pd.DataFrame, enc) -> np.ndarray:
    cats = X[CATEGORICAL].astype(str).fillna("NA")
    cat_arr = enc.transform(cats)
    num = X.drop(columns=CATEGORICAL).to_numpy(dtype=float)
    return np.hstack([cat_arr, num])


def _predict(df: pd.DataFrame, bundle) -> np.ndarray:
    s = bundle["stats"]
    feat, _ = build(df, s["weight_median"], s["market_median"], s.get("categories"))
    X = feat[bundle["features"]]
    X_num = _numeric(X, bundle["enc"])
    p_h = np.expm1(bundle["hgb"].predict(X))
    p_e = np.expm1(bundle["et"].predict(X_num))
    p_r = np.expm1(bundle["ridge"].predict(X_num))
    wh, we, wr = bundle["weights"]
    return wh * p_h + we * p_e + wr * p_r


def validation_predictions(bundle) -> None:
    val = pd.read_csv(DATA / "validation.csv")
    preds = _predict(val, bundle)
    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(preds, 2)})
    path = ROOT / "validation_predictions.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out):,} rows)")


def _project_daily(train: pd.DataFrame, col: str, dates: pd.Series) -> np.ndarray:
    """Project a daily series forward: linear trend from the last 60 days + a per-dayofweek
    residual pulled from the training tail. Keeps the trend and the weekly cycle both."""
    daily = train.groupby("date")[col].mean().reset_index().sort_values("date")
    tail = daily.tail(60).copy()
    tail["t"] = (tail["date"] - tail["date"].min()).dt.days
    slope, intercept = np.polyfit(tail["t"], tail[col], 1)
    tail["trend"] = slope * tail["t"] + intercept
    tail["resid"] = tail[col] - tail["trend"]
    tail["dow"] = tail["date"].dt.dayofweek
    dow_offset = tail.groupby("dow")["resid"].mean().to_dict()
    out = []
    for d in dates:
        t = (d - tail["date"].min()).days
        out.append(slope * t + intercept + dow_offset.get(d.dayofweek, 0.0))
    return np.array(out)


def december_predictions(bundle) -> None:
    path = DATA / "december_chart_inputs.csv"
    dec = pd.read_csv(path)
    train = pd.read_csv(DATA / "train_test.csv")
    train["date"] = pd.to_datetime(train["date"])

    lex = train[train["pickup"] == "Lexington"][["pickup_lat", "pickup_lon"]].iloc[0]
    fw = train[train["delivery"] == "Fort Wayne"][["delivery_lat", "delivery_lon"]].iloc[0]
    dec_feat = dec.copy()
    dec_feat["pickup_lat"] = lex["pickup_lat"]
    dec_feat["pickup_lon"] = lex["pickup_lon"]
    dec_feat["delivery_lat"] = fw["delivery_lat"]
    dec_feat["delivery_lon"] = fw["delivery_lon"]

    dates = pd.to_datetime(dec_feat["date"])
    dec_feat["market_index"] = _project_daily(train, "market_index", dates)
    dec_feat["quote_signal"] = _project_daily(train, "quote_signal", dates)

    preds = _predict(dec_feat, bundle)
    dec["predicted_rate"] = np.round(preds, 2)
    dec.to_csv(path, index=False)
    print(f"wrote {path}")


def main():
    bundle = _load()
    validation_predictions(bundle)
    december_predictions(bundle)


if __name__ == "__main__":
    main()
