from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load():
    with open(ROOT / "models" / "model.pkl", "rb") as f:
        return pickle.load(f)


def _predict(df: pd.DataFrame, bundle) -> np.ndarray:
    feat, _ = build(df, bundle["stats"]["weight_median"], bundle["stats"]["market_median"])
    return np.expm1(bundle["model"].predict(feat[bundle["features"]]))


def validation_predictions(bundle) -> None:
    val = pd.read_csv(DATA / "validation.csv")
    preds = _predict(val, bundle)
    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": np.round(preds, 2)})
    path = ROOT / "validation_predictions.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out):,} rows)")


def december_predictions(bundle) -> None:
    path = DATA / "december_chart_inputs.csv"
    dec = pd.read_csv(path)
    # scorer needs the same seven columns, in order, so we fill in place
    # the fixed-route rows don't carry lat/lon; use Lexington KY and Fort Wayne IN
    dec_feat = dec.copy()
    dec_feat["pickup_lat"] = 38.0406
    dec_feat["pickup_lon"] = -84.5037
    dec_feat["delivery_lat"] = 41.0793
    dec_feat["delivery_lon"] = -85.1394
    dec_feat["market_index"] = np.nan
    dec_feat["quote_signal"] = 2.05  # near the training median
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
