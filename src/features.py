from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "haversine",
    "dayofweek",
    "dayofyear",
    "month",
    "week",
]
FEATURES = CATEGORICAL + NUMERIC


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build(df: pd.DataFrame, weight_median: float | None = None,
          market_median: float | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # negative/zero weights are garbage — drop them to NaN and fill with median
    out.loc[out["weight"] <= 0, "weight"] = np.nan
    if weight_median is None:
        weight_median = float(out["weight"].median())
    out["weight"] = out["weight"].fillna(weight_median)

    if market_median is None:
        market_median = float(out["market_index"].median())
    out["market_index"] = out["market_index"].fillna(market_median)

    out["haversine"] = _haversine(
        out["pickup_lat"], out["pickup_lon"],
        out["delivery_lat"], out["delivery_lon"],
    )
    out["dayofweek"] = out["date"].dt.dayofweek
    out["dayofyear"] = out["date"].dt.dayofyear
    out["month"] = out["date"].dt.month
    out["week"] = out["date"].dt.isocalendar().week.astype(int)

    for col in CATEGORICAL:
        out[col] = out[col].astype("category")

    stats = {"weight_median": weight_median, "market_median": market_median}
    return out, stats
