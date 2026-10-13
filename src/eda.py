"""Quick look at train_test.csv — run once to sanity-check things."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    df = pd.read_csv(DATA / "train_test.csv")
    df["date"] = pd.to_datetime(df["date"])

    print(f"rows: {len(df):,}   cols: {len(df.columns)}")
    print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print()
    print("nulls per column:")
    print(df.isna().sum().to_string())
    print()

    # negative weights show up a few hundred times — treat as missing
    bad_weight = (df["weight"] <= 0).sum()
    print(f"non-positive weights: {bad_weight}")

    rpm = df["posted_rate"] / df["distance"]
    print()
    print("rate per mile by equipment:")
    print(rpm.groupby(df["equipment"]).describe()[["mean", "50%", "std"]])

    print()
    print("correlations with posted_rate:")
    print(df[["distance", "weight", "market_index", "quote_signal", "posted_rate"]]
          .corr()["posted_rate"].sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
