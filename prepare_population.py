"""Assign raters to the common discovery and confirmation partition."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_parquet(args.ratings)
    required = ["comment_id", "annotator_id", "hatespeech", "text", "reconstructed_gender"]
    if set(required) - set(raw):
        raise ValueError("Missing rating or demographic fields")
    if raw[["comment_id", "annotator_id"]].isna().any().any() or raw.duplicated(["annotator_id", "comment_id"]).any():
        raise ValueError("Missing or repeated rating keys")
    if (raw.groupby("comment_id").text.nunique(dropna=False) > 1).any():
        raise ValueError("An item has conflicting text")
    ratings = pd.to_numeric(raw.hatespeech, errors="raise")
    if not ratings.dropna().isin([0, 1, 2]).all():
        raise ValueError("Expected raw hate-speech scores 0, 1, 2")
    data = raw.loc[ratings.notna()].rename(columns={"annotator_id": "rater_id"})
    if (data.groupby("rater_id").reconstructed_gender.nunique(dropna=False) > 1).any():
        raise ValueError("A rater has conflicting demographic metadata")
    metadata = data[["rater_id", "reconstructed_gender"]].drop_duplicates().sort_values("rater_id").reset_index(drop=True)
    order = np.random.default_rng(20261008).permutation(len(metadata))
    phase = np.full(len(metadata), "confirmation", dtype=object)
    phase[order[:len(metadata) // 2]] = "discovery"
    metadata["phase"] = phase
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
