"""Attach documented research-copy gender codes to original MHS ratings.

The outcome and text always come from the 22-column raw-rating copy. The
four-group carrier supplies only a rater-level grouping, corroborated against
the separate small release retaining self-reported demographic fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GENDER_CODES = {1: "men", 2: "women", 3: "men", 4: "women"}


def attach(raw_path: Path, carrier_path: Path, evidence_path: Path, output: Path) -> dict:
    raw = pd.read_csv(raw_path)
    carrier = pd.read_csv(carrier_path)
    evidence = pd.read_csv(evidence_path)
    keys = ["comment_id", "annotator_id"]
    for name, frame in (("raw", raw), ("carrier", carrier), ("evidence", evidence)):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: missing or repeated original observation keys")
    if not carrier.social_group.isin(GENDER_CODES).all():
        raise ValueError("Carrier has an undocumented social_group code")
    if (carrier.groupby("annotator_id").social_group.nunique() > 1).any():
        raise ValueError("A carrier rater has inconsistent social-group codes")

    aligned = carrier.merge(raw[keys + ["text", "violence"]], on=keys,
                            how="left", validate="one_to_one", indicator=True,
                            suffixes=("_carrier", "_raw"))
    unmatched = int(aligned["_merge"].ne("both").sum())
    text_mismatch = int(aligned.text_carrier.ne(aligned.text_raw).sum())
    label_mismatch = int(aligned.label.ne(aligned.violence).sum())
    if unmatched or text_mismatch or label_mismatch:
        raise ValueError("Carrier is not exactly aligned with the original rating copy")

    observed = evidence.merge(carrier, on=keys, validate="one_to_one",
                              suffixes=("_evidence", "_carrier"))
    if set(observed.social_group) != set(GENDER_CODES):
        raise ValueError("The demographic evidence does not cover all four codes")
    expected = observed.social_group.map(GENDER_CODES)
    women = observed.annotator_gender_women.eq(True) & observed.annotator_gender_men.eq(False)
    men = observed.annotator_gender_men.eq(True) & observed.annotator_gender_women.eq(False)
    if not ((expected.eq("women") & women) | (expected.eq("men") & men)).all():
        raise ValueError("The carrier's gender coding disagrees with self-reported fields")
    if not observed.annotator_race_white.eq(observed.social_group.isin([1, 2])).all():
        raise ValueError("The carrier's white/non-white coding differs from its notebook")
    # The small demographic copy normalizes some whitespace. It is evidence for
    # metadata coding, never a replacement for the original text or outcomes.
    whitespace_equal = observed.text_evidence.map(lambda x: " ".join(x.split())).eq(
        observed.text_carrier.map(lambda x: " ".join(x.split())))
    if not whitespace_equal.all() or not observed.violence.eq(observed.label).all():
        raise ValueError("Demographic evidence is not aligned with the carrier")
    gold = observed.merge(raw[keys + ["hatespeech"]], on=keys,
                          validate="one_to_one", suffixes=("_evidence", "_raw"))
    if not gold.hatespeech_evidence.eq(gold.hatespeech_raw).all():
        raise ValueError("The small evidence copy has different raw hate-speech scores")

    metadata = carrier[["annotator_id", "social_group"]].drop_duplicates()
    metadata["reconstructed_gender"] = metadata.social_group.map(GENDER_CODES)
    joined = raw.merge(metadata, on="annotator_id", how="left", validate="many_to_one")
    if len(joined) != len(raw):
        raise RuntimeError("Metadata attachment changed the rating denominator")
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output, index=False)
    evidence_table = observed.groupby(
        ["social_group", "annotator_gender", "annotator_gender_men",
         "annotator_gender_women", "annotator_race_white"], dropna=False
    ).agg(rows=("comment_id", "size"), raters=("annotator_id", "nunique")).reset_index()
    evidence_table.to_csv(output.parent / "gender_code_evidence.csv", index=False)
    summary = {
        "raw_rows": len(raw), "raw_items": int(raw.comment_id.nunique()),
        "raw_raters": int(raw.annotator_id.nunique()),
        "carrier_rows": len(carrier), "carrier_items": int(carrier.comment_id.nunique()),
        "carrier_raters": int(carrier.annotator_id.nunique()),
        "unmatched_carrier_rows": unmatched, "carrier_text_mismatches": text_mismatch,
        "carrier_violence_mismatches": label_mismatch,
        "metadata_evidence_rows": len(observed),
        "metadata_evidence_raters": int(observed.annotator_id.nunique()),
        "evidence_exact_text_differences": int(observed.text_evidence.ne(observed.text_carrier).sum()),
        "evidence_differences_after_whitespace_normalization": int((~whitespace_equal).sum()),
        "reconstructed_gender_raters": metadata.reconstructed_gender.value_counts().to_dict(),
        "raters_without_gender_code": int(joined.loc[joined.reconstructed_gender.isna(), "annotator_id"].nunique()),
        "rows_without_gender_code": int(joined.reconstructed_gender.isna().sum()),
        "code_map": GENDER_CODES,
        "outcome": "original raw hatespeech 0/1/2; no violence or derived score substitution",
        "scope": "research-copy reconstruction; full native demographic flags remain unavailable",
        "unavailable_contrast": "Black/White; non-white is not Black",
    }
    (output.parent / "gender_attachment.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(attach(args.raw, args.carrier, args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
