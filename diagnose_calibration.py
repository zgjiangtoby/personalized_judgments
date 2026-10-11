"""Summarize ordinal calibration and category discrimination."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def equal_person_weights(frame):
    counts = frame.groupby("rater_id").size()
    values = 1 / frame.rater_id.map(counts).to_numpy(dtype=float)
    return values / values.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    responses = pd.read_parquet(HERE / "work/split/responses.parquet")
    predictions = pd.read_parquet(args.analysis / "heldout_predictions.parquet")
    target = predictions[predictions.primary_women_men].copy()
    assert target.rater_split.eq("test").all() and target.text_split.eq("test").all()
    assert not target.duplicated(["rater_id", "item_id"]).any()
    people = np.sort(target.rater_id.unique())
    all_anchors = responses[responses.rater_id.isin(people) & responses.text_split.eq("anchor")].copy()
    selected = all_anchors[all_anchors.calibration].copy()
    counts = pd.crosstab(selected.rater_id, selected.hatespeech).reindex(index=people, columns=[0., 1., 2.], fill_value=0)
    counts = counts.fillna(0).astype(int)
    assert counts.sum(axis=1).le(2).all()
    pattern = {person: "/".join(str(int(v)) for v in row) for person, row in zip(counts.index, counts.to_numpy())}
    target["anchor_pattern_no_unclear_yes_counts"] = target.rater_id.map(pattern)
    records = []
    for key, part in target.groupby("anchor_pattern_no_unclear_yes_counts", sort=True):
        w = equal_person_weights(part)
        records.append({
            "anchor_pattern_no_unclear_yes_counts": key,
            "raters": int(part.rater_id.nunique()), "responses": len(part), "texts": int(part.item_id.nunique()),
            "pooled_nll": float(w @ part.pooled_nll.to_numpy()),
            "personal_nll": float(w @ part.personal_nll.to_numpy()),
            "personal_minus_pooled_nll": float(w @ (part.personal_nll - part.pooled_nll).to_numpy()),
            **{f"target_response_{j}_share": float(w @ part.hatespeech.eq(j).to_numpy(dtype=float)) for j in range(3)},
            **{f"mean_predicted_p{j}_shift": float(w @ (part[f"personal_p{j}"] - part[f"pooled_p{j}"]).to_numpy()) for j in range(3)},
        })
    by_pattern = pd.DataFrame(records)
    weighted_delta = float(np.average(by_pattern.personal_minus_pooled_nll, weights=by_pattern.raters))
    overall_delta = float(equal_person_weights(target) @ (target.personal_nll - target.pooled_nll).to_numpy())
    assert np.isclose(weighted_delta, overall_delta, rtol=0, atol=1e-12)
    available = all_anchors.groupby("rater_id").size().reindex(people, fill_value=0)
    selected_counts = selected.groupby("rater_id").size().reindex(people, fill_value=0)
    mix = []
    for name, part in [("selected_calibration", selected), ("all_available_anchors", all_anchors), ("primary_target", target)]:
        w = equal_person_weights(part)
        mix.append({"source": name, "raters": int(part.rater_id.nunique()), "responses": len(part),
                    **{f"response_{j}_share": float(w @ part.hatespeech.eq(j).to_numpy(dtype=float)) for j in range(3)}})
    report = {
        "analysis": str(args.analysis), "post_result_diagnostic": True, "gold_used": True,
        "purpose": "describe the fixed calibration design and error allocation; no refitting, prompt change, or test-driven tuning",
        "primary_raters": len(people), "primary_responses": len(target),
        "available_anchor_count_frequencies": {str(k): int(v) for k, v in available.value_counts().sort_index().items()},
        "selected_anchor_count_frequencies": {str(k): int(v) for k, v in selected_counts.value_counts().sort_index().items()},
        "available_anchor_count_quantiles": {str(q): float(available.quantile(q)) for q in [0, .25, .5, .75, .9, 1]},
        "personal_minus_pooled_nll": overall_delta,
        "pattern_weighted_reconstruction": weighted_delta,
        "uncertainty": "descriptive conditional error accounting; no subgroup tests or causal interpretation",
        "primary_protocol_changed": False,
    }
    args.output.mkdir(parents=True)
    by_pattern.to_csv(args.output / "calibration_patterns.csv", index=False)
    pd.DataFrame(mix).to_csv(args.output / "response_mix.csv", index=False)
    (args.output / "support.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
