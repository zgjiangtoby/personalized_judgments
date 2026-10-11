"""Combine the nine matched personal-direction comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analyze_calibration_budget import sha256
from analyze_thresholds import SEED, holm, rater_weights
from analyze_matched_adaptation import BUDGETS, CODE, SPECS, write_json

EXPECTED = {spec[0] for spec in SPECS.values()}


def complete_family(comparisons):
    assert len(comparisons) == 9
    assert {(r["representation"], r["budget"]) for r in comparisons} == {
        (model, budget) for model in EXPECTED for budget in BUDGETS}
    inputs, complete = [], True
    for row in comparisons:
        delta, se = row["delta_nll"], row["combined_se_maximum"]
        assert np.isfinite([delta, se]).all() and se >= 0
        if se <= 1e-12:
            if abs(delta) <= 1e-12:
                p, status = 1., "numerically_zero_contrast"
            else:
                p, status, complete = None, "nonzero_contrast_with_degenerate_resampling", False
        else:
            p, status = float(2 * norm.sf(abs(delta) / se)), "normal_approximation"
        row.update({"adaptation_family_p_before_Holm": p, "adaptation_family_inference_status": status,
                    "adaptation_family_input_p": 1. if p is None else p})
        inputs.append(row["adaptation_family_input_p"])
    for row, adjusted in zip(comparisons, holm(inputs)):
        row["holm_p_nine_adaptation"] = None if row["adaptation_family_p_before_Holm"] is None else float(adjusted)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.analysis) == 3 and not args.output.exists()
    comparisons, metrics, validation, records, draws_out = [], [], [], [], {}
    target_keys, shared_draws, seen = None, None, set()
    protocol = sha256(CODE / "README.md")
    for directory in args.analysis:
        summary = json.loads((directory / "summary.json").read_text())
        freeze = json.loads((directory / "validation_freeze.json").read_text())
        draw_ids = json.loads((directory / "training_draws.json").read_text())
        model = summary["representation"]
        assert model in EXPECTED and model not in seen
        seen.add(model)
        assert summary["complete"] and summary["training_refits"] == freeze["training_refits"] == 100
        assert summary["readme_sha256"] == freeze["readme_sha256"] == protocol
        assert freeze["representation"] == model and freeze["new_adapted_test_predictions_computed"] is False
        assert draw_ids["seed"] == SEED and len(draw_ids["multiplicity_sha256"]) == 100
        assert draw_ids["training_people_sha256"] == freeze["training_people_sha256"]
        if shared_draws is None:
            shared_draws = draw_ids
        else:
            assert draw_ids == shared_draws
        rows = pd.read_parquet(directory / "heldout_predictions.parquet")
        keys = rows[["rater_id", "item_id", "hatespeech", "reconstructed_gender"]]
        if target_keys is None:
            target_keys = keys
        else:
            pd.testing.assert_frame_equal(keys, target_keys)
        weights = rater_weights(rows)
        arrays = np.load(directory / "training_draws.npz")
        assert set(arrays.files) == {f"{name}_{b}" for name in ("reference", "adapted") for b in BUDGETS}
        assert len(summary["comparisons"]) == 3 and {r["budget"] for r in summary["comparisons"]} == set(BUDGETS)
        assert len(summary["metrics"]) == 3 and {r["budget"] for r in summary["metrics"]} == set(BUDGETS)
        for row in summary["comparisons"]:
            b = row["budget"]
            assert row["representation"] == model and row["draws"] == 2000 and row["training_refits"] == 100
            assert all(arrays[f"{name}_{b}"].shape == (100,) and np.isfinite(arrays[f"{name}_{b}"]).all()
                       for name in ("reference", "adapted"))
            paired = arrays[f"adapted_{b}"] - arrays[f"reference_{b}"]
            draws_out[f"{model}_budget_{b}"] = paired
            train_se = float(np.std(paired, ddof=1))
            point = float(np.average(rows[f"adapted_{b}_nll"] - rows[f"reference_{b}_nll"], weights=weights))
            assert abs(point - row["delta_nll"]) < 1e-12
            assert abs(train_se - row["training_refit_standard_error"]) < 1e-12
            assert abs(max(train_se, row["primary_se_maximum"]) - row["combined_se_maximum"]) < 1e-12
            chosen = freeze["selected_direction_precisions"][str(b)]
            if chosen is None:
                assert np.max(np.abs(paired)) == 0 and np.max(np.abs(rows[f"adapted_{b}_nll"] - rows[f"reference_{b}_nll"])) == 0
            comparisons.append({**row, "selected_direction_precision": chosen,
                                "boundary_precision": freeze["boundary_precisions"][str(b)]})
        metrics.extend({"representation": model, **row} for row in summary["metrics"])
        validation.extend({"representation": model, **row} for row in freeze["tuning"])
        records.append({"representation": model,
            "reference_point_replication_max_abs_probability_difference": summary["reference_point_replication_max_abs_probability_difference"],
            "reference_training_refit_max_abs_difference": summary["reference_training_refit_max_abs_difference"],
            "feature_scale_RMS_training_unique": freeze["feature_scale_RMS_training_unique"],
            "test_rows": summary["test_rows"], "test_people": summary["test_people"], "test_texts": summary["test_texts"]})
    assert seen == EXPECTED
    complete = complete_family(comparisons)
    args.output.mkdir(parents=True)
    pd.DataFrame(comparisons).to_csv(args.output / "comparisons.csv", index=False)
    pd.DataFrame(validation).to_csv(args.output / "validation_grid.csv", index=False)
    np.savez_compressed(args.output / "private_paired_training_draws.npz", **draws_out)
    write_json(args.output / "summary.json", {"family_size": 9, "comparisons": comparisons, "metrics": metrics,
        "representations": records, "readme_sha256": protocol, "complete": True, "inference_complete": complete,
        "timing": "post-result analysis; original results and comparison families retained",
        "uncertainty": "approximate observed-graph conditional inference; maximum conditional and paired training SE, not an unconditional coverage guarantee"})
    print(json.dumps({"complete": True, "inference_complete": complete, "family_size": 9}), flush=True)


if __name__ == "__main__":
    main()
