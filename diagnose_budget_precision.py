"""Report the full budget/precision grid and both decomposition paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_calibration_budget import rank_anchors, selected_anchors, sha256
from analyze_thresholds import (HERE, PRECISIONS, losses, mean_loss, paired_uncertainty,
                                personalize, prepare_features, probabilities)


def interval(frame, difference):
    record = paired_uncertainty(frame, difference)
    record.pop("normal_approximation_p")
    record["status"] = "post-result descriptive component; no new confirmatory test"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--budget-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    freeze = json.loads((args.analysis / "validation_freeze.json").read_text())
    budget_freeze = json.loads((args.budget_analysis / "validation_freeze.json").read_text())
    budget_summary = json.loads((args.budget_analysis / "summary.json").read_text())
    assert budget_summary["complete"]
    assert Path(budget_freeze["source_embeddings"]).resolve() == args.embeddings.resolve()
    assert budget_freeze["source_population_parameters_sha256"] == sha256(args.analysis / "population_parameters.npz")
    responses = pd.read_parquet(HERE / "work/split/responses.parquet")
    items = pd.read_csv(HERE / "work/split/items.csv")
    texts = {row["id"]: row["text"] for row in map(json.loads, (HERE / "work/split/texts.jsonl").open())}
    train = responses.loc[responses.rater_split.eq("train") & responses.text_split.isin(["train", "anchor"])]
    test = responses.loc[responses.rater_split.eq("test") & responses.text_split.eq("test") &
                         responses.reconstructed_gender.isin(["women", "men"])].reset_index(drop=True)
    anchors = rank_anchors(responses)
    x, _, indices, _ = prepare_features(args.embeddings, items, texts, set(train.item_id))

    def matrix(frame):
        return x[[indices[str(item)] for item in frame.item_id]]

    parameters = np.load(args.analysis / "population_parameters.npz")["shared"]
    shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1],
              "parameters": parameters, "ridge": freeze["text_ridge"]}
    xe = matrix(test)
    old = pd.read_parquet(args.budget_analysis / "heldout_predictions.parquet")
    assert test[["rater_id", "item_id", "hatespeech"]].equals(old[["rater_id", "item_id", "hatespeech"]])
    args.output.mkdir(parents=True)
    nll, grid, fits, replication = {}, [], [], {}
    selected_precisions = {int(k): float(v) for k, v in budget_freeze["selected_personal_precisions"].items()}
    for budget in (1, 2, 4, 6):
        selected = selected_anchors(anchors, "test", budget)
        xa = matrix(selected)
        for precision in PRECISIONS:
            personal, record = personalize(xa, selected, shared, None, precision)
            p = probabilities(xe, test, shared, personal=personal)
            nll[budget, precision] = losses(test, p)
            grid.append({"budget": budget, "precision": precision, "nll": mean_loss(test, p),
                         "chosen_on_validation": precision == selected_precisions[budget]})
            fits.append({"budget": budget, "precision": precision, **record})
            if precision == selected_precisions[budget]:
                previous = old[[f"budget_{budget}_p{k}" for k in range(3)]].to_numpy()
                error = float(np.max(np.abs(p - previous)))
                assert error < 1e-7, (budget, error)
                replication[str(budget)] = error
    fixed_precision = []
    for precision in PRECISIONS:
        for budget in (4, 6):
            fixed_precision.append({"budget": budget, "reference_budget": 2, "precision": precision,
                                    **interval(test, nll[budget, precision] - nll[2, precision])})
    components = []
    original_precision = selected_precisions[2]
    for budget in (4, 6):
        final_precision = selected_precisions[budget]
        total = nll[budget, final_precision] - nll[2, original_precision]
        changes = {
            "precision_first": (
                ("precision_at_budget_2", nll[2, final_precision] - nll[2, original_precision]),
                ("budget_at_final_precision", nll[budget, final_precision] - nll[2, final_precision])),
            "budget_first": (
                ("budget_at_original_precision", nll[budget, original_precision] - nll[2, original_precision]),
                ("precision_at_final_budget", nll[budget, final_precision] - nll[budget, original_precision])),
        }
        for path, parts in changes.items():
            assert np.allclose(parts[0][1] + parts[1][1], total, atol=1e-12, rtol=0.)
            for name, values in parts:
                components.append({"budget": budget, "path": path, "component": name,
                                   "original_precision": original_precision, "final_precision": final_precision,
                                   **interval(test, values)})
    pd.DataFrame(grid).to_csv(args.output / "grid.csv", index=False)
    (args.output / "fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    summary = {"model": budget_summary["model"], "fixed_precision_budget_differences": fixed_precision,
               "procedure_difference_components": components, "matched_selected_probability_max_abs_difference": replication,
               "test_rows": len(test), "test_people": test.rater_id.nunique(), "test_texts": test.item_id.nunique(),
               "timing": "specified after Euro budget results and before any Qwen outcomes",
               "scope": "complete frozen precision grid and both exact decomposition paths; fixed-head conditional intervals; no test-selected model or component p-values",
               "readme_sha256": sha256(HERE / "README.md"), "complete": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"grid_cells": len(grid), "fixed_precision_differences": len(fixed_precision),
                      "decomposition_components": len(components), "complete": True}), flush=True)


if __name__ == "__main__":
    main()
