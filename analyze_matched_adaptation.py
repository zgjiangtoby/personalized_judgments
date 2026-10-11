"""Compare personal text directions under matched anchor budgets and priors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
from analyze_calibration_budget import rank_anchors, selected_anchors, sha256
from analyze_lexical_calibration import id_digest, multiplicity_digest
from analyze_thresholds import (SEED, fit_shared, losses, mean_loss, paired_uncertainty,
                                personalize, prepare_features, probabilities, rater_weights)
from personal_direction import (DIRECTION_PRECISIONS, anchor_coordinates, direction_probabilities,
                                fit_direction, rms_scale, select_precision)

BUDGETS = (2, 4, 6)
SPECS = {
    "qwen": ("Qwen3-8B-Base", "embeddings_qwen_base.npy", "analysis_qwen_base", "analysis_qwen_base_budget"),
    "euro": ("EuroLLM-1.7B-Instruct", "embeddings_euro.npy", "analysis_euro", "analysis_euro_budget"),
    "tfidf": ("TFIDF", "embeddings_euro.npy", "analysis_euro", "analysis_tfidf_budget"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=SPECS, required=True)
    parser.add_argument("--base-directory", type=Path, default=CODE/"work")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-refits", type=int, default=100)
    args = parser.parse_args()
    assert args.training_refits == 100 and not args.output.exists()
    base = args.base_directory
    model, embedding_name, analysis_name, budget_name = SPECS[args.representation]
    analysis, budget_path = base / analysis_name, base / budget_name
    original = json.loads((analysis / "summary.json").read_text())
    original_freeze = json.loads((analysis / "validation_freeze.json").read_text())
    budget_summary = json.loads((budget_path / "summary.json").read_text())
    budget_freeze = json.loads((budget_path / "validation_freeze.json").read_text())
    lexical_path = base / "analysis_tfidf_budget"
    lexical_freeze = json.loads((lexical_path / "validation_freeze.json").read_text())
    archived_draws = json.loads((lexical_path / "training_draws.json").read_text())
    assert original["complete"] and budget_summary["complete"]
    assert archived_draws["seed"] == SEED and len(archived_draws["multiplicity_sha256"]) == 100
    assert original["model"] == ("EuroLLM-1.7B-Instruct" if model == "TFIDF" else model)
    assert budget_freeze["training_refits"] == 100
    assert budget_freeze["source_population_parameters_sha256"] == sha256(analysis / "population_parameters.npz")
    if model == "TFIDF":
        assert Path(lexical_freeze["source_analysis"]).name == analysis.name
        precision_key, parameter_key, ridge_key = "selected_precision_by_budget", "TFIDF", "TFIDF_ridge"
    else:
        assert budget_summary["model"] == model and budget_freeze["training_refit_seed"] == SEED
        assert budget_freeze["source_analysis_summary_sha256"] == sha256(analysis / "summary.json")
        precision_key, parameter_key, ridge_key = "selected_personal_precisions", "shared", "text_ridge"
    boundary_precisions = {b: budget_freeze[precision_key][str(b)] for b in BUDGETS}
    responses = pd.read_parquet(base / "split/responses.parquet")
    items = pd.read_csv(base / "split/items.csv")
    texts = {r["id"]: r["text"] for r in map(json.loads, (base / "split/texts.jsonl").open())}
    train = responses.loc[responses.rater_split.eq("train") & responses.text_split.isin(["train", "anchor"])].reset_index(drop=True)
    targets = {split: responses.loc[responses.rater_split.eq(split) & responses.text_split.eq(split) &
        responses.reconstructed_gender.isin(["women", "men"])].reset_index(drop=True) for split in ("validation", "test")}
    people = np.unique(train.rater_id)
    assert id_digest(people) == archived_draws["training_people_sha256"] == lexical_freeze["training_people_sha256"]
    anchors = rank_anchors(responses)
    x, lexical, indices, feature_record = prepare_features(base / embedding_name, items, texts, set(train.item_id))
    for key, value in feature_record.items():
        previous = original_freeze["features"][key]
        assert (np.isclose(value, previous, rtol=0, atol=1e-12) if isinstance(value, float) else value == previous), key
    if model == "TFIDF":
        x = lexical
    unique_training = items.item_id.isin(set(train.item_id)).to_numpy()
    assert int(unique_training.sum()) == feature_record["PCA_fit_unique_texts"]
    scale = rms_scale(x[unique_training])
    parameters = np.load(analysis / "population_parameters.npz")[parameter_key]
    assert len(parameters) == x.shape[1] + 2
    shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1],
              "parameters": parameters, "ridge": original_freeze[ridge_key]}

    def matrix(frame):
        return x[[indices[str(item)] for item in frame.item_id]]

    frames = {(split, b): selected_anchors(anchors, split, b) for split in targets for b in BUDGETS}
    for split in targets:
        old = responses.loc[responses.rater_split.eq(split) & responses.calibration].reset_index(drop=True)
        pd.testing.assert_frame_equal(frames[split, 2].drop(columns="anchor_rank"), old)
    matrices = {key: matrix(frame) for key, frame in frames.items()}
    target_matrices = {split: matrix(frame) for split, frame in targets.items()}
    # Coordinates use only features and IDs; target labels are not used here.
    designs = {key: anchor_coordinates(matrices[key], frame, target_matrices[key[0]], targets[key[0]], scale)
               for key, frame in frames.items()}
    args.output.mkdir(parents=True)
    fits, tuning, chosen, validation_replication = [], [], {}, []
    for b in BUDGETS:
        key = ("validation", b)
        reference, record = personalize(matrices[key], frames[key], shared, None, boundary_precisions[b])
        fits.append({"stage": "validation_reference", "budget": b, **record})
        p = probabilities(target_matrices["validation"], targets["validation"], shared, personal=reference)
        reference_nll = mean_loss(targets["validation"], p)
        archived = next(r["validation_nll"] for r in budget_freeze["tuning"]
                        if r["budget"] == b and r["precision"] == boundary_precisions[b])
        difference = abs(reference_nll - archived)
        assert difference < 1e-9, ("original validation reference", model, b, difference)
        validation_replication.append(difference)
        candidates = [{"budget": b, "direction_precision": None, "validation_nll": reference_nll}]
        for precision in DIRECTION_PRECISIONS[1:]:
            fitted, record = fit_direction(matrices[key], frames[key], shared, reference,
                                            designs[key], boundary_precisions[b], precision)
            fits.append({"stage": "validation_direction", "budget": b, **record})
            p = direction_probabilities(target_matrices["validation"], targets["validation"], shared, fitted, designs[key])
            candidates.append({"budget": b, "direction_precision": precision,
                               "validation_nll": mean_loss(targets["validation"], p)})
        chosen[b] = select_precision(candidates)
        tuning.extend(candidates)
    freeze = {"representation": model, "budgets": list(BUDGETS), "direction_precision_candidates": list(DIRECTION_PRECISIONS),
              "selected_direction_precisions": chosen, "boundary_precisions": boundary_precisions, "tuning": tuning,
              "feature_scale_RMS_training_unique": scale, "features": feature_record,
              "anchor_rank_by_split_budget": [
                  {"split": split, "budget": b, "people": len(designs[split, b]["people"]),
                   "mean": float(designs[split, b]["ranks"].mean()),
                   "maximum": int(designs[split, b]["ranks"].max()),
                   "histogram": np.bincount(designs[split, b]["ranks"], minlength=b + 1).tolist()}
                  for split in ("validation", "test") for b in BUDGETS],
              "training_people_sha256": id_digest(people), "seed": SEED, "training_refits": 100,
              "new_adapted_test_predictions_computed": False, "original_test_outcomes_known": True,
              "reference_validation_replication_max_abs_difference": max(validation_replication),
              "source_analysis": str(analysis), "source_budget": str(budget_path),
              "source_population_parameters_sha256": sha256(analysis / "population_parameters.npz"),
              "readme_sha256": sha256(CODE / "README.md")}
    write_json(args.output / "validation_freeze.json", freeze)
    print(json.dumps({"stage": "adaptation_validation_frozen", "representation": model}), flush=True)

    test, xe = targets["test"], target_matrices["test"]
    archived_rows = pd.read_parquet(budget_path / "heldout_predictions.parquet")
    keys = ["rater_id", "item_id", "hatespeech", "reconstructed_gender"]
    pd.testing.assert_frame_equal(test[keys], archived_rows[keys])
    assert (len(test), test.rater_id.nunique(), test.item_id.nunique()) == (
        original["test_rows"], original["test_people"], original["test_texts"])
    rows, metrics, differences, replication = test.copy(), [], {}, []
    weights, y = rater_weights(test), test.hatespeech.to_numpy(dtype=int)

    def predict(fitted_shared, budget, stage, draw=None):
        key = ("test", budget)
        reference, record = personalize(matrices[key], frames[key], fitted_shared, None, boundary_precisions[budget])
        fits.append({"stage": stage + "_reference", "budget": budget, "draw": draw, **record})
        pr = probabilities(xe, test, fitted_shared, personal=reference)
        precision = chosen[budget]
        if precision is None:
            return pr, pr.copy(), None
        fitted, record = fit_direction(matrices[key], frames[key], fitted_shared, reference,
                                       designs[key], boundary_precisions[budget], precision)
        fits.append({"stage": stage + "_direction", "budget": budget, "draw": draw, **record})
        return pr, direction_probabilities(xe, test, fitted_shared, fitted, designs[key]), fitted

    for b in BUDGETS:
        reference, adapted, fitted = predict(shared, b, "test_point")
        error = float(np.max(np.abs(reference - archived_rows[[f"budget_{b}_p{k}" for k in range(3)]].to_numpy())))
        assert error < 1e-7, ("archived reference probabilities", b, error)
        replication.append(error)
        point = {"budget": b, "direction_precision": chosen[b],
                 "anchor_rank_mean": float(designs["test", b]["ranks"].mean()),
                 "anchor_rank_maximum": int(designs["test", b]["ranks"].max()),
                 "personal_direction_norm_mean": 0. if fitted is None else float(np.linalg.norm(fitted[:, 2:], axis=1).mean())}
        for name, p in (("reference", reference), ("adapted", adapted)):
            assert np.isfinite(p).all() and np.max(np.abs(p.sum(1) - 1)) < 1e-8
            point[name] = {"nll": mean_loss(test, p),
                           "brier": float(np.average(np.sum((p - np.eye(3)[y]) ** 2, axis=1), weights=weights)),
                           "accuracy": float(np.average(p.argmax(1) == y, weights=weights))}
            rows[f"{name}_{b}_nll"] = losses(test, p)
            for k in range(3):
                rows[f"{name}_{b}_p{k}"] = p[:, k]
        differences[b] = paired_uncertainty(test, rows[f"adapted_{b}_nll"].to_numpy() - rows[f"reference_{b}_nll"].to_numpy())
        metrics.append(point)
    rows.to_parquet(args.output / "heldout_predictions.parquet", index=False)

    lexical_risks = np.load(lexical_path / "training_nll_by_budget.npz")
    paired_original = np.load(base / "comparison_representations/private_paired_training_draws.npz")
    previous = {b: lexical_risks[f"budget_{b}"].copy() for b in BUDGETS}
    if model != "TFIDF":
        for b in BUDGETS:
            previous[b] += paired_original[f"{model}_budget_{b}"]
    assert all(len(values) == 100 and np.isfinite(values).all() for values in previous.values())
    risks = {f"{name}_{b}": [] for name in ("reference", "adapted") for b in BUDGETS}
    rng, xt, draw_ids, refit_replication = np.random.default_rng(SEED), matrix(train), [], []
    for draw in range(100):
        counts = rng.multinomial(len(people), np.full(len(people), 1 / len(people)))
        digest = multiplicity_digest(counts)
        assert digest == archived_draws["multiplicity_sha256"][draw]
        draw_ids.append(digest)
        fitted_shared = fit_shared(xt, train, shared["ridge"], dict(zip(people, counts)), parameters)
        fits.append({"stage": "population_training_refit", "draw": draw, **fitted_shared["record"]})
        for b in BUDGETS:
            reference, adapted, _ = predict(fitted_shared, b, "training_refit", draw)
            nr, na = mean_loss(test, reference), mean_loss(test, adapted)
            error = abs(nr - previous[b][draw])
            assert error < 1e-8, ("archived paired reference refit", model, b, draw, error)
            refit_replication.append(error)
            risks[f"reference_{b}"].append(nr)
            risks[f"adapted_{b}"].append(na)
        if (draw + 1) % 10 == 0:
            print(json.dumps({"representation": model, "training_refits_completed": draw + 1}), flush=True)
    contrasts = []
    for b in BUDGETS:
        draws = np.asarray(risks[f"adapted_{b}"]) - np.asarray(risks[f"reference_{b}"])
        conditional = differences[b]
        train_se = float(np.std(draws, ddof=1))
        se = max(conditional["primary_se_maximum"], train_se)
        contrasts.append({"representation": model, "budget": b, "comparison": "adapted-minus-two-offset",
            **conditional, "training_refit_standard_error": train_se, "combined_se_maximum": se,
            "combined_normal_95_interval": [conditional["delta_nll"] - 1.96 * se, conditional["delta_nll"] + 1.96 * se],
            "training_only_95_interval": np.quantile(draws, [.025, .975]).tolist(), "training_refits": 100})
    np.savez_compressed(args.output / "training_draws.npz", **{k: np.asarray(v) for k, v in risks.items()})
    write_json(args.output / "training_draws.json", {"seed": SEED, "training_people_sha256": id_digest(people),
        "multiplicity_sha256": draw_ids})
    write_json(args.output / "fits.json", fits)
    write_json(args.output / "summary.json", {"representation": model, "metrics": metrics, "comparisons": contrasts,
        "test_rows": len(test), "test_people": test.rater_id.nunique(), "test_texts": test.item_id.nunique(),
        "readme_sha256": freeze["readme_sha256"], "training_refits": 100,
        "reference_point_replication_max_abs_probability_difference": max(replication),
        "reference_training_refit_max_abs_difference": max(refit_replication),
        "timing": "post-result analysis; direction precision selected on validation before target predictions",
        "uncertainty": "original observed target graph, anchor responses, features and tuning fixed; maximum conditional/refit SE is not a joint coverage guarantee",
        "complete": True})
    print(json.dumps({"complete": True, "representation": model, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
