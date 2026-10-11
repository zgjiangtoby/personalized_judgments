"""Compare LLM and TF-IDF calibration with paired training refits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analyze_calibration_budget import BUDGETS, rank_anchors, selected_anchors, sha256
from analyze_lexical_calibration import id_digest, multiplicity_digest
from analyze_thresholds import (HERE, SEED, fit_shared, holm, losses, mean_loss,
                                ordinal, paired_uncertainty, personalize, prepare_features, probabilities, rater_weights)

PERSONAL_BUDGETS = (2, 4, 6)
EXPECTED_MODELS = {"EuroLLM-1.7B-Instruct", "Qwen3-8B-Base"}
ZERO_TOLERANCE = 1e-12


def complete_six_family(comparisons):
    assert len(comparisons) == 6
    assert {(r["model"], r["budget"]) for r in comparisons} == {
        (model, budget) for model in EXPECTED_MODELS for budget in PERSONAL_BUDGETS}
    inputs = []
    inference_complete = True
    for row in comparisons:
        delta, se = row["delta_nll"], row["combined_se_maximum"]
        assert np.isfinite([delta, se]).all() and se >= 0
        if se <= ZERO_TOLERANCE:
            if abs(delta) <= ZERO_TOLERANCE:
                value, status = 1., "numerically_zero_contrast"
            else:
                value, status = None, "nonzero_contrast_with_degenerate_resampling"
                inference_complete = False
        else:
            value, status = float(2 * norm.sf(abs(delta) / se)), "normal_approximation"
        row["additional_family_p_before_Holm"] = value
        row["additional_family_inference_status"] = status
        row["additional_family_input_p"] = 1. if value is None else value
        inputs.append(row["additional_family_input_p"])
    for row, value in zip(comparisons, holm(inputs)):
        row["holm_p_six_calibrated_representation"] = (
            None if row["additional_family_p_before_Holm"] is None else float(value))
    return inference_complete


def verify_lexical_control(lexical_path, analysis_path, lexical_matrix, indices, train, test,
                           old_predictions, lexical_rows, freeze):
    """Reuse one canonical lexical fit; verify each independent archived fit separately."""
    lexical_freeze = json.loads((lexical_path/"validation_freeze.json").read_text())
    canonical_path = Path(lexical_freeze["source_analysis"])
    assert lexical_freeze["source_population_parameters_sha256"] == sha256(canonical_path/"population_parameters.npz")
    assert freeze["TFIDF_ridge"] == lexical_freeze["text_ridge"]
    assert freeze["features"]["TFIDF_vocabulary"] == lexical_freeze["vocabulary_size"] == lexical_matrix.shape[1]
    assert freeze["features"]["PCA_fit_unique_texts"] == lexical_freeze["training_unique_texts"]
    xt = lexical_matrix[[indices[str(item)] for item in train.item_id]]
    xe = lexical_matrix[[indices[str(item)] for item in test.item_id]]
    y = train.hatespeech.to_numpy(dtype=int)
    weights = rater_weights(train)
    ridge = lexical_freeze["text_ridge"]
    records, point = {}, {}
    for name, directory, target in (("canonical", canonical_path, lexical_rows[[f"budget_0_p{k}" for k in range(3)]].to_numpy()),
                                    ("independent_original", analysis_path, old_predictions[[f"TFIDF_p{k}" for k in range(3)]].to_numpy())):
        parameters = np.load(directory/"population_parameters.npz")["TFIDF"]
        assert len(parameters) == lexical_matrix.shape[1]+2
        shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1]}
        point[name] = probabilities(xe, test, shared)
        difference = float(np.max(np.abs(point[name]-target)))
        assert difference < 1e-7, (name, "saved-fit probability reproduction", difference)
        loss, de, dc, dg = ordinal(np.asarray(xt@shared["beta"]).reshape(-1),
            np.full(len(y), shared["center"]), np.full(len(y), shared["raw_gap"]), y)
        gradient = np.r_[np.asarray(xt.T@(weights*de)).reshape(-1)+ridge*shared["beta"], weights@dc, weights@dg]/weights.sum()
        maximum_gradient = float(np.abs(gradient).max())
        assert maximum_gradient < 1e-5, (name, maximum_gradient)
        objective = float((weights@loss+ridge/2*(shared["beta"]@shared["beta"]))/weights.sum())
        records[name] = {"saved_probability_reproduction_max_abs_difference": difference,
                         "objective": objective, "max_gradient": maximum_gradient}
    objective_difference = records["independent_original"]["objective"]-records["canonical"]["objective"]
    assert abs(objective_difference) < 1e-9, ("independent lexical objectives disagree", objective_difference)
    return {"canonical_source_analysis": str(canonical_path), "checks": records,
            "independent_objective_difference": objective_difference,
            "independent_fit_probability_max_abs_difference": float(np.max(np.abs(point["canonical"]-point["independent_original"]))),
            "control_used": "one pre-Qwen canonical lexical fit and its matched budget/refit artifacts; no fit chosen by test performance"}


def checkpoint(embedding_path, analysis_path, budget_path, lexical_path,
               responses, items, texts, anchors, train, test, people, lexical_rows,
               lexical_risks, lexical_draws, draw_count):
    original = json.loads((analysis_path / "summary.json").read_text())
    freeze = json.loads((analysis_path / "validation_freeze.json").read_text())
    budget_summary = json.loads((budget_path / "summary.json").read_text())
    budget_freeze = json.loads((budget_path / "validation_freeze.json").read_text())
    assert original["complete"] and budget_summary["complete"]
    assert original["model"] in EXPECTED_MODELS
    assert Path(original["embeddings"]).resolve() == embedding_path.resolve()
    assert budget_freeze["source_analysis_summary_sha256"] == sha256(analysis_path / "summary.json")
    assert budget_freeze["source_population_parameters_sha256"] == sha256(analysis_path / "population_parameters.npz")
    assert budget_freeze["training_refit_seed"] == SEED and budget_freeze["training_refits"] == draw_count
    assert freeze["training_refits"] == draw_count
    assert budget_summary["model"] == original["model"]
    assert (len(test), test.rater_id.nunique(), test.item_id.nunique()) == (
        original["test_rows"], original["test_people"], original["test_texts"])
    rows = pd.read_parquet(budget_path / "heldout_predictions.parquet")
    keys = ["rater_id", "item_id", "hatespeech", "reconstructed_gender"]
    pd.testing.assert_frame_equal(rows[keys], test[keys])
    pd.testing.assert_frame_equal(rows[keys], lexical_rows[keys])
    old_predictions = pd.read_parquet(analysis_path / "heldout_predictions.parquet")
    old_predictions = old_predictions.loc[old_predictions.primary_women_men].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_predictions[keys], test[keys])
    lexical_replication = float(np.max(np.abs(
        old_predictions[[f"TFIDF_p{k}" for k in range(3)]].to_numpy() -
        lexical_rows[[f"budget_0_p{k}" for k in range(3)]].to_numpy())))
    for budget in BUDGETS:
        assert rows[f"budget_{budget}_anchor_count"].equals(lexical_rows[f"budget_{budget}_anchor_count"])
        p = rows[[f"budget_{budget}_p{k}" for k in range(3)]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.)
        assert np.max(np.abs(losses(test, p) - rows[f"budget_{budget}_nll"].to_numpy())) < 1e-12
    x, lexical_matrix, indices, feature_record = prepare_features(embedding_path, items, texts, set(train.item_id))
    assert feature_record.keys() == freeze["features"].keys()
    for key, value in feature_record.items():
        previous = freeze["features"][key]
        assert (np.isclose(value, previous, rtol=0., atol=1e-12) if isinstance(value, float)
                else value == previous), ("feature transformation changed", key)
    lexical_verification = verify_lexical_control(lexical_path, analysis_path, lexical_matrix, indices, train, test,
                                                   old_predictions, lexical_rows, freeze)
    assert abs(lexical_verification["independent_fit_probability_max_abs_difference"]-lexical_replication) < 1e-7

    def matrix(frame):
        return x[[indices[str(item)] for item in frame.item_id]]

    parameters = np.load(analysis_path / "population_parameters.npz")["shared"]
    assert len(parameters) == x.shape[1] + 2
    shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1],
              "parameters": parameters, "ridge": freeze["text_ridge"]}
    precisions = {int(k): v for k, v in budget_freeze["selected_personal_precisions"].items()}
    assert precisions[2] == freeze["personal_precision"]
    frames = {budget: selected_anchors(anchors, "test", budget) for budget in PERSONAL_BUDGETS}
    old_anchors = responses.loc[responses.rater_split.eq("test") & responses.calibration].reset_index(drop=True)
    pd.testing.assert_frame_equal(frames[2].drop(columns="anchor_rank"), old_anchors)
    matrices = {budget: matrix(frame) for budget, frame in frames.items()}
    xe, xt = matrix(test), matrix(train)
    pooled_replication = float(np.max(np.abs(probabilities(xe, test, shared) -
        rows[[f"budget_0_p{k}" for k in range(3)]].to_numpy())))
    assert pooled_replication < 1e-7
    old_draws = np.load(analysis_path / "training_refit_draws.npz")["personal-pooled"]
    budget_draws = np.load(budget_path / "training_refit_draws.npz")
    assert len(old_draws) == draw_count
    assert all(len(budget_draws[f"budget_{b}-budget_2"]) == draw_count for b in (4, 6))
    fits, digests, replication = [], [], []
    risks = {f"budget_{budget}": [] for budget in (0, *PERSONAL_BUDGETS)}
    rng = np.random.default_rng(SEED)
    for draw in range(draw_count):
        counts = rng.multinomial(len(people), np.full(len(people), 1 / len(people)))
        digest = multiplicity_digest(counts)
        assert digest == lexical_draws[draw], ("unpaired training multiplicities", draw)
        digests.append(digest)
        fitted = fit_shared(xt, train, shared["ridge"], dict(zip(people, counts)), parameters)
        fits.append({"model": original["model"], "stage": "training_shared_refit", "draw": draw, **fitted["record"]})
        point = {0: mean_loss(test, probabilities(xe, test, fitted))}
        for budget in PERSONAL_BUDGETS:
            personal, record = personalize(matrices[budget], frames[budget], fitted, None, precisions[budget])
            point[budget] = mean_loss(test, probabilities(xe, test, fitted, personal=personal))
            fits.append({"model": original["model"], "stage": "training_personal_refit",
                         "draw": draw, "budget": budget, **record})
        difference = max(abs(point[2] - point[0] - old_draws[draw]),
                         *(abs(point[b] - point[2] - budget_draws[f"budget_{b}-budget_2"][draw]) for b in (4, 6)))
        assert difference < 1e-8, ("original training-refit draw not reproduced", draw, difference)
        replication.append(float(difference))
        for budget, value in point.items():
            risks[f"budget_{budget}"].append(value)
        if (draw + 1) % 10 == 0:
            print(json.dumps({"model": original["model"], "paired_representation_refits": draw + 1,
                              "requested": draw_count}), flush=True)
    contrasts = []
    paired_draws = {}
    for budget in PERSONAL_BUDGETS:
        result = paired_uncertainty(test, rows[f"budget_{budget}_nll"].to_numpy() -
                                    lexical_rows[f"budget_{budget}_nll"].to_numpy())
        draws = np.asarray(risks[f"budget_{budget}"]) - lexical_risks[f"budget_{budget}"]
        assert len(draws) == draw_count and np.isfinite(draws).all()
        training_se = float(np.std(draws, ddof=1))
        combined_se = max(result["primary_se_maximum"], training_se)
        contrasts.append({"model": original["model"], "budget": budget,
            "comparison": "LLM-minus-TFIDF at the same up-to calibration budget",
            **result, "training_refit_standard_error": training_se,
            "training_only_95_interval": np.quantile(draws, [.025, .975]).tolist(),
            "combined_se_maximum": combined_se,
            "combined_normal_95_interval": [result["delta_nll"] - 1.96 * combined_se,
                                            result["delta_nll"] + 1.96 * combined_se],
            "training_refits": draw_count})
        paired_draws[f"{original['model']}_budget_{budget}"] = draws
    metrics = {str(b): {"LLM": budget_summary["metrics_by_up_to_budget"][str(b)],
                       "TFIDF": json.loads((lexical_path / "summary.json").read_text())["metrics_by_up_to_budget"][str(b)]}
               for b in BUDGETS}
    record = {"model": original["model"], "analysis": str(analysis_path), "budget_analysis": str(budget_path),
              "features": feature_record, "matched_training_draws": len(digests),
              "original_training_draw_max_abs_difference": max(replication),
              "independent_TFIDF_fit_probability_max_abs_difference": lexical_replication,
              "lexical_control_verification": lexical_verification,
              "original_LLM_pooled_replication_max_abs_difference": pooled_replication,
              "metrics_by_up_to_budget": metrics}
    return contrasts, paired_draws, fits, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, action="append", required=True)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--budget-analysis", type=Path, action="append", required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-refits", type=int, default=100)
    args = parser.parse_args()
    assert not args.output.exists() and args.training_refits == 100
    assert len(args.embeddings) == len(args.analysis) == len(args.budget_analysis) == 2
    assert {json.loads((p / "summary.json").read_text())["model"] for p in args.analysis} == EXPECTED_MODELS
    lexical_summary = json.loads((args.lexical / "summary.json").read_text())
    lexical_freeze = json.loads((args.lexical / "validation_freeze.json").read_text())
    draw_record = json.loads((args.lexical / "training_draws.json").read_text())
    assert lexical_summary["complete"] and lexical_summary["training_refits"] == args.training_refits
    assert draw_record["seed"] == SEED and len(draw_record["multiplicity_sha256"]) == args.training_refits
    lexical_risks = np.load(args.lexical / "training_nll_by_budget.npz")
    assert all(len(lexical_risks[f"budget_{b}"]) == args.training_refits and
               np.isfinite(lexical_risks[f"budget_{b}"]).all() for b in BUDGETS)
    responses = pd.read_parquet(HERE / "work/split/responses.parquet")
    items = pd.read_csv(HERE / "work/split/items.csv")
    texts = {r["id"]: r["text"] for r in map(json.loads, (HERE / "work/split/texts.jsonl").open())}
    anchors = rank_anchors(responses)
    train = responses.loc[responses.rater_split.eq("train") & responses.text_split.isin(["train", "anchor"])].reset_index(drop=True)
    test = responses.loc[responses.rater_split.eq("test") & responses.text_split.eq("test") &
                         responses.reconstructed_gender.isin(["women", "men"])].reset_index(drop=True)
    people = np.unique(train.rater_id)
    assert draw_record["training_people_sha256"] == lexical_freeze["training_people_sha256"] == id_digest(people)
    lexical_rows = pd.read_parquet(args.lexical / "heldout_predictions.parquet")
    keys = ["rater_id", "item_id", "hatespeech", "reconstructed_gender"]
    pd.testing.assert_frame_equal(lexical_rows[keys], test[keys])
    for budget in BUDGETS:
        p = lexical_rows[[f"budget_{budget}_p{k}" for k in range(3)]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.)
        assert np.max(np.abs(losses(test, p) - lexical_rows[f"budget_{budget}_nll"].to_numpy())) < 1e-12
    all_contrasts, all_draws, all_fits, records = [], {}, [], []
    args.output.mkdir(parents=True)
    for embedding, analysis, budget in zip(args.embeddings, args.analysis, args.budget_analysis):
        contrasts, draws, fits, record = checkpoint(
            embedding, analysis, budget, args.lexical, responses, items, texts, anchors, train, test, people,
            lexical_rows, lexical_risks, draw_record["multiplicity_sha256"], args.training_refits)
        all_contrasts.extend(contrasts)
        all_draws.update(draws)
        all_fits.extend(fits)
        records.append(record)
    inference_complete = complete_six_family(all_contrasts)
    np.savez_compressed(args.output / "private_paired_training_draws.npz", **all_draws)
    (args.output / "fits.json").write_text(json.dumps(all_fits, indent=2, allow_nan=False) + "\n")
    summary = {
        "models": records, "comparisons": all_contrasts,
        "family": "six additional same-budget LLM-minus-TFIDF contrasts: budgets 2,4,6 by two frozen checkpoints",
        "test_rows": len(test), "test_people": test.rater_id.nunique(), "test_texts": test.item_id.nunique(),
        "training_refits_per_representation": args.training_refits,
        "uncertainty": "paired conditional two-way test uncertainty and paired training-rater refits; additional normal calculation uses maximum SE, not a joint coverage guarantee",
        "timing": "matched lexical calibration specified after Euro budget diagnostics and before Qwen inference; same human response table, no independent population confirmation",
        "readme_sha256": sha256(HERE / "README.md"),
        "inference_complete": inference_complete, "complete": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "inference_complete": inference_complete,
                      "contrasts": len(all_contrasts), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
