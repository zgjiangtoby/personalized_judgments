"""Fit the TF-IDF control with the same anchor budgets as the LLM representations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from analyze_calibration_budget import BUDGETS, rank_anchors, selected_anchors, sha256
from analyze_thresholds import (HERE, PRECISIONS, SEED, fit_shared, losses,
                                mean_loss, personalize, probabilities, rater_weights)


def id_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def multiplicity_digest(values):
    return id_digest(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="Original completed representation analysis containing its matched TF-IDF control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-refits", type=int, default=100)
    args = parser.parse_args()
    assert not args.output.exists() and args.training_refits == 100
    original_freeze = json.loads((args.analysis / "validation_freeze.json").read_text())
    original_summary = json.loads((args.analysis / "summary.json").read_text())
    assert original_summary["complete"]
    responses = pd.read_parquet(HERE / "work/split/responses.parquet")
    items = pd.read_csv(HERE / "work/split/items.csv")
    texts = {r["id"]: r["text"] for r in map(json.loads, (HERE / "work/split/texts.jsonl").open())}
    anchors = rank_anchors(responses)
    train = responses.loc[responses.rater_split.eq("train") & responses.text_split.isin(["train", "anchor"])].reset_index(drop=True)
    cells = {
        split: responses.loc[responses.rater_split.eq(split) & responses.text_split.eq(split) &
                             responses.reconstructed_gender.isin(["women", "men"])].reset_index(drop=True)
        for split in ("validation", "test")
    }
    test = cells["test"]
    assert (len(test), test.rater_id.nunique(), test.item_id.nunique()) == (
        original_summary["test_rows"], original_summary["test_people"], original_summary["test_texts"])
    indices = {str(item): i for i, item in enumerate(items.item_id)}
    training = items.item_id.isin(set(train.item_id)).to_numpy()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                                 sublinear_tf=True, dtype=np.float64)
    vectorizer.fit([texts[str(item)] for item in items.loc[training, "item_id"]])
    lexical = vectorizer.transform([texts[str(item)] for item in items.item_id])
    assert int(training.sum()) == original_freeze["features"]["PCA_fit_unique_texts"]
    assert lexical.shape[1] == original_freeze["features"]["TFIDF_vocabulary"]

    def matrix(frame):
        return lexical[[indices[str(item)] for item in frame.item_id]]

    parameters = np.load(args.analysis / "population_parameters.npz")["TFIDF"]
    assert len(parameters) == lexical.shape[1] + 2
    shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1],
              "parameters": parameters, "ridge": original_freeze["TFIDF_ridge"]}
    frames = {(split, budget): selected_anchors(anchors, split, budget)
              for split in cells for budget in BUDGETS if budget}
    matrices = {key: matrix(frame) for key, frame in frames.items()}
    for split in cells:
        old = responses.loc[responses.rater_split.eq(split) & responses.calibration].reset_index(drop=True)
        pd.testing.assert_frame_equal(frames[split, 2].drop(columns="anchor_rank"), old)
    xv = matrix(cells["validation"])
    validation_pooled = mean_loss(cells["validation"], probabilities(xv, cells["validation"], shared))
    old_validation = next(r["validation_nll"] for r in original_freeze["tuning"]
                          if r["stage"] == "TFIDF" and r["ridge"] == shared["ridge"])
    assert abs(validation_pooled - old_validation) < 1e-9
    args.output.mkdir(parents=True)
    fits, tuning, precisions = [], [], {}
    for budget in BUDGETS:
        if not budget:
            continue
        candidates = []
        for precision in PRECISIONS:
            personal, record = personalize(matrices["validation", budget], frames["validation", budget],
                                           shared, None, precision)
            value = mean_loss(cells["validation"], probabilities(xv, cells["validation"], shared, personal=personal))
            row = {"budget": budget, "precision": precision, "validation_nll": value}
            candidates.append(row)
            tuning.append(row)
            fits.append({"stage": "validation_personal", "budget": budget, "precision": precision, **record})
        precisions[budget] = min(candidates, key=lambda row: row["validation_nll"])["precision"]
    people = np.unique(train.rater_id)
    freeze = {
        "representation": "TFIDF", "budgets": list(BUDGETS), "text_ridge": shared["ridge"],
        "selected_precision_by_budget": precisions, "tuning": tuning,
        "new_calibrated_test_losses_opened": False, "original_pooled_outcomes_already_known": True,
        "vocabulary_size": lexical.shape[1], "training_unique_texts": int(training.sum()),
        "vocabulary_sha256": hashlib.sha256("\n".join(vectorizer.get_feature_names_out()).encode()).hexdigest(),
        "training_people_sha256": id_digest(people), "seed": SEED, "training_refits": args.training_refits,
        "original_validation_pooled_replication_abs_difference": abs(validation_pooled - old_validation),
        "source_analysis": str(args.analysis),
        "source_population_parameters_sha256": sha256(args.analysis / "population_parameters.npz"),
        "readme_sha256": sha256(HERE / "README.md"),
    }
    (args.output / "validation_freeze.json").write_text(json.dumps(freeze, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": "lexical_validation_frozen", "budgets": list(BUDGETS)}), flush=True)
    xe = matrix(test)
    predictions = {0: probabilities(xe, test, shared)}
    old_predictions = pd.read_parquet(args.analysis / "heldout_predictions.parquet")
    old_predictions = old_predictions.loc[old_predictions.primary_women_men].reset_index(drop=True)
    assert test[["rater_id", "item_id", "hatespeech"]].equals(old_predictions[["rater_id", "item_id", "hatespeech"]])
    old_pooled = old_predictions[[f"TFIDF_p{k}" for k in range(3)]].to_numpy()
    replication = float(np.max(np.abs(predictions[0] - old_pooled)))
    assert replication < 1e-7
    for budget in BUDGETS:
        if not budget:
            continue
        personal, record = personalize(matrices["test", budget], frames["test", budget], shared, None, precisions[budget])
        predictions[budget] = probabilities(xe, test, shared, personal=personal)
        fits.append({"stage": "test_personal", "budget": budget, "precision": precisions[budget], **record})
    metrics, rows = {}, test.copy()
    weights = rater_weights(test)
    y = test.hatespeech.to_numpy(dtype=int)
    for budget, p in predictions.items():
        counts = frames["test", budget].groupby("rater_id").size() if budget else pd.Series(dtype=int)
        actual = test.rater_id.map(counts).fillna(0).astype(int)
        metrics[str(budget)] = {
            "nll": mean_loss(test, p), "brier": float(np.average(((p - np.eye(3)[y]) ** 2).sum(1), weights=weights)),
            "accuracy": float(np.average(p.argmax(1) == y, weights=weights)),
            "rater_mean_anchor_count": float(np.average(actual, weights=weights)),
            "people_with_full_budget": int(test.loc[actual.eq(budget), "rater_id"].nunique()),
            "selected_precision": precisions.get(budget),
        }
        rows[f"budget_{budget}_anchor_count"] = actual
        rows[f"budget_{budget}_nll"] = losses(test, p)
        for k in range(3):
            rows[f"budget_{budget}_p{k}"] = p[:, k]
    rows.to_parquet(args.output / "heldout_predictions.parquet", index=False)
    risks = {f"budget_{budget}": [] for budget in BUDGETS}
    digests = []
    rng = np.random.default_rng(SEED)
    xt = matrix(train)
    for draw in range(args.training_refits):
        counts = rng.multinomial(len(people), np.full(len(people), 1 / len(people)))
        digests.append(multiplicity_digest(counts))
        multiplicity = dict(zip(people, counts))
        fitted = fit_shared(xt, train, shared["ridge"], multiplicity, parameters)
        fits.append({"stage": "training_shared_refit", "draw": draw, **fitted["record"]})
        risks["budget_0"].append(mean_loss(test, probabilities(xe, test, fitted)))
        for budget in BUDGETS:
            if not budget:
                continue
            personal, record = personalize(matrices["test", budget], frames["test", budget], fitted, None, precisions[budget])
            risks[f"budget_{budget}"].append(mean_loss(test, probabilities(xe, test, fitted, personal=personal)))
            fits.append({"stage": "training_personal_refit", "draw": draw, "budget": budget, **record})
        if (draw + 1) % 10 == 0:
            print(json.dumps({"lexical_training_refits_completed": draw + 1, "requested": args.training_refits}), flush=True)
    np.savez_compressed(args.output / "training_nll_by_budget.npz", **{k: np.asarray(v) for k, v in risks.items()})
    (args.output / "training_draws.json").write_text(json.dumps({"seed": SEED,
        "training_people_sha256": id_digest(people), "multiplicity_sha256": digests}, indent=2, allow_nan=False) + "\n")
    (args.output / "fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    summary = {
        "representation": "TFIDF", "metrics_by_up_to_budget": metrics, "test_rows": len(test),
        "test_people": test.rater_id.nunique(), "test_texts": test.item_id.nunique(),
        "pooled_replication_max_abs_probability_difference": replication,
        "training_refits": args.training_refits,
        "training_refit_scope": "training-rater multiplicities; text head and personal calibration refitted; vocabulary, hyperparameters, test anchors and target population fixed",
        "matched_comparisons": "six LLM-minus-TFIDF contrasts at budgets 2,4,6 across two checkpoints are computed only after both checkpoints complete",
        "timing": "specified after Euro primary/budget/precision outcomes, before calibrated TFIDF outcomes and Qwen inference; same human response table",
        "complete": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "training_refits": args.training_refits}), flush=True)


if __name__ == "__main__":
    main()
