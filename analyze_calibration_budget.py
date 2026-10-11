"""Evaluate ordinal personalization at fixed anchor-label budgets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analyze_thresholds import (HERE, PRECISIONS, SEED, fit_shared, holm, losses,
                                mean_loss, paired_uncertainty, personalize,
                                prepare_features, probabilities, rater_weights)

BUDGETS = (0, 1, 2, 4, 6)
ORDER_SEED = 20261010
ZERO_TOLERANCE = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_anchors(responses):
    """Rank using native IDs and original membership, never response values."""
    anchors = responses.loc[responses.text_split.eq("anchor")].copy()
    assert not anchors.duplicated(["rater_id", "item_id"]).any()
    keys = anchors[["rater_id", "item_id", "calibration"]].copy()
    keys["selected_block"] = ~keys.calibration
    keys["digest"] = [
        hashlib.sha256(f"{ORDER_SEED}|{int(r)}|{int(i)}".encode()).hexdigest()
        for r, i in keys[["rater_id", "item_id"]].itertuples(index=False, name=None)
    ]
    ordered = keys.sort_values(["rater_id", "selected_block", "digest", "item_id"])
    ordered["anchor_rank"] = ordered.groupby("rater_id", sort=False).cumcount() + 1
    anchors["anchor_rank"] = ordered.anchor_rank.reindex(anchors.index)
    expected = anchors.groupby("rater_id").size().clip(upper=2)
    actual = anchors[anchors.calibration].groupby("rater_id").size()
    assert expected.equals(actual.reindex(expected.index)), "original two-anchor support changed"
    assert anchors.anchor_rank.le(2).equals(anchors.calibration)
    assert anchors.anchor_rank.notna().all()
    # Keep original source row order; budget two is the original fitting frame.
    return anchors.reset_index(drop=True)


def selected_anchors(anchors, split, budget):
    selected = anchors.loc[anchors.rater_split.eq(split) & anchors.anchor_rank.le(budget)].copy()
    selected["calibration"] = True
    return selected.reset_index(drop=True)


def complete_family(comparisons):
    inputs = []
    complete = True
    for row in comparisons:
        delta, se = row["delta_nll"], row["primary_se_maximum"]
        assert np.isfinite([delta, se]).all() and se >= 0
        if se <= ZERO_TOLERANCE:
            if abs(delta) <= ZERO_TOLERANCE:
                p, status = 1., "numerically_zero_contrast"
            else:
                p, status = None, "nonzero_contrast_with_degenerate_resampling"
                complete = False
        else:
            p, status = float(2 * norm.sf(abs(delta) / se)), "normal_approximation"
        row["normal_approximation_p"] = p
        row["inference_status"] = status
        row["family_input_p"] = 1. if p is None else p
        inputs.append(row["family_input_p"])
    for row, adjusted in zip(comparisons, holm(inputs)):
        row["holm_p_two_followup"] = None if row["normal_approximation_p"] is None else float(adjusted)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-refits", type=int, default=100)
    args = parser.parse_args()
    assert not args.output.exists() and args.training_refits >= 0
    original_freeze = json.loads((args.analysis / "validation_freeze.json").read_text())
    original_summary = json.loads((args.analysis / "summary.json").read_text())
    assert original_summary["complete"]
    assert Path(original_summary["embeddings"]).resolve() == args.embeddings.resolve()
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
    x, _, indices, feature_record = prepare_features(args.embeddings, items, texts, set(train.item_id))
    assert feature_record.keys() == original_freeze["features"].keys()
    for key, value in feature_record.items():
        previous = original_freeze["features"][key]
        assert (np.isclose(value, previous, rtol=0., atol=1e-12) if isinstance(value, float)
                else value == previous), ("representation transformation changed", key)

    def matrix(frame):
        return x[[indices[str(item)] for item in frame.item_id]]

    parameters = np.load(args.analysis / "population_parameters.npz")["shared"]
    assert len(parameters) == x.shape[1] + 2
    shared = {"beta": parameters[:-2], "center": parameters[-2], "raw_gap": parameters[-1],
              "parameters": parameters, "ridge": original_freeze["text_ridge"]}
    frames = {(split, budget): selected_anchors(anchors, split, budget)
              for split in cells for budget in BUDGETS if budget}
    matrices = {key: matrix(frame) for key, frame in frames.items()}
    for split in cells:
        old = responses.loc[responses.rater_split.eq(split) & responses.calibration].reset_index(drop=True)
        pd.testing.assert_frame_equal(frames[split, 2].drop(columns="anchor_rank"), old)
    args.output.mkdir(parents=True)
    fits, tuning, precisions = [], [], {}
    xv = matrix(cells["validation"])
    for budget in BUDGETS:
        if budget == 0:
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
        precisions[budget] = min(candidates, key=lambda r: r["validation_nll"])["precision"]
    assert precisions[2] == original_freeze["personal_precision"]
    old_tuning = {r["precision"]: r["validation_nll"] for r in original_freeze["tuning"] if r["stage"] == "personal"}
    validation_difference = max(abs(r["validation_nll"] - old_tuning[r["precision"]]) for r in tuning if r["budget"] == 2)
    assert validation_difference < 1e-10, validation_difference
    freeze = {
        "model": original_summary["model"], "budgets": list(BUDGETS), "selected_personal_precisions": precisions,
        "tuning": tuning, "features": feature_record, "shared_text_ridge": shared["ridge"],
        "anchor_order_seed": ORDER_SEED, "training_refit_seed": SEED, "training_refits": args.training_refits,
        "new_budget_test_losses_opened": False,
        "earlier_two_anchor_Euro_outcomes_known_when_protocol_written": True,
        "Qwen_outcomes_unavailable_when_protocol_written": True,
        "budget_two_validation_max_abs_difference": validation_difference,
        "source_analysis": str(args.analysis), "source_embeddings": str(args.embeddings),
        "source_analysis_summary_sha256": sha256(args.analysis / "summary.json"),
        "source_population_parameters_sha256": sha256(args.analysis / "population_parameters.npz"),
        "readme_sha256": sha256(HERE / "README.md"),
    }
    (args.output / "validation_freeze.json").write_text(json.dumps(freeze, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": "followup_validation_frozen", "budgets": list(BUDGETS)}), flush=True)
    xe = matrix(test)
    predictions = {0: probabilities(xe, test, shared)}
    for budget in BUDGETS:
        if budget == 0:
            continue
        personal, record = personalize(matrices["test", budget], frames["test", budget], shared, None, precisions[budget])
        predictions[budget] = probabilities(xe, test, shared, personal=personal)
        fits.append({"stage": "test_personal_calibration", "budget": budget, "precision": precisions[budget], **record})
    old_predictions = pd.read_parquet(args.analysis / "heldout_predictions.parquet")
    old_predictions = old_predictions.loc[old_predictions.primary_women_men].reset_index(drop=True)
    assert test[["rater_id", "item_id", "hatespeech"]].equals(old_predictions[["rater_id", "item_id", "hatespeech"]])
    replication = {}
    for budget, name in ((0, "pooled"), (2, "personal")):
        previous = old_predictions[[f"{name}_p{k}" for k in range(3)]].to_numpy()
        replication[str(budget)] = float(np.max(np.abs(predictions[budget] - previous)))
        assert replication[str(budget)] < 1e-7, replication
    nll = {budget: losses(test, p) for budget, p in predictions.items()}
    weights = rater_weights(test)
    y = test.hatespeech.to_numpy(dtype=int)
    metrics, support = {}, []
    prediction_rows = test.copy()
    for budget, p in predictions.items():
        counts = frames["test", budget].groupby("rater_id").size() if budget else pd.Series(dtype=int)
        actual_counts = test.rater_id.map(counts).fillna(0).astype(int)
        metrics[str(budget)] = {
            "nll": mean_loss(test, p), "brier": float(np.average(((p - np.eye(3)[y]) ** 2).sum(1), weights=weights)),
            "accuracy": float(np.average(p.argmax(1) == y, weights=weights)),
            "rater_mean_anchor_count": float(np.average(actual_counts, weights=weights)),
            "people_with_full_budget": int(test.loc[actual_counts.eq(budget), "rater_id"].nunique()),
            "selected_precision": precisions.get(budget),
        }
        prediction_rows[f"budget_{budget}_anchor_count"] = actual_counts
        prediction_rows[f"budget_{budget}_nll"] = nll[budget]
        for k in range(3):
            prediction_rows[f"budget_{budget}_p{k}"] = p[:, k]
        for count in sorted(actual_counts.unique()):
            mask = actual_counts.eq(count).to_numpy()
            frame = test.loc[mask].reset_index(drop=True)
            support.append({"budget": budget, "actual_anchor_count": int(count), "ratings": len(frame),
                            "raters": frame.rater_id.nunique(), "texts": frame.item_id.nunique(),
                            "nll": float(np.average(nll[budget][mask], weights=rater_weights(frame)))})
    comparisons = [{"comparison": f"budget_{budget}-budget_2", **paired_uncertainty(test, nll[budget] - nll[2])}
                   for budget in (4, 6)]
    inference_complete = complete_family(comparisons)
    prediction_rows.to_parquet(args.output / "heldout_predictions.parquet", index=False)
    pd.DataFrame(support).to_csv(args.output / "calibration_support.csv", index=False)
    anchors[["rater_id", "item_id", "rater_split", "calibration", "anchor_rank"]].to_parquet(
        args.output / "private_anchor_ranks.parquet", index=False)
    refit_values = {f"budget_{budget}-budget_2": [] for budget in (4, 6)}
    refit_replication = []
    previous_draws = np.load(args.analysis / "training_refit_draws.npz")["personal-pooled"]
    rng = np.random.default_rng(SEED)
    people = np.unique(train.rater_id)
    xt = matrix(train)
    for draw in range(args.training_refits):
        multiplicities = dict(zip(people, rng.multinomial(len(people), np.full(len(people), 1 / len(people)))))
        fitted = fit_shared(xt, train, shared["ridge"], multiplicities, parameters)
        fits.append({"stage": "training_shared_refit", "draw": draw, **fitted["record"]})
        point = {}
        for budget in (2, 4, 6):
            personal, record = personalize(matrices["test", budget], frames["test", budget], fitted, None, precisions[budget])
            point[budget] = mean_loss(test, probabilities(xe, test, fitted, personal=personal))
            fits.append({"stage": "training_personal_refit", "draw": draw, "budget": budget, **record})
        for budget in (4, 6):
            refit_values[f"budget_{budget}-budget_2"].append(point[budget] - point[2])
        if draw < len(previous_draws):
            replicated = point[2] - mean_loss(test, probabilities(xe, test, fitted))
            difference = abs(replicated - previous_draws[draw])
            assert difference < 1e-8, (draw, difference)
            refit_replication.append(float(difference))
        if (draw + 1) % 10 == 0:
            print(json.dumps({"training_refits_completed": draw + 1, "requested": args.training_refits}), flush=True)
    refit_summary = {
        name: {"draws": len(values),
               "training_only_95_interval": np.quantile(values, [.025, .975]).tolist() if values else None,
               "standard_error": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        for name, values in refit_values.items()
    }
    np.savez_compressed(args.output / "training_refit_draws.npz", **{k: np.asarray(v) for k, v in refit_values.items()})
    (args.output / "fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False) + "\n")
    summary = {
        "model": original_summary["model"], "metrics_by_up_to_budget": metrics, "comparisons": comparisons,
        "training_refit": refit_summary, "test_rows": len(test), "test_people": test.rater_id.nunique(),
        "test_texts": test.item_id.nunique(), "budget_two_replication_max_abs_probability_difference": replication,
        "matched_original_training_draws": len(refit_replication),
        "matched_training_draw_max_abs_difference": max(refit_replication) if refit_replication else None,
        "family": "two follow-up contrasts, 4 minus 2 and 6 minus 2; separate from original three primary tests",
        "timing": "specified after Euro two-anchor outcomes and before all Qwen outcomes; Euro diagnostic, Qwen prospective follow-up",
        "training_refit_scope": "paired training-rater multiplicities; pooled head refit; PCA/scaling, ridge, selected personal precisions and test anchors fixed; separate from conditional test uncertainty",
        "anchor_policy": "original selected two first, then remaining anchors; SHA256 ordering using IDs only; original fitting row order retained",
        "inference_complete": inference_complete, "complete": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
