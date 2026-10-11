"""Fit personal ordinal score directions in the span of the anchor features."""

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from analyze_thresholds import ordinal, population_predictions

RANK_TOLERANCE = 1e-10
DIRECTION_PRECISIONS = (None, 8., 2., .5)  # None is the exactly nested procedure.


def dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def rms_scale(x):
    squared = x.multiply(x).sum() if sparse.issparse(x) else np.sum(np.asarray(x) ** 2)
    scale = float(np.sqrt(squared / x.shape[0]))
    assert np.isfinite(scale) and scale > 0
    return scale


def anchor_coordinates(xa, anchors, xt, targets, scale):
    """No target labels enter a person's feature basis or projection."""
    assert anchors.calibration.all() and anchors.text_split.eq("anchor").all()
    assert len(anchors) == xa.shape[0] and len(targets) == xt.shape[0]
    assert xa.shape[1] == xt.shape[1] and scale > 0
    people, ai = np.unique(anchors.rater_id.to_numpy(), return_inverse=True)
    lookup = {r: i for i, r in enumerate(people)}
    ti = np.array([lookup.get(r, -1) for r in targets.rater_id], dtype=int)
    maximum = int(np.bincount(ai).max())
    za, zt = np.zeros((len(anchors), maximum)), np.zeros((len(targets), maximum))
    ranks = np.zeros(len(people), dtype=int)
    # Group positions once; the same coordinates are reused for all refits.
    anchor_positions = anchors.groupby("rater_id", sort=False).indices
    target_positions = targets.groupby("rater_id", sort=False).indices
    for person, index in lookup.items():
        rows = np.asarray(anchor_positions[person], dtype=int)
        a = xa[rows] / scale
        gram = dense(a @ a.T)
        values, vectors = np.linalg.eigh((gram + gram.T) / 2)
        keep = values > RANK_TOLERANCE * max(1., float(values[-1]))
        values, vectors = values[keep], vectors[:, keep]
        rank = len(values)
        ranks[index] = rank
        if not rank:
            continue
        za[rows, :rank] = vectors * np.sqrt(values)
        target_rows = np.asarray(target_positions.get(person, []), dtype=int)
        if len(target_rows):
            cross = dense((xt[target_rows] / scale) @ a.T)
            zt[target_rows, :rank] = cross @ (vectors / np.sqrt(values))
    assert np.isfinite(za).all() and np.isfinite(zt).all()
    return {"people": people, "anchor_person": ai, "target_person": ti,
            "anchor_z": za, "target_z": zt, "ranks": ranks,
            "anchor_raters": anchors.rater_id.to_numpy().copy(),
            "target_raters": targets.rater_id.to_numpy().copy()}


def direction_objective(parameters, design, eta, center, gap, y,
                        boundary_precision, direction_precision):
    people = len(design["people"])
    z, index = design["anchor_z"], design["anchor_person"]
    offsets = parameters.reshape(people, 2 + z.shape[1])
    score = eta + np.einsum("ij,ij->i", z, offsets[index, 2:])
    loss, de, dc, dg = ordinal(score, center + offsets[index, 0], gap + offsets[index, 1], y)
    gradient = np.zeros_like(offsets)
    gradient[:, 0] = np.bincount(index, dc, minlength=people)
    gradient[:, 1] = np.bincount(index, dg, minlength=people)
    for j in range(z.shape[1]):
        gradient[:, j + 2] = np.bincount(index, de * z[:, j], minlength=people)
    gradient[:, :2] += boundary_precision * offsets[:, :2]
    gradient[:, 2:] += direction_precision * offsets[:, 2:]
    penalty = (boundary_precision * np.sum(offsets[:, :2] ** 2) +
               direction_precision * np.sum(offsets[:, 2:] ** 2)) / 2
    return float((loss.sum() + penalty) / people), gradient.reshape(-1) / people


def fit_direction(xa, anchors, shared, reference_offsets, design,
                  boundary_precision, direction_precision):
    assert direction_precision in DIRECTION_PRECISIONS[1:]
    assert np.array_equal(anchors.rater_id.to_numpy(), design["anchor_raters"])
    eta, center, gap = population_predictions(xa, anchors, shared)
    y = anchors.hatespeech.to_numpy(dtype=int)
    people, rank = len(design["people"]), design["anchor_z"].shape[1]
    initial = np.zeros((people, 2 + rank))
    initial[:, :2] = np.array([reference_offsets[r] for r in design["people"]])

    def objective(parameters):
        return direction_objective(parameters, design, eta, center, gap, y,
                                   boundary_precision, direction_precision)

    before = objective(initial.reshape(-1))[0]
    # Disable function-change stopping so convergence must meet the gradient tolerance.
    solution = minimize(objective, initial.reshape(-1), jac=True, method="L-BFGS-B",
                        options={"maxiter": 2000, "ftol": 0., "gtol": 1e-9, "maxls": 50})
    parameters = solution.x
    value, gradient = objective(parameters)
    normalized = float(np.max(np.abs(gradient)))
    assert solution.success and normalized < 1e-8, ("personal direction", str(solution.message), normalized)
    record = {"fit": "personal_text_direction_MAP", "iterations": int(solution.nit),
              "objective": value, "max_gradient": normalized, "message": str(solution.message),
              "function_tolerance": 0., "gradient_tolerance": 1e-9}
    maximum = float(np.max(np.abs(gradient)) * people)
    assert maximum < 1e-4, ("unnormalized personal gradient", maximum)
    assert value <= before + 1e-9, ("nested initialization objective", before, value)
    record.update({"boundary_precision": boundary_precision, "direction_precision": direction_precision,
                   "initial_objective": before, "unnormalized_max_gradient": maximum,
                   "people": people, "maximum_direction_rank": int(design["ranks"].max())})
    return parameters.reshape(people, 2 + rank), record


def direction_probabilities(xt, targets, shared, fitted, design):
    assert np.array_equal(targets.rater_id.to_numpy(), design["target_raters"])
    eta, center, gap = population_predictions(xt, targets, shared)
    index = design["target_person"]
    has = index >= 0
    local = fitted[index[has]]
    eta[has] += np.einsum("ij,ij->i", design["target_z"][has], local[:, 2:])
    center[has] += local[:, 0]
    gap[has] += local[:, 1]
    return ordinal(eta, center, gap)


def select_precision(candidates):
    assert [r["direction_precision"] for r in candidates] == list(DIRECTION_PRECISIONS)
    assert all(np.isfinite(r["validation_nll"]) for r in candidates)
    minimum = min(r["validation_nll"] for r in candidates)
    return next(r["direction_precision"] for r in candidates if r["validation_nll"] <= minimum + 1e-12)
