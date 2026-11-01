import warnings
from copy import deepcopy

import numpy as np

from cdrc.common.utils import get_nested_attribute
from cdrc.profees_coding.em.data.schema import ComplexityLevel
from cdrc.profees_coding.em.models.utils import get_complexity_from_code

COMPLEXITY_LEVELS = [1, 2, 3, 4]
QUADRATIC_WEIGHTS = np.array([[0, 1, 4, 9], [1, 0, 1, 4], [4, 1, 0, 1], [9, 4, 1, 0]])


def _normalize_complexity_level(x):
    """Coerce various representations into a ComplexityLevel enum, or None if unknown."""
    if x is None:
        return None
    if isinstance(x, ComplexityLevel):
        return x
    if isinstance(x, dict):
        # common shape: {"level": "Low"} or {"level": "ComplexityLevel.LOW"}
        return _normalize_complexity_level(x.get("level"))
    if isinstance(x, str):
        s = x.strip()
        # allow "ComplexityLevel.LOW" style
        if "." in s:
            s = s.split(".")[-1]
        s_upper = s.upper()
        # lightweight aliasing
        alias = {
            "MINIMAL": "Minimal",
            "LOW": "Low",
            "MODERATE": "Moderate",
            "HIGH": "High",
        }
        label = alias.get(s_upper, s.title())
        try:
            return ComplexityLevel(label)
        except ValueError:
            return None
    # unsupported type
    return None


def complexity_level_to_ordinal(level):
    lvl = _normalize_complexity_level(level)
    mapping = {ComplexityLevel.MINIMAL: 1, ComplexityLevel.LOW: 2, ComplexityLevel.MODERATE: 3, ComplexityLevel.HIGH: 4}
    return mapping.get(lvl, None)


def extract_complexity_levels(gold, outputs, label_key):
    """
    Extracts complexity levels of the label_key for the gold and outputs.
    If the label is not present for a gold instance, then the instance is skipped.

    Parameters:
    gold (array-like): True values.
    outputs (array-like): Predicted values.
    label_key (str): path to the complexity level.

    Returns:
    Gold and output complexity levels.
    """
    # hack to handle the rci case
    if label_key.startswith("before_rci."):
        rci_case = True
        label_key = label_key.replace("before_rci.", "")
    else:
        rci_case = False

    gold_levels = []
    output_levels = []
    for g, o in zip(gold, outputs):
        # hack to handle the rci case
        if rci_case:
            o = deepcopy(o).additional_output["mdm_results"]

        try:
            gold_complexity = get_nested_attribute(g, label_key)
        except ValueError:
            print(f"Gold data doesn't have an attribute {label_key}!")
            continue
        try:
            output_complexity = get_nested_attribute(o, label_key)
        except ValueError:
            print(f"Output data doesn't have an attribute {label_key}!")
            continue
            output_complexity = None

        # if label_key is cpt code, we need to extract the level from the code
        if label_key == "cpt.code":
            if not isinstance(gold_complexity, list):
                gold_complexity = [gold_complexity]
            if not isinstance(output_complexity, list):
                output_complexity = [output_complexity]

            for gc, oc in zip(gold_complexity, output_complexity):
                gc_level = complexity_level_to_ordinal(get_complexity_from_code(gc))
                oc_level = complexity_level_to_ordinal(get_complexity_from_code(oc))
                if gc_level is not None and oc_level is not None:
                    gold_levels.append(gc_level)
                    output_levels.append(oc_level)
        else:
            # the previous method might results in different lengths for gold and output levels
            if not isinstance(gold_complexity, list):
                gold_complexity = [gold_complexity]
            if not isinstance(output_complexity, list):
                output_complexity = [output_complexity]

            for gc, oc in zip(gold_complexity, output_complexity):
                gc_level = complexity_level_to_ordinal(gc)
                oc_level = complexity_level_to_ordinal(oc)
                if gc_level is None or oc_level is None:
                    continue
                gold_levels.append(gc_level)
                output_levels.append(oc_level)
    if not gold_levels:
        warnings.warn(f"No Gold levels, either the output schema doesn't have an attribute {label_key}! or gold data has no mdm-complexity levels")
    return gold_levels, output_levels


def custom_weighted_kappa(y_true, y_pred, W):
    """
    Calculate custom weighted kappa

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    W (2D array): Weights

    Returns:
    Custom weighted kappa
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."

    N = len(W)  # len(np.unique(y_true))  # Number of classes

    assert all(value in range(1, N + 1) for value in y_true), "All y_true values must be within the dimensions of the provided weight matrix."
    assert all(value in range(1, N + 1) for value in y_pred), "All y_pred values must be within the dimensions of the provided weight matrix."

    # Confusion matrix
    confusion = np.zeros((N, N))
    for t, p in zip(y_true, y_pred):
        confusion[t - 1, p - 1] += 1  # Shift by 1 if classes start at 1

    # Expected matrix
    row_marginals = np.sum(confusion, axis=1)
    col_marginals = np.sum(confusion, axis=0)
    expected = np.outer(row_marginals, col_marginals) / np.sum(confusion)

    # Compute weighted kappa
    confusion_w = np.sum(W * confusion)
    expected_w = np.sum(W * expected)

    if np.isclose(expected_w, 0):
        kappa = 1
    else:
        kappa = 1 - confusion_w / expected_w
    return kappa


def calculate_coding_errors(y_true, y_pred, num_levels=len(COMPLEXITY_LEVELS)):
    """
    Calculate overcoding and undercoding error

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    num_levels (int): number of levels

    Returns:
    dict: A dictionary containing overall overcoding/undercoding errors.
    """

    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."

    if y_true:
        mean_undercoding_error = sum([g - p for p, g in zip(y_pred, y_true) if p < g]) / len(y_true) / num_levels
        mean_overcoding_error = sum([p - g for p, g in zip(y_pred, y_true) if p > g]) / len(y_true) / num_levels
    else:
        mean_undercoding_error, mean_overcoding_error = None, None

    # Return the results as a combined dictionary
    return {"mean_overcoding_error": mean_overcoding_error, "mean_undercoding_error": mean_undercoding_error}


def calculate_coding_rates(y_true, y_pred):
    """
    Calculate overcoding and undercoding rates

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: A dictionary containing overall overcoding/undercoding rates
          and rates for each difference level.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."

    # Initialize dictionaries to store rates
    overcoding_rate = 0
    undercoding_rate = 0
    correct = 0
    # Calculate total number of cases
    total_cases = len(y_true)

    # Loop through and calculate the rates
    for i in range(total_cases):
        diff = y_pred[i] - y_true[i]
        if diff > 0:  # Overcoding case
            overcoding_rate += 1
        elif diff < 0:  # Undercoding case
            undercoding_rate += 1
        else:
            correct += 1

    # Convert counts to percentages
    if total_cases == 0:
        overcoding_rate = 0.0  # or float('nan') if you prefer
        undercoding_rate = 0.0  # or float('nan') if you prefer
        accuracy = 0.0  # or float('nan') if you prefer
    else:
        overcoding_rate = (overcoding_rate / total_cases) * 100
        undercoding_rate = (undercoding_rate / total_cases) * 100
        accuracy = (correct / total_cases) * 100
    # Return the results as a combined dictionary
    return {"overcoding_rate": overcoding_rate, "undercoding_rate": undercoding_rate, "accuracy": accuracy}
