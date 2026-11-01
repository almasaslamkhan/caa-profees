import json
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import binarize, get_nested_attribute, get_nested_attribute_list, get_values_for_metrics_calculations, is_listed_attribute

LABEL_WEIGHT_PATH = {
    "em.code": "cdrc/common/metric/priors/em_codes.json",
    "procedure.code": "cdrc/common/metric/priors/procedure_codes.json",
    "modifiers": "cdrc/common/metric/priors/modifiers.json",
    "diagnosis": "cdrc/common/metric/priors/icd-10_codes.json",
    "hcpcs": "cdrc/common/metric/priors/hcpcs_codes.json",
}


def load_label_weights(label_key: str, weight_by: str) -> dict:
    if label_key == "em.code":
        path = LABEL_WEIGHT_PATH["em.code"]
    elif label_key == "procedure.code":
        path = LABEL_WEIGHT_PATH["procedure.code"]
    elif "hcpcs" in label_key:
        path = LABEL_WEIGHT_PATH["hcpcs"]
    elif "modifiers" in label_key:
        path = LABEL_WEIGHT_PATH["modifiers"]
    elif "diagnosis" in label_key:
        path = LABEL_WEIGHT_PATH["diagnosis"]

    with open(path, "r") as f:
        weight_dict = json.load(f)

    if weight_by not in weight_dict:
        raise ValueError(f"Weight dict provider {weight_by} not found in weight dict")

    return weight_dict[weight_by]


def _safe_f1(y_true, y_pred, average, zero_division=0):
    """Compute F1 with a safe fallback if 'samples' is invalid."""
    try:
        return f1_score(y_true, y_pred, average=average, zero_division=zero_division)
    except ValueError:
        # Fallback: for non-multilabel cases, 'samples' is invalid.
        # Use 'micro' as a reasonable per-sample/overall fallback.
        if average == "samples":
            return None
        raise  # other ValueErrors should surface


def compute_weighted_metrics(y_true, y_pred, label_classes, label_weight_dict, weight_epsilon=0.0):
    """
    Compute weighted F1 score for multi-label classification
    with custom label weights (subset-safe).

    Parameters:
    - y_true: np.ndarray (n_samples, n_labels)
    - y_pred: np.ndarray (n_samples, n_labels)
    - label_classes: list of labels in the same order as y_true/y_pred columns
    - label_weight_dict: dict {label: weight}
    - weight_epsilon: float, default weight for labels not in label_weight_dict

    Returns:
    - weighted_f1: float
    - per_label_f1: dict {label: f1_value}
    - used_weights: dict {label: normalized_weight}
    """
    # Compute per-label F1 scores
    per_label_f1_values = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label_precision_values = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_label_recall_values = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Build a map for presented labels
    presented_labels = label_classes
    weights = []

    for label in presented_labels:
        # Default to 0 if not found in dict
        weights.append(label_weight_dict.get(label, weight_epsilon))

    weights = np.array(weights)

    # Only keep labels with non-zero weights or that appear in data
    active_mask = weights > 0
    if not np.any(active_mask):
        raise ValueError("No active labels with non-zero weights found among presented labels.")

    per_label_f1_values = per_label_f1_values[active_mask]
    per_label_precision_values = per_label_precision_values[active_mask]
    per_label_recall_values = per_label_recall_values[active_mask]
    weights = weights[active_mask]
    labels_used = np.array(presented_labels)[active_mask]

    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    # Weighted F1
    weighted_f1 = np.sum(per_label_f1_values * weights)

    # Weighted Precision
    weighted_precision = np.sum(per_label_precision_values * weights)

    # Weighted Recall
    weighted_recall = np.sum(per_label_recall_values * weights)

    return {
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "per_label_f1": {label: f1 for label, f1 in zip(labels_used, per_label_f1_values)},
        "used_weights": {label: weight for label, weight in zip(labels_used, weights)},
    }


class FScore(BaseMetric):
    """Concrete implementation of accuracy metric."""

    def __init__(self, label_key: str):
        """Initialize with supervised flag.

        Args:
            label_key (str): The label key which is going to be used for comparing the fields.
                             It MUST point to the location of label in the schema.
                             For example, 'cpt.code'.
        """
        super().__init__(supervised=True)
        self.label_key = label_key

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        """Calculate metrics for outputs comparing to gold data.

        Args:
            outputs (list[Any]): The model outputs.
            gold (Optional[list[Any]]): The gold standard outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """

        if not is_listed_attribute(type(outputs[0]), self.label_key):
            true = []
            pred = []
            for i, output in enumerate(outputs):
                true.append(get_nested_attribute(gold[i], self.label_key))
                pred.append(get_nested_attribute(output, self.label_key))

            confusion = confusion_matrix(true, pred)
            report = classification_report(true, pred, output_dict=True, zero_division=0)

            result: MetricResult = {
                "aggregated_metric": {
                    "confusion": confusion,
                    "macro avg": report.pop("macro avg"),
                    "classification_report": report,
                },
            }
            return result

        else:
            base_label_key = self.label_key.split(".")[0]
            sub_label_key = ".".join(self.label_key.split(".")[1:])
            instance_metrics = []
            for i, output in enumerate(outputs):
                true = [get_nested_attribute(obj, sub_label_key) for obj in getattr(gold[i], base_label_key)]
                pred = [get_nested_attribute(obj, sub_label_key) for obj in getattr(output, base_label_key)]
                confusion = confusion_matrix(true, pred)
                report = classification_report(true, pred, output_dict=True, zero_division=0)

                instance_metrics.append(
                    {
                        "confusion": confusion,
                        "classification_report": report,
                    }
                )

            result: MetricResult = {
                "instance_metrics": instance_metrics,
            }

            return result


class FScoreMultilabel(BaseMetric):
    """Concrete implementation of F1-score metric (multi-label, reports all three F1 modes)."""

    def __init__(self, label_key: str | list[str], weight_by: Optional[Union[str, None]] = None):
        super().__init__(supervised=True)
        self.label_key = label_key
        self.weight_by = weight_by

        if weight_by:
            self.label_weights = load_label_weights(self.label_key, weight_by)

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        """Calculate metrics for outputs comparing to gold data.

        Args:
            outputs (list[Any]): The model outputs.
            gold (Optional[list[Any]]): The gold standard outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """
        # Extract per-sample label-lists (list of lists)
        y_true = []
        y_pred = []
        for i, output in enumerate(outputs):
            if isinstance(self.label_key, (list, tuple)):
                for lk in self.label_key:
                    y_true.append(get_nested_attribute(gold[i], lk) or [])
                    y_pred.append(get_nested_attribute(output, lk) or [])
            else:
                y_true.append(get_nested_attribute(gold[i], self.label_key) or [])
                y_pred.append(get_nested_attribute(output, self.label_key) or [])

        # binarize
        mlb = MultiLabelBinarizer()
        mlb.fit(y_true + y_pred)
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        mapping = {i: label for i, label in enumerate(mlb.classes_)}

        # ---- Guard for "no classes" case (e.g., everyone empty) ----
        def _no_classes(arr: np.ndarray) -> bool:
            return not isinstance(arr, np.ndarray) or arr.size == 0 or (arr.ndim == 2 and arr.shape[1] == 0)

        if _no_classes(y_true) and _no_classes(y_pred):
            n = len(outputs)
            instance_metrics = [
                {
                    "f1_sample": None,
                    "f1_macro_per_sample": None,
                    "f1_micro_per_sample": None,
                    "classification_report": {},
                }
                for _ in range(n)
            ]
            aggregated_metric = {
                "f1_micro": None,
                "f1_macro": None,
                "f1_samples": None,
                "classification_report": {},
                "label_mapping": mapping,
                "support": 0,
                "all_empty": True,
            }

            if self.weight_by:
                aggregated_metric.update(
                    {
                        "weighted_f1": None,
                        "weighted_precision": None,
                        "weighted_recall": None,
                        "per_label_f1": {},
                        "used_weights": {},
                    }
                )

            return {
                "instance_metrics": instance_metrics,
                "aggregated_metric": aggregated_metric,
            }
        else:
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_samples = _safe_f1(y_true, y_pred, average="samples", zero_division=0)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            # Per-sample metrics (e.g. macro f1 per sample)
            instance_metrics = [
                {
                    "f1_sample": _safe_f1([y_true[i]], [y_pred[i]], average="samples", zero_division=0),
                    "f1_macro_per_sample": f1_score([y_true[i]], [y_pred[i]], average="macro", zero_division=0),
                    "f1_micro_per_sample": f1_score([y_true[i]], [y_pred[i]], average="micro", zero_division=0),
                    "classification_report": classification_report([y_true[i]], [y_pred[i]], output_dict=True, zero_division=0),
                }
                for i in range(len(y_true))
            ]

        aggregated_metric = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_samples": f1_samples,
            "classification_report": report,
            "label_mapping": mapping,  # Helpful for interpreting per-label f1
        }

        if self.weight_by:
            weighted_metrics = compute_weighted_metrics(y_true, y_pred, label_classes=mlb.classes_, label_weight_dict=self.label_weights)
            aggregated_metric.update(
                {
                    "weighted_f1": weighted_metrics["weighted_f1"],
                    "weighted_precision": weighted_metrics["weighted_precision"],
                    "weighted_recall": weighted_metrics["weighted_recall"],
                    "per_label_f1": weighted_metrics["per_label_f1"],
                    "used_weights": weighted_metrics["used_weights"],
                }
            )

        return {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }


class FScoreCodeLevelMultilabel(BaseMetric):
    """Concrete implementation of F1-score metric (multi-label, reports all three F1 modes)."""

    def __init__(self, label_key: str | list[str]):
        super().__init__(supervised=True)
        self.label_key = label_key

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        """Calculate metrics for outputs comparing to gold data.

        Args:
            outputs (list[Any]): The model outputs.
            gold (Optional[list[Any]]): The gold standard outputs.

        Returns:
            MetricResult: A dictionary containing instance metrics and aggregated metrics.
        """
        # Extract per-sample label-lists (list of lists)
        y_true = []
        y_pred = []
        for i, output in enumerate(outputs):
            if isinstance(self.label_key, (list, tuple)):
                for lk in self.label_key:
                    y_true.extend(get_nested_attribute_list(gold[i], lk) or [])
                    y_pred.extend(get_nested_attribute_list(output, lk) or [])
            else:
                y_true.extend(get_nested_attribute_list(gold[i], self.label_key) or [])
                y_pred.extend(get_nested_attribute_list(output, self.label_key) or [])

        # binarize them
        mlb = MultiLabelBinarizer()
        mlb.fit(y_true + y_pred)
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        mapping = {i: label for i, label in enumerate(mlb.classes_)}

        # ---- Guard for "no classes" case (e.g., everyone empty) ----
        def _no_classes(arr: np.ndarray) -> bool:
            return not isinstance(arr, np.ndarray) or arr.size == 0 or (arr.ndim == 2 and arr.shape[1] == 0)

        if _no_classes(y_true) and _no_classes(y_pred):
            n = len(outputs)
            instance_metrics = [
                {
                    "f1_sample": None,
                    "f1_macro_per_sample": None,
                    "f1_micro_per_sample": None,
                    "classification_report": {},
                }
                for _ in range(n)
            ]
            aggregated_metric = {
                "f1_micro": None,
                "f1_macro": None,
                "f1_samples": None,
                "classification_report": {},
                "label_mapping": mapping,
                "support": 0,
                "all_empty": True,
            }
            return {
                "instance_metrics": instance_metrics,
                "aggregated_metric": aggregated_metric,
            }
        else:
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_samples = _safe_f1(y_true, y_pred, average="samples", zero_division=0)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            # Per-sample metrics (e.g. macro f1 per sample)
            instance_metrics = [
                {
                    "f1_sample": _safe_f1([y_true[i]], [y_pred[i]], average="samples", zero_division=0),
                    "f1_macro_per_sample": f1_score([y_true[i]], [y_pred[i]], average="macro", zero_division=0),
                    "f1_micro_per_sample": f1_score([y_true[i]], [y_pred[i]], average="micro", zero_division=0),
                    "classification_report": classification_report([y_true[i]], [y_pred[i]], output_dict=True, zero_division=0),
                }
                for i in range(len(y_true))
            ]

        aggregated_metric = {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_samples": f1_samples,
            "classification_report": report,
            "label_mapping": mapping,  # Helpful for interpreting per-label f1
        }

        return {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
