from ast import List
from collections import Counter, defaultdict
from typing import Any, Union

import numpy as np
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer

from cdrc.common.metric.base import BaseMetric, MetricResult
from cdrc.common.utils import (
    binarize,
    get_nested_attribute,
    get_nested_attribute_list,
    get_nested_attribute_with_prefix,
    get_values_for_metrics_calculations,
)


class Accuracy(BaseMetric):
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
        instance_metrics = []
        for i, output in enumerate(outputs):
            gold_output = gold[i]
            correct = get_nested_attribute(output, self.label_key) == get_nested_attribute(gold_output, self.label_key)
            instance_metrics.append({"correct": correct})

        correct_values = [m["correct"] for m in instance_metrics]
        accuracy = np.mean(correct_values)
        aggregated_metric = {"accuracy": accuracy}

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result


class AccuracyMultiLabel(BaseMetric):
    """Concrete implementation of accuracy metric (supports multi-label and returns all modes)."""

    def __init__(self, label_key: str | list[str]):
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
        # obtain values for metrics calculations
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
        # binarize them
        mlb = MultiLabelBinarizer()
        mlb.fit(y_true + y_pred)
        y_true = mlb.transform(y_true)
        y_pred = mlb.transform(y_pred)
        # mapping = {i: label for i, label in enumerate(mlb.classes_)}

        # Strict/subset accuracy: all labels for a sample must match
        strict_accs = [np.array_equal(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        strict_accuracy = np.mean(strict_accs)

        # Macro accuracy: per-label (column) accuracy, then mean
        macro_accs = (y_true == y_pred).mean(axis=0)
        macro_accuracy = macro_accs.mean()

        # Micro accuracy: total correct labels divided by total labels
        micro_accuracy = (y_true == y_pred).mean()

        instance_metrics = []
        for i in range(len(y_pred)):
            instance_metrics.append(
                {
                    "strict_correct": int(np.array_equal(y_true[i], y_pred[i])),
                    "macro_correct": (y_true[i] == y_pred[i]).mean(),
                }
            )

        aggregated_metric = {
            "strict_accuracy": strict_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy": micro_accuracy,
        }

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result


class AccuracyCodeLevelMultiLabel(BaseMetric):
    """Concrete implementation of accuracy metric (supports multi-label and returns all modes)."""

    def __init__(self, label_key: str | list[str]):
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
        # obtain values for metrics calculations
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
        # mapping = {i: label for i, label in enumerate(mlb.classes_)}

        # Strict/subset accuracy: all labels for a sample must match
        strict_accs = [np.array_equal(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        strict_accuracy = np.mean(strict_accs)

        # Macro accuracy: per-label (column) accuracy, then mean
        macro_accs = (y_true == y_pred).mean(axis=0)
        macro_accuracy = macro_accs.mean()

        # Micro accuracy: total correct labels divided by total labels
        micro_accuracy = (y_true == y_pred).mean()

        instance_metrics = []
        for i in range(len(y_pred)):
            instance_metrics.append(
                {
                    "strict_correct": int(np.array_equal(y_true[i], y_pred[i])),
                    "macro_correct": (y_true[i] == y_pred[i]).mean(),
                }
            )

        aggregated_metric = {
            "strict_accuracy": strict_accuracy,
            "macro_accuracy": macro_accuracy,
            "micro_accuracy": micro_accuracy,
        }

        result: MetricResult = {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
        return result


class AccuracyPairedCode(BaseMetric):
    """
    Accuracy for multi-label predictions that may include duplicate labels.
    Uses 'paired' labels (e.g., DX#1, DX#2) per instance—same pattern as FScorePairedCode.
    Reports strict/subset, macro, and micro accuracies with both unweighted and weighted aggregates.
    """

    def __init__(self, label_key: Union[str, list[str]], prefix: str):
        super().__init__(supervised=True)
        self.label_key = [label_key] if isinstance(label_key, str) else label_key
        self.prefix = prefix

    def calculate(self, outputs: list[Any], gold: list[Any]) -> MetricResult:
        instance_metrics, weights = [], []
        metric_scores_all = defaultdict(list)
        for i, output in enumerate(outputs):
            # Build paired labels per instance (retain duplicates by numbering)
            y_true_tokens, y_pred_tokens = [], []
            gold_code_counts, pred_code_counts = Counter(), Counter()
            for k in self.label_key:
                _gold = get_nested_attribute_with_prefix(gold[i], k, self.prefix) or []
                _out = get_nested_attribute_with_prefix(output, k, self.prefix) or []
                for _label in _gold:
                    gold_code_counts[_label] += 1
                    y_true_tokens.append(f"{_label}#{gold_code_counts[_label]}")
                for _label in _out:
                    pred_code_counts[_label] += 1
                    y_pred_tokens.append(f"{_label}#{pred_code_counts[_label]}")
            # Weight = number of gold tokens in this instance (you can choose len(union) if preferred)
            weights.append(len(y_true_tokens))
            # Binarize with a shared mapping built from the union of the instance tokens
            _, mapping = binarize([y_true_tokens + y_pred_tokens])
            y_true_bin, _ = binarize([y_true_tokens], mapping)  # shape: (1, L)
            y_pred_bin, _ = binarize([y_pred_tokens], mapping)  # shape: (1, L)
            # Convert to 1D arrays for equality ops (keep as 2D then index [0] to reuse logic)
            y_t = y_true_bin[0]
            y_p = y_pred_bin[0]
            # Strict/subset accuracy (all labels match for the instance)
            strict_correct = int(np.array_equal(y_t, y_p))
            strict_accuracy = strict_correct  # per-instance
            # Macro accuracy: mean of per-label correctness for this instance
            if y_t.size > 0:
                macro_accuracy = (y_t == y_p).mean()
            else:
                macro_accuracy = 0.0  # or np.nan; keep consistent with your fscore defaulting
            # Micro accuracy (for a single instance, same as macro here, but kept for symmetry)
            micro_accuracy = macro_accuracy
            metric_scores = {
                "accuracy_strict": strict_accuracy,
                "accuracy_macro": macro_accuracy,
                "accuracy_micro": micro_accuracy,
            }
            # Collect per-instance for aggregation
            for k in metric_scores:
                metric_scores_all[k].append(metric_scores[k])
            instance_metrics.append(metric_scores)
        # Aggregation (unweighted and weighted by gold count)
        aggregated_metric = {}
        # Guard against all-zero weights
        total_weight = sum(weights) if len(weights) > 0 else 0
        for key in ["accuracy_strict", "accuracy_macro", "accuracy_micro"]:
            vals = metric_scores_all[key]
            aggregated_metric[key] = float(np.average(vals)) if len(vals) else 0.0
            if total_weight > 0:
                aggregated_metric[f"{key}_weighted"] = float(np.average(vals, weights=weights))
            else:
                aggregated_metric[f"{key}_weighted"] = aggregated_metric[key]
        return {
            "instance_metrics": instance_metrics,
            "aggregated_metric": aggregated_metric,
        }
