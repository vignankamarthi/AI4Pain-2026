"""AI4Pain 2026 metrics suite.

ANTIPATTERNS rule 7: every fitness vector reports the full suite. No
single-number summary is ever sufficient on a 3-class imbalanced problem.

Returns:
  balanced_acc, macro_f1, per_class_pr (dict of (precision, recall) per class),
  confusion_3x3 (3x3 list of lists), auc_ovr (one-vs-rest), ece (calibration),
  param_count is provided externally and merged at the call site.
"""
import numpy as np


def full_metric_suite(y_true: np.ndarray, y_pred: np.ndarray,
                      proba: np.ndarray) -> dict:
    """Compute the entire metrics suite.

    Args:
        y_true: shape (n,) int labels in {0, 1, 2}
        y_pred: shape (n,) int predictions in {0, 1, 2}
        proba: shape (n, 3) predicted probabilities

    Returns:
        dict with all keys listed in the module docstring.
    """
    raise NotImplementedError
