import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Standard metric for ordinal DR grading (Kaggle / APTOS)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def summarize(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
    }
