import json
import logging
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_inputs(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
) -> tuple[pd.Series, pd.Series]:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    if y_true_series.empty or y_pred_series.empty:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true_series) != len(y_pred_series):
        raise ValueError("y_true and y_pred must have the same length.")

    return y_true_series, y_pred_series


def _default_report_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "reports"


def _default_figure_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "figures"


def evaluate_regression(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
    output_dir: str | Path | None = None,
    file_name: str = "evaluation_metrics.json",
) -> dict[str, float]:
    """Compute regression metrics and persist them to disk.

    Parameters
    ----------
    y_true:
        Actual target values.
    y_pred:
        Predicted target values.
    output_dir:
        Optional directory path for saving the metrics file.
    file_name:
        Optional output filename for the metrics JSON file.

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, RMSE, R², and MAPE.
    """
    logger.info("Evaluation started.")
    y_true_series, y_pred_series = _validate_inputs(y_true, y_pred)

    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for evaluation") from exc

    y_true_nonzero = y_true_series.replace(0, pd.NA).abs()
    percentage_errors = (y_true_series - y_pred_series).abs() / y_true_nonzero
    percentage_errors = percentage_errors.replace([pd.NA, pd.NaT], pd.NA).dropna()
    mape = float(percentage_errors.mean() * 100) if not percentage_errors.empty else 0.0

    metrics = {
        "mae": float(mean_absolute_error(y_true_series, y_pred_series)),
        "rmse": float(sqrt(mean_squared_error(y_true_series, y_pred_series))),
        "r2": float(r2_score(y_true_series, y_pred_series)),
        "mape": mape,
    }

    report_dir = Path(output_dir) if output_dir is not None else _default_report_directory()
    _ensure_directory(report_dir)

    output_path = report_dir / file_name
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Metrics calculated: %s", metrics)
    logger.info("Metrics saved to %s", output_path)
    logger.info("Evaluation completed.")
    return metrics


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Load a previously saved evaluation metrics report."""
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    logger.info("Loaded evaluation report from %s", report_path)
    return report


def plot_actual_vs_predicted(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
    output_dir: str | Path | None = None,
    file_name: str = "actual_vs_predicted.png",
) -> Path:
    """Plot actual versus predicted values and save the figure."""
    logger.info("Generating actual vs predicted plot.")
    y_true_series, y_pred_series = _validate_inputs(y_true, y_pred)

    figure_dir = Path(output_dir) if output_dir is not None else _default_figure_directory()
    _ensure_directory(figure_dir)

    min_value = min(y_true_series.min(), y_pred_series.min())
    max_value = max(y_true_series.max(), y_pred_series.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_series, y_pred_series, alpha=0.7, edgecolors="k", linewidths=0.5)
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=1)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.5)

    output_path = figure_dir / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Plot generated: %s", output_path)
    return output_path


def plot_residuals(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
    output_dir: str | Path | None = None,
    file_name: str = "residuals.png",
) -> Path:
    """Plot residuals against predicted values and save the figure."""
    logger.info("Generating residual plot.")
    y_true_series, y_pred_series = _validate_inputs(y_true, y_pred)

    figure_dir = Path(output_dir) if output_dir is not None else _default_figure_directory()
    _ensure_directory(figure_dir)

    residuals = y_true_series - y_pred_series

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_series, residuals, alpha=0.7, edgecolors="k", linewidths=0.5)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.5)

    output_path = figure_dir / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Plot generated: %s", output_path)
    return output_path


def plot_feature_importance(
    model: Any,
    feature_names: Sequence[str],
    output_dir: str | Path | None = None,
    file_name: str = "feature_importance.png",
) -> Path | None:
    """Plot feature importances for models that expose feature_importances_."""
    logger.info("Generating feature importance plot.")

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_. Skipping feature importance plot.")
        return None

    import numpy as np

    importances = getattr(model, "feature_importances_")
    if len(importances) != len(feature_names):
        raise ValueError("Feature names length must match model.feature_importances_.")

    sorted_indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    figure_dir = Path(output_dir) if output_dir is not None else _default_figure_directory()
    _ensure_directory(figure_dir)

    plt.figure(figsize=(10, max(4, len(feature_names) * 0.5)))
    plt.barh(sorted_names, sorted_importances, color="tab:blue")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    output_path = figure_dir / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Plot generated: %s", output_path)
    return output_path


def load_model(model_path: str | Path):
    """Load a serialized model from disk using joblib."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the model") from exc

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = joblib.load(model_file)
    logger.info("Loaded model from %s", model_file)
    return model
