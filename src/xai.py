import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_features(
    model: Any,
    features: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if features.empty:
        raise ValueError("Feature data must not be empty.")

    if feature_names is not None and len(feature_names) != len(features.columns):
        raise ValueError("feature_names must match the number of feature columns.")

    validated_feature_names = list(feature_names) if feature_names is not None else list(features.columns)
    return features, validated_feature_names


def calculate_shap_values(
    model: Any,
    features: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Calculate SHAP values for a trained model and feature dataset."""
    logger.info("Calculating SHAP values.")

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("The SHAP library is required for explainability") from exc

    features, validated_feature_names = _validate_features(model, features, feature_names)

    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)

    logger.info("SHAP values calculated.")
    return shap_values, validated_feature_names


def plot_shap_summary(
    shap_values: Any,
    feature_names: Sequence[str],
    output_dir: str | Path | None = None,
    file_name: str = "shap_summary.png",
) -> Path:
    """Create and save a SHAP summary plot."""
    logger.info("Generating SHAP summary plot.")

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("The SHAP library is required for plotting") from exc

    figure_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "outputs" / "figures"
    _ensure_directory(figure_dir)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, feature_names=list(feature_names), show=False)

    output_path = figure_dir / file_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("SHAP summary plot saved to %s", output_path)
    return output_path


def plot_shap_dependence(
    shap_values: Any,
    feature: str,
    features: pd.DataFrame,
    output_dir: str | Path | None = None,
    file_name: str | None = None,
) -> Path:
    """Create and save a SHAP dependence plot for a single feature."""
    logger.info("Generating SHAP dependence plot for feature %s.", feature)

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("The SHAP library is required for plotting") from exc

    if feature not in features.columns:
        raise ValueError(f"Feature '{feature}' was not found in the feature dataset.")

    figure_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "outputs" / "figures"
    _ensure_directory(figure_dir)
    plot_name = file_name if file_name is not None else f"shap_dependence_{feature}.png"

    plt.figure(figsize=(10, 7))
    shap.dependence_plot(feature, shap_values.values, features, show=False)

    output_path = figure_dir / plot_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("SHAP dependence plot saved to %s", output_path)
    return output_path
