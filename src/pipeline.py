from pathlib import Path
from typing import Any
import logging

from src.data_ingestion import ingest_air_quality_data
from src.evaluate import (
    evaluate_regression,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
)
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_air_quality_data
from src.train import prepare_training_data, train_and_save_model
from src.predict import load_model, predict_values
from src.xai import calculate_shap_values, plot_shap_summary, plot_shap_dependence

logger = logging.getLogger(__name__)


def run_full_pipeline(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the end-to-end pipeline and produce evaluation and XAI outputs.

    The function creates missing artifacts (ingest, preprocess, feature engineering,
    training) and saves reports and figures under `outputs/`.
    Returns a dictionary with paths and metrics.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    else:
        base_dir = Path(base_dir)

    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    outputs_reports = base_dir / "outputs" / "reports"
    outputs_figures = base_dir / "outputs" / "figures"

    raw_file = raw_dir / "air_quality_data.csv"
    if not raw_file.exists():
        logger.info("Raw data not found — ingesting from OpenAQ API.")
        ingest_air_quality_data(output_dir=raw_dir)

    processed_file = processed_dir / "processed_air_quality_data.csv"
    try:
        if not processed_file.exists():
            logger.info("Processed data not found — running preprocessing.")
            preprocess_air_quality_data(input_path=raw_file, output_dir=processed_dir)
    except Exception as exc:  # pragma: no cover - runtime condition
        logger.warning("Preprocessing failed: %s", exc)
        return {"error": "preprocessing_failed", "reason": str(exc)}

    engineered_file = processed_dir / "engineered_air_quality_data.csv"
    try:
        if not engineered_file.exists():
            logger.info("Engineered data not found — running feature engineering.")
            engineer_features(input_path=processed_file, output_dir=processed_dir)
    except Exception as exc:  # pragma: no cover - runtime condition
        logger.warning("Feature engineering failed: %s", exc)
        return {"error": "feature_engineering_failed", "reason": str(exc)}

    try:
        features, target = prepare_training_data(input_path=engineered_file, output_dir=processed_dir)
    except Exception as exc:  # pragma: no cover - runtime condition
        logger.warning("Preparing training data failed: %s", exc)
        return {"error": "prepare_training_failed", "reason": str(exc)}

    model_path = train_and_save_model(features, target, output_dir=models_dir)
    logger.info("Trained model saved to %s", model_path)

    model = load_model(model_path)
    predictions = predict_values(model, features)

    metrics = evaluate_regression(target, predictions, output_dir=outputs_reports)
    plot_actual_vs_predicted(target, predictions, output_dir=outputs_figures)
    plot_residuals(target, predictions, output_dir=outputs_figures)

    shap_info = None
    try:
        shap_values, feature_names = calculate_shap_values(model, features)
        plot_shap_summary(shap_values, feature_names, output_dir=outputs_figures)
        if "hour" in feature_names:
            plot_shap_dependence(shap_values, "hour", features, output_dir=outputs_figures)
        shap_info = {"feature_names": feature_names}
    except Exception as exc:
        logger.warning("SHAP explainability skipped: %s", exc)

    return {"model_path": str(model_path), "metrics": metrics, "shap": shap_info}
