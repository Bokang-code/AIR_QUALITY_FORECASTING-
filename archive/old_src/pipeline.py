from pathlib import Path
from typing import Any

from src.train import main as train_models


def run_full_pipeline(
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete XGBoost multi-horizon forecasting pipeline.

    The current training pipeline produces:
        - 48-hour forecast
        - 72-hour forecast
        - 7-day forecast
        - 14-day forecast
        - 30-day forecast

    Models, predictions, reports, and evaluation results are
    saved by src.train.
    """

    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    else:
        base_dir = Path(base_dir)

    print("=" * 80)
    print("AIR QUALITY FORECASTING PIPELINE")
    print("=" * 80)

    print(f"\nProject root: {base_dir}")
    print("\nStarting XGBoost multi-horizon training...\n")

    train_models()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    return {
        "status": "completed",
        "project_root": str(base_dir),
        "model_directory": str(base_dir / "models"),
        "report_directory": str(base_dir / "outputs" / "reports"),
        "figure_directory": str(base_dir / "outputs" / "figures"),
    }


if __name__ == "__main__":
    run_full_pipeline()