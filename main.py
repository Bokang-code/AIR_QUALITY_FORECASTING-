from src.data_ingestion import run_ingestion
from src.preprocessing import run_preprocessing
from src.feature_engineering import main as run_feature_engineering
from src.train import main as run_training
from src.model_comparison import main as run_model_comparison


def main():
    print("=" * 80)
    print("AIR QUALITY FORECASTING PROJECT")
    print("SENSOR 218")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. DATA INGESTION
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("STEP 1: DATA INGESTION")
    print("=" * 80)

    run_ingestion()

    # -------------------------------------------------------------------------
    # 2. PREPROCESSING
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("STEP 2: PREPROCESSING")
    print("=" * 80)

    run_preprocessing()

    # -------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)

    run_feature_engineering()

    # -------------------------------------------------------------------------
    # 4. XGBOOST MULTI-HORIZON TRAINING
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("STEP 4: XGBOOST MULTI-HORIZON TRAINING")
    print("=" * 80)

    run_training()

    # -------------------------------------------------------------------------
    # 5. RANDOM FOREST VS XGBOOST COMPARISON
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("STEP 5: RANDOM FOREST VS XGBOOST COMPARISON")
    print("=" * 80)

    run_model_comparison()

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("ENTIRE AIR QUALITY FORECASTING PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()