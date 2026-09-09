from pathlib import Path

import pandas as pd

from xai import load_model, get_feature_importance


DATA_PATH = Path(
    "data/processed/v4/chengdu_features_v4.csv"
)

MODEL_PATH = Path(
    "models/v4/temporal_meteorology_copollutants/"
    "xgboost_temporal_meteorology_copollutants_24h_purged.joblib"
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape}")


# --------------------------------------------------
# Load trained V4-C model
# --------------------------------------------------

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Get EXACT features used by the trained model
# --------------------------------------------------

FEATURE_COLUMNS = model.get_booster().feature_names

print(f"Number of model features: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# Check that every model feature exists in dataset
# --------------------------------------------------

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Model features missing from dataset:")
    for feature in missing_features:
        print(f"  - {feature}")

    raise ValueError(
        "The dataset does not contain all features required by the model."
    )


# --------------------------------------------------
# Select rows with valid 24-hour targets
# --------------------------------------------------

df_valid = df[
    df["target_pm25_24h"].notna()
].copy()


# --------------------------------------------------
# Sample observations for SHAP
# --------------------------------------------------

X = df_valid[FEATURE_COLUMNS].sample(
    n=min(1000, len(df_valid)),
    random_state=42
)

print(f"Rows used for SHAP: {len(X)}")


# --------------------------------------------------
# Generate SHAP feature importance
# --------------------------------------------------

importance = get_feature_importance(
    model,
    X
)


# --------------------------------------------------
# Display top 20 features
# --------------------------------------------------

print("\nTOP 20 FEATURES BY SHAP IMPORTANCE")
print("=" * 60)

print(
    importance.head(20).to_string(index=False)
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

output_path = Path(
    "reports/v4/xai/v4c_24h_shap_importance.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

importance.to_csv(
    output_path,
    index=False
)

print("\nSHAP results saved to:")
print(output_path)