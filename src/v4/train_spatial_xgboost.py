from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


# ============================================================
# V4-D: TEMPORAL + METEOROLOGY + CO-POLLUTANTS + SPATIAL
# ============================================================

INPUT_FILE = Path(
    "data/processed/v4/chengdu_features_v4_spatial.csv"
)

MODEL_DIR = Path(
    "models/v4/temporal_meteorology_copollutants_spatial"
)

REPORT_DIR = Path(
    "reports/v4/temporal_meteorology_copollutants_spatial"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FORECAST HORIZONS
# ============================================================

HORIZONS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


# ============================================================
# FEATURE GROUP 1: TEMPORAL
# ============================================================

TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# ============================================================
# FEATURE GROUP 2: PM2.5 HISTORY
# ============================================================

PM25_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
    336,
    720,
]

PM25_ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

PM25_FEATURES = []

for lag in PM25_LAGS:
    PM25_FEATURES.append(
        f"pm25_lag_{lag}h"
    )

for window in PM25_ROLLING_WINDOWS:
    PM25_FEATURES.append(
        f"pm25_rolling_mean_{window}h"
    )
    PM25_FEATURES.append(
        f"pm25_rolling_std_{window}h"
    )


# ============================================================
# FEATURE GROUP 3: METEOROLOGY
# ============================================================

METEOROLOGICAL_VARIABLES = [
    "temperature",
    "relative_humidity",
    "u10",
    "v10",
    "boundary_layer_height",
]

METEOROLOGY_FEATURES = []

for variable in METEOROLOGICAL_VARIABLES:

    METEOROLOGY_FEATURES.append(
        variable
    )

    for lag in [
        1,
        3,
        6,
        12,
        24,
    ]:

        METEOROLOGY_FEATURES.append(
            f"{variable}_lag_{lag}h"
        )

    for window in [
        6,
        12,
        24,
        48,
        72,
    ]:

        METEOROLOGY_FEATURES.append(
            f"{variable}_rolling_mean_{window}h"
        )


# Wind speed and direction features
METEOROLOGY_FEATURES += [
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
]


# ============================================================
# FEATURE GROUP 4: CO-POLLUTANTS
# ============================================================

CO_POLLUTANT_VARIABLES = [
    "pm10",
    "so2",
    "no2",
    "o3_8h",
    "co",
]

COPOLLUTANT_FEATURES = []

for variable in CO_POLLUTANT_VARIABLES:

    for lag in [
        1,
        3,
        6,
        12,
        24,
    ]:

        COPOLLUTANT_FEATURES.append(
            f"{variable}_lag_{lag}h"
        )

    for window in [
        6,
        12,
        24,
        48,
    ]:

        COPOLLUTANT_FEATURES.append(
            f"{variable}_rolling_mean_{window}h"
        )


# ============================================================
# FEATURE GROUP 5: SPATIAL
# ============================================================

SPATIAL_FEATURES = []

for neighbour_rank in [
    1,
    2,
    3,
]:

    for lag in [
        1,
        3,
        6,
        12,
        24,
    ]:

        SPATIAL_FEATURES.append(
            f"neighbour_{neighbour_rank}"
            f"_pm25_lag_{lag}h"
        )


# ============================================================
# COMBINE FEATURES
# ============================================================

FEATURE_GROUPS = {
    "temporal": TEMPORAL_FEATURES,
    "pm25_history": PM25_FEATURES,
    "meteorology": METEOROLOGY_FEATURES,
    "co_pollutants": COPOLLUTANT_FEATURES,
    "spatial": SPATIAL_FEATURES,
}

FEATURE_COLUMNS = (
    TEMPORAL_FEATURES
    + PM25_FEATURES
    + METEOROLOGY_FEATURES
    + COPOLLUTANT_FEATURES
    + SPATIAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading spatial V4 feature dataset..."
)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    [
        "station_code",
        "timestamp",
    ]
).reset_index(drop=True)


print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Total features: "
    f"{len(FEATURE_COLUMNS)}"
)

print()


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing required features:\n"
        + "\n".join(
            missing_features
        )
    )


print(
    "Feature groups:"
)

for group_name, features in FEATURE_GROUPS.items():

    print(
        f"  {group_name}: "
        f"{len(features)}"
    )

print()


# ============================================================
# XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ============================================================
# PURGED TIME SPLIT
# ============================================================

def create_purged_splits(
    data,
    target_column,
    horizon,
):

    # --------------------------------------------------------
    # Remove rows where target does not exist.
    # --------------------------------------------------------

    usable = data[
        data[target_column].notna()
    ].copy()


    # --------------------------------------------------------
    # Training:
    # timestamp < 2020-01-01 - horizon
    #
    # Validation:
    # 2020-01-01 <= timestamp
    # < 2021-01-01 - horizon
    #
    # Test:
    # timestamp >= 2021-01-01
    # --------------------------------------------------------

    train_cutoff = (
        pd.Timestamp("2020-01-01")
        - pd.Timedelta(
            hours=horizon
        )
    )

    validation_cutoff = (
        pd.Timestamp("2021-01-01")
        - pd.Timedelta(
            hours=horizon
        )
    )


    train = usable[
        usable["timestamp"]
        < train_cutoff
    ].copy()

    validation = usable[
        (
            usable["timestamp"]
            >= pd.Timestamp(
                "2020-01-01"
            )
        )
        &
        (
            usable["timestamp"]
            < validation_cutoff
        )
    ].copy()

    test = usable[
        usable["timestamp"]
        >= pd.Timestamp(
            "2021-01-01"
        )
    ].copy()


    return (
        train,
        validation,
        test,
    )


# ============================================================
# TRAIN MODELS
# ============================================================

all_results = []


for horizon_name, horizon in HORIZONS.items():

    print("=" * 70)

    print(
        f"V4-D: TRAINING {horizon_name}"
    )

    print("=" * 70)


    target_column = (
        f"target_pm25_{horizon_name}"
    )


    if target_column not in df.columns:

        raise ValueError(
            f"Target column not found: "
            f"{target_column}"
        )


    # --------------------------------------------------------
    # Create purged train / validation / test sets
    # --------------------------------------------------------

    train_df, validation_df, test_df = (
        create_purged_splits(
            df,
            target_column,
            horizon,
        )
    )


    print(
        f"Train samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples: "
        f"{len(test_df)}"
    )


    # --------------------------------------------------------
    # Features / targets
    # --------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        target_column
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        target_column
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        target_column
    ]


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()

    print(
        "Training XGBoost..."
    )

    model = XGBRegressor(
        **XGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_predictions = model.predict(
        X_train
    )

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    test_predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )


    print()

    print(
        "Validation:"
    )

    print(
        f"  MAE  : "
        f"{validation_metrics['mae']:.4f}"
    )

    print(
        f"  RMSE : "
        f"{validation_metrics['rmse']:.4f}"
    )

    print(
        f"  R²   : "
        f"{validation_metrics['r2']:.4f}"
    )


    print()

    print(
        "TEST:"
    )

    print(
        f"  MAE  : "
        f"{test_metrics['mae']:.4f}"
    )

    print(
        f"  RMSE : "
        f"{test_metrics['rmse']:.4f}"
    )

    print(
        f"  R²   : "
        f"{test_metrics['r2']:.4f}"
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR
        / (
            "xgboost_"
            "temporal_meteorology_"
            "copollutants_spatial_"
            f"{horizon_name}_purged.joblib"
        )
    )

    joblib.dump(
        model,
        model_file,
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_file = (
        REPORT_DIR
        / (
            "temporal_meteorology_"
            "copollutants_spatial_"
            f"{horizon_name}_purged_predictions.csv"
        )
    )


    prediction_df = test_df[
        [
            "timestamp",
            "station_code",
            "station_name",
            "pm25",
            target_column,
        ]
    ].copy()


    prediction_df[
        "prediction"
    ] = test_predictions


    prediction_df[
        "residual"
    ] = (
        prediction_df[
            target_column
        ]
        - prediction_df[
            "prediction"
        ]
    )


    prediction_df.to_csv(
        prediction_file,
        index=False,
    )


    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    all_results.append({
        "horizon": horizon_name,
        "horizon_hours": horizon,

        "feature_count": len(
            FEATURE_COLUMNS
        ),

        "train_samples": len(
            train_df
        ),

        "validation_samples": len(
            validation_df
        ),

        "test_samples": len(
            test_df
        ),

        "train_mae": train_metrics[
            "mae"
        ],

        "train_rmse": train_metrics[
            "rmse"
        ],

        "train_r2": train_metrics[
            "r2"
        ],

        "validation_mae": validation_metrics[
            "mae"
        ],

        "validation_rmse": validation_metrics[
            "rmse"
        ],

        "validation_r2": validation_metrics[
            "r2"
        ],

        "test_mae": test_metrics[
            "mae"
        ],

        "test_rmse": test_metrics[
            "rmse"
        ],

        "test_r2": test_metrics[
            "r2"
        ],
    })


    print()

    print(
        f"Model saved: {model_file}"
    )

    print(
        f"Predictions saved: "
        f"{prediction_file}"
    )

    print()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_file = (
    REPORT_DIR
    / (
        "temporal_meteorology_"
        "copollutants_spatial_"
        "xgboost_purged_results.csv"
    )
)

results_df.to_csv(
    results_file,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)

print(
    "V4-D TRAINING COMPLETE"
)

print("=" * 70)

print()

print(
    results_df[
        [
            "horizon",
            "test_mae",
            "test_rmse",
            "test_r2",
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    f"Results saved to: "
    f"{results_file}"
)
