import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENGINEERED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "engineered_sensor_218.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "sensor_218_random_forest.pkl"
)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

def load_data():

    df = pd.read_csv(ENGINEERED_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def load_model():

    model_data = joblib.load(
        MODEL_FILE
    )

    return (
        model_data["model"],
        model_data["features"]
    )


# ============================================================
# PREPARE EXACT TEST SET
# ============================================================

def prepare_test_data(
    df,
    feature_columns
):

    target_column = "target_pm25"

    modelling_df = df.dropna(
        subset=[target_column]
    ).copy()

    modelling_df[feature_columns] = (
        modelling_df[feature_columns].ffill()
    )

    modelling_df = modelling_df.dropna(
        subset=feature_columns
    ).copy()

    modelling_df = modelling_df.reset_index(
        drop=True
    )

    total = len(modelling_df)

    train_end = int(total * 0.70)

    validation_end = int(total * 0.85)

    test_df = modelling_df.iloc[
        validation_end:
    ].copy()

    return test_df


# ============================================================
# CREATE COMPARISON DATASET
# ============================================================

def create_comparison_dataset(
    df,
    model,
    feature_columns
):

    test_df = prepare_test_data(
        df,
        feature_columns
    )

    X_test = test_df[
        feature_columns
    ]

    test_df["random_forest"] = model.predict(
        X_test
    )

    # Naive forecast:
    # next-hour PM2.5 = current-hour PM2.5

    raw_df = df[
        [
            "timestamp",
            "pm25"
        ]
    ].copy()

    raw_df["naive_prediction"] = (
        raw_df["pm25"]
    )

    comparison_df = test_df[
        [
            "timestamp",
            "target_pm25",
            "random_forest"
        ]
    ].merge(
        raw_df[
            [
                "timestamp",
                "naive_prediction"
            ]
        ],
        on="timestamp",
        how="left"
    )

    comparison_df = comparison_df.dropna(
        subset=[
            "target_pm25",
            "random_forest",
            "naive_prediction"
        ]
    )

    comparison_df = comparison_df.rename(
        columns={
            "target_pm25": "actual"
        }
    )

    comparison_df["rf_error"] = (
        comparison_df["actual"]
        - comparison_df["random_forest"]
    )

    comparison_df["naive_error"] = (
        comparison_df["actual"]
        - comparison_df["naive_prediction"]
    )

    comparison_df["rf_absolute_error"] = (
        comparison_df["rf_error"].abs()
    )

    comparison_df["naive_absolute_error"] = (
        comparison_df["naive_error"].abs()
    )

    comparison_df["rf_squared_error"] = (
        comparison_df["rf_error"] ** 2
    )

    comparison_df["naive_squared_error"] = (
        comparison_df["naive_error"] ** 2
    )

    comparison_df["rf_better"] = (
        comparison_df["rf_absolute_error"]
        < comparison_df["naive_absolute_error"]
    )

    comparison_df["actual_range"] = pd.cut(
        comparison_df["actual"],
        bins=[
            -float("inf"),
            25,
            50,
            100,
            250,
            float("inf")
        ],
        labels=[
            "0-25",
            "25-50",
            "50-100",
            "100-250",
            "250+"
        ]
    )

    return comparison_df


# ============================================================
# OVERALL ERROR ANALYSIS
# ============================================================

def calculate_overall_results(
    comparison_df
):

    actual = comparison_df["actual"]

    rf = comparison_df["random_forest"]

    naive = comparison_df["naive_prediction"]

    rf_mae = mean_absolute_error(
        actual,
        rf
    )

    rf_rmse = mean_squared_error(
        actual,
        rf
    ) ** 0.5

    rf_r2 = r2_score(
        actual,
        rf
    )

    naive_mae = mean_absolute_error(
        actual,
        naive
    )

    naive_rmse = mean_squared_error(
        actual,
        naive
    ) ** 0.5

    naive_r2 = r2_score(
        actual,
        naive
    )

    rf_better_count = comparison_df[
        "rf_better"
    ].sum()

    total = len(comparison_df)

    return {
        "test_rows": int(total),

        "random_forest": {
            "mae": float(rf_mae),
            "rmse": float(rf_rmse),
            "r2": float(rf_r2)
        },

        "naive_baseline": {
            "mae": float(naive_mae),
            "rmse": float(naive_rmse),
            "r2": float(naive_r2)
        },

        "random_forest_better_cases": int(
            rf_better_count
        ),

        "random_forest_better_percentage": float(
            rf_better_count / total * 100
        )
    }


# ============================================================
# PERFORMANCE BY PM2.5 RANGE
# ============================================================

def analyse_by_pollution_range(
    comparison_df
):

    results = []

    for pollution_range, group in comparison_df.groupby(
        "actual_range",
        observed=False
    ):

        if len(group) == 0:
            continue

        actual = group["actual"]

        rf = group["random_forest"]

        naive = group["naive_prediction"]

        rf_mae = mean_absolute_error(
            actual,
            rf
        )

        naive_mae = mean_absolute_error(
            actual,
            naive
        )

        rf_rmse = mean_squared_error(
            actual,
            rf
        ) ** 0.5

        naive_rmse = mean_squared_error(
            actual,
            naive
        ) ** 0.5

        results.append(
            {
                "pm25_range": str(
                    pollution_range
                ),
                "rows": int(len(group)),
                "rf_mae": float(rf_mae),
                "naive_mae": float(naive_mae),
                "rf_rmse": float(rf_rmse),
                "naive_rmse": float(naive_rmse),
                "rf_better_percentage": float(
                    group["rf_better"].mean() * 100
                )
            }
        )

    return pd.DataFrame(results)


# ============================================================
# EXTREME EVENTS
# ============================================================

def analyse_extreme_events(
    comparison_df
):

    extreme = comparison_df[
        comparison_df["actual"] > 250
    ].copy()

    high = comparison_df[
        comparison_df["actual"] > 100
    ].copy()

    results = {
        "values_above_100": int(len(high)),
        "values_above_250": int(len(extreme))
    }

    if len(high) > 0:

        results["above_100"] = {
            "rf_mae": float(
                mean_absolute_error(
                    high["actual"],
                    high["random_forest"]
                )
            ),
            "naive_mae": float(
                mean_absolute_error(
                    high["actual"],
                    high["naive_prediction"]
                )
            )
        }

    if len(extreme) > 0:

        results["above_250"] = {
            "rf_mae": float(
                mean_absolute_error(
                    extreme["actual"],
                    extreme["random_forest"]
                )
            ),
            "naive_mae": float(
                mean_absolute_error(
                    extreme["actual"],
                    extreme["naive_prediction"]
                )
            )
        }

    return results


# ============================================================
# LARGEST ERRORS
# ============================================================

def find_largest_errors(
    comparison_df,
    number_of_rows=10
):

    largest = comparison_df.sort_values(
        "rf_absolute_error",
        ascending=False
    ).head(
        number_of_rows
    ).copy()

    return largest[
        [
            "timestamp",
            "actual",
            "random_forest",
            "naive_prediction",
            "rf_error",
            "naive_error",
            "rf_absolute_error",
            "naive_absolute_error"
        ]
    ]


# ============================================================
# BIAS ANALYSIS
# ============================================================

def calculate_bias(
    comparison_df
):

    rf_error = comparison_df[
        "rf_error"
    ]

    naive_error = comparison_df[
        "naive_error"
    ]

    return {
        "random_forest_mean_error": float(
            rf_error.mean()
        ),

        "random_forest_mean_absolute_error": float(
            rf_error.abs().mean()
        ),

        "naive_mean_error": float(
            naive_error.mean()
        ),

        "random_forest_underprediction_rate": float(
            (rf_error > 0).mean() * 100
        ),

        "random_forest_overprediction_rate": float(
            (rf_error < 0).mean() * 100
        )
    }


# ============================================================
# PLOT ERROR BY POLLUTION RANGE
# ============================================================

def plot_range_comparison(
    range_results
):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    x = range_results[
        "pm25_range"
    ]

    rf_mae = range_results[
        "rf_mae"
    ]

    naive_mae = range_results[
        "naive_mae"
    ]

    positions = range(
        len(x)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        [p - width / 2 for p in positions],
        rf_mae,
        width=width,
        label="Random Forest"
    )

    plt.bar(
        [p + width / 2 for p in positions],
        naive_mae,
        width=width,
        label="Naive Baseline"
    )

    plt.xticks(
        positions,
        x
    )

    plt.xlabel(
        "Actual PM2.5 Range (µg/m³)"
    )

    plt.ylabel(
        "MAE (µg/m³)"
    )

    plt.title(
        "Model Performance by PM2.5 Concentration"
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    output_file = (
        FIGURE_DIR
        / "sensor_218_error_by_pollution_range.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# PLOT PREDICTION BIAS
# ============================================================

def plot_prediction_bias(
    comparison_df
):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        comparison_df["actual"],
        comparison_df["random_forest"],
        alpha=0.5,
        s=20
    )

    minimum = min(
        comparison_df["actual"].min(),
        comparison_df["random_forest"].min()
    )

    maximum = max(
        comparison_df["actual"].max(),
        comparison_df["random_forest"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Actual PM2.5 (µg/m³)"
    )

    plt.ylabel(
        "Random Forest Prediction (µg/m³)"
    )

    plt.title(
        "Random Forest Prediction Bias"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    output_file = (
        FIGURE_DIR
        / "sensor_218_prediction_bias.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# SAVE ERROR ANALYSIS REPORT
# ============================================================

def save_report(
    overall,
    range_results,
    extreme_results,
    bias_results,
    largest_errors
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "overall": overall,
        "performance_by_pollution_range":
            range_results.to_dict(
                orient="records"
            ),
        "extreme_events": extreme_results,
        "bias_analysis": bias_results,
        "largest_random_forest_errors":
            largest_errors.assign(
                timestamp=lambda x:
                    x["timestamp"].astype(str)
            ).to_dict(
                orient="records"
            )
    }

    output_file = (
        REPORT_DIR
        / "sensor_218_error_analysis.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    overall,
    range_results,
    extreme_results,
    bias_results,
    largest_errors
):

    print("\n")
    print("=" * 80)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 80)

    print(
        f"Test rows analysed: "
        f"{overall['test_rows']:,}"
    )

    print(
        f"Random Forest MAE: "
        f"{overall['random_forest']['mae']:.4f}"
    )

    print(
        f"Naive Baseline MAE: "
        f"{overall['naive_baseline']['mae']:.4f}"
    )

    print(
        f"Random Forest better in: "
        f"{overall['random_forest_better_percentage']:.2f}% "
        f"of cases"
    )

    print("\n")
    print("=" * 80)
    print("PERFORMANCE BY PM2.5 RANGE")
    print("=" * 80)

    print(
        range_results.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("EXTREME EVENT ANALYSIS")
    print("=" * 80)

    print(
        f"Values above 100 µg/m³: "
        f"{extreme_results['values_above_100']}"
    )

    print(
        f"Values above 250 µg/m³: "
        f"{extreme_results['values_above_250']}"
    )

    if "above_100" in extreme_results:

        print(
            f"\nAbove 100 µg/m³:"
        )

        print(
            f"Random Forest MAE: "
            f"{extreme_results['above_100']['rf_mae']:.4f}"
        )

        print(
            f"Naive Baseline MAE: "
            f"{extreme_results['above_100']['naive_mae']:.4f}"
        )

    if "above_250" in extreme_results:

        print(
            f"\nAbove 250 µg/m³:"
        )

        print(
            f"Random Forest MAE: "
            f"{extreme_results['above_250']['rf_mae']:.4f}"
        )

        print(
            f"Naive Baseline MAE: "
            f"{extreme_results['above_250']['naive_mae']:.4f}"
        )

    print("\n")
    print("=" * 80)
    print("PREDICTION BIAS")
    print("=" * 80)

    print(
        f"Random Forest mean error: "
        f"{bias_results['random_forest_mean_error']:.4f}"
    )

    print(
        f"Random Forest underprediction rate: "
        f"{bias_results['random_forest_underprediction_rate']:.2f}%"
    )

    print(
        f"Random Forest overprediction rate: "
        f"{bias_results['random_forest_overprediction_rate']:.2f}%"
    )

    print("\n")
    print("=" * 80)
    print("LARGEST RANDOM FOREST ERRORS")
    print("=" * 80)

    print(
        largest_errors.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SENSOR 218 ERROR ANALYSIS")
    print("=" * 80)

    df = load_data()

    model, feature_columns = load_model()

    comparison_df = create_comparison_dataset(
        df,
        model,
        feature_columns
    )

    overall = calculate_overall_results(
        comparison_df
    )

    range_results = analyse_by_pollution_range(
        comparison_df
    )

    extreme_results = analyse_extreme_events(
        comparison_df
    )

    bias_results = calculate_bias(
        comparison_df
    )

    largest_errors = find_largest_errors(
        comparison_df
    )

    print_results(
        overall,
        range_results,
        extreme_results,
        bias_results,
        largest_errors
    )

    plot_range_comparison(
        range_results
    )

    plot_prediction_bias(
        comparison_df
    )

    save_report(
        overall,
        range_results,
        extreme_results,
        bias_results,
        largest_errors
    )

    print("\n")
    print("=" * 80)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()