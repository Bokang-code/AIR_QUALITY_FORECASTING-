from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# V4-D: BUILD SPATIAL FEATURES
# ============================================================

INPUT_FILE = Path(
    "data/processed/v4/chengdu_features_v4.csv"
)

OUTPUT_FILE = Path(
    "data/processed/v4/chengdu_features_v4_spatial.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of nearest neighbouring stations used for each
# target station.
N_NEIGHBOURS = 3


# Historical PM2.5 information from neighbouring stations.
NEIGHBOUR_LAGS = [
    1,
    3,
    6,
    12,
    24,
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V4 feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    ["station_code", "timestamp"]
).reset_index(drop=True)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Number of stations: "
    f"{df['station_code'].nunique()}"
)

print()


# ============================================================
# STATION INFORMATION
# ============================================================

station_info = (
    df[
        [
            "station_code",
            "station_name",
            "longitude",
            "latitude",
        ]
    ]
    .drop_duplicates(
        subset=["station_code"]
    )
    .reset_index(drop=True)
)


print("Stations:")

print(
    station_info.to_string(
        index=False
    )
)

print()


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two geographic coordinates
    using the Haversine formula.

    Returns distance in kilometres.
    """

    earth_radius_km = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(
        lat2 - lat1
    )

    delta_lon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(delta_lat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = (
        2
        * np.arcsin(
            np.sqrt(a)
        )
    )

    return (
        earth_radius_km * c
    )


# ============================================================
# FIND NEAREST STATIONS
# ============================================================

print(
    "Finding nearest neighbouring stations..."
)

neighbour_map = {}


for _, target in station_info.iterrows():

    target_code = target[
        "station_code"
    ]

    distances = []

    for _, candidate in station_info.iterrows():

        candidate_code = candidate[
            "station_code"
        ]

        # Don't select the target station itself.
        if candidate_code == target_code:
            continue

        distance = haversine_distance(
            target["latitude"],
            target["longitude"],
            candidate["latitude"],
            candidate["longitude"],
        )

        distances.append(
            (
                candidate_code,
                distance,
            )
        )

    distances.sort(
        key=lambda x: x[1]
    )

    neighbour_map[
        target_code
    ] = distances[
        :N_NEIGHBOURS
    ]

# ============================================================
# SAVE NEAREST-STATION MAPPING
# ============================================================

SPATIAL_REPORT_DIR = Path(
    "reports/v4/spatial"
)

SPATIAL_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

mapping_rows = []

for target_code, neighbours in neighbour_map.items():

    target_name = station_info.loc[
        station_info["station_code"] == target_code,
        "station_name",
    ].iloc[0]

    for rank, (
        neighbour_code,
        distance
    ) in enumerate(
        neighbours,
        start=1
    ):

        neighbour_name = station_info.loc[
            station_info["station_code"] == neighbour_code,
            "station_name",
        ].iloc[0]

        mapping_rows.append({
            "target_station_code": target_code,
            "target_station_name": target_name,
            "neighbour_rank": rank,
            "neighbour_station_code": neighbour_code,
            "neighbour_station_name": neighbour_name,
            "distance_km": round(distance, 4),
        })


mapping_df = pd.DataFrame(
    mapping_rows
)

mapping_file = (
    SPATIAL_REPORT_DIR
    / "nearest_station_mapping.csv"
)

mapping_df.to_csv(
    mapping_file,
    index=False
)

print(
    f"Nearest-station mapping saved to: "
    f"{mapping_file}"
)

# ============================================================
# PRINT NEIGHBOUR MAP
# ============================================================

print()

print(
    "Nearest-neighbour configuration:"
)

print()

for target_code, neighbours in neighbour_map.items():

    target_name = station_info.loc[
        station_info["station_code"]
        == target_code,
        "station_name",
    ].iloc[0]

    print(
        f"{target_code} "
        f"({target_name})"
    )

    for rank, (
        neighbour_code,
        distance
    ) in enumerate(
        neighbours,
        start=1
    ):

        neighbour_name = (
            station_info.loc[
                station_info[
                    "station_code"
                ]
                == neighbour_code,
                "station_name",
            ]
            .iloc[0]
        )

        print(
            f"  {rank}. "
            f"{neighbour_code} "
            f"({neighbour_name}) "
            f"- {distance:.2f} km"
        )

    print()


# ============================================================
# CREATE LOOKUP TABLE
# ============================================================

# We only need timestamp, station code and PM2.5 from the
# original dataset to create spatial lag features.

pm25_lookup = df[
    [
        "timestamp",
        "station_code",
        "pm25",
    ]
].copy()


# ============================================================
# CREATE SPATIAL FEATURES
# ============================================================

print(
    "Creating spatial PM2.5 features..."
)

spatial_feature_frames = []


for target_code, neighbours in neighbour_map.items():

    target_rows = df[
        df["station_code"]
        == target_code
    ][
        [
            "timestamp",
            "station_code",
        ]
    ].copy()


    # --------------------------------------------------------
    # Process each neighbouring station
    # --------------------------------------------------------

    for neighbour_rank, (
        neighbour_code,
        distance
    ) in enumerate(
        neighbours,
        start=1
    ):

        neighbour_data = pm25_lookup[
            pm25_lookup[
                "station_code"
            ]
            == neighbour_code
        ][
            [
                "timestamp",
                "pm25",
            ]
        ].copy()


        # ----------------------------------------------------
        # Create historical lags
        # ----------------------------------------------------

        neighbour_data = (
            neighbour_data
            .sort_values("timestamp")
            .reset_index(drop=True)
        )


        for lag in NEIGHBOUR_LAGS:

            neighbour_data[
                f"neighbour_{neighbour_rank}"
                f"_pm25_lag_{lag}h"
            ] = (
                neighbour_data[
                    "pm25"
                ].shift(lag)
            )


        # ----------------------------------------------------
        # Keep only generated features
        # ----------------------------------------------------

        feature_columns = [
            "timestamp"
        ]

        feature_columns += [
            f"neighbour_{neighbour_rank}"
            f"_pm25_lag_{lag}h"
            for lag in NEIGHBOUR_LAGS
        ]


        neighbour_features = (
            neighbour_data[
                feature_columns
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Merge into target station
        # ----------------------------------------------------

        target_rows = target_rows.merge(
            neighbour_features,
            on="timestamp",
            how="left",
        )


        # Store distance as a static feature.
        target_rows[
            f"neighbour_{neighbour_rank}"
            "_distance_km"
        ] = distance


    spatial_feature_frames.append(
        target_rows
    )


# ============================================================
# COMBINE SPATIAL DATA
# ============================================================

spatial_features = pd.concat(
    spatial_feature_frames,
    ignore_index=True
)


# ============================================================
# MERGE WITH ORIGINAL DATA
# ============================================================

print(
    "Merging spatial features..."
)

df = df.merge(
    spatial_features,
    on=[
        "timestamp",
        "station_code",
    ],
    how="left",
)


# ============================================================
# VALIDATION
# ============================================================

spatial_columns = [

    column
    for column in df.columns
    if column.startswith(
        "neighbour_"
    )
]


print()

print(
    f"Spatial features created: "
    f"{len(spatial_columns)}"
)

print(
    f"Final dataset shape: "
    f"{df.shape}"
)

print()


# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicates = df.duplicated(
    subset=[
        "timestamp",
        "station_code",
    ]
).sum()


print(
    f"Duplicate station timestamps: "
    f"{duplicates}"
)


if duplicates != 0:

    raise ValueError(
        "Duplicate station timestamps "
        "were created during spatial merge."
    )


# ============================================================
# CHECK SPATIAL FEATURE EXAMPLES
# ============================================================

print()

print(
    "Example spatial features:"
)

print(
    df[
        [
            "timestamp",
            "station_code",
        ]
        + spatial_columns[:10]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

print()

print(
    "Saving spatial feature dataset..."
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()

print(
    "Spatial feature dataset saved to:"
)

print(
    OUTPUT_FILE
)

print()