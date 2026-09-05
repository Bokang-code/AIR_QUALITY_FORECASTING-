import os
import math
import requests
import pandas as pd
from dotenv import load_dotenv


# =============================================================================
# SETTINGS
# =============================================================================

LATITUDE = 30.63
LONGITUDE = 104.07

RADIUS_KM = 25

OUTPUT_FILE = "data/processed/v3/nearby_pm25_sensors.csv"

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAQ_API_KEY was not found in your .env file."
    )


# =============================================================================
# DISTANCE CALCULATION
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    radius = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


# =============================================================================
# FIND LOCATIONS
# =============================================================================

def find_locations():

    url = "https://api.openaq.org/v3/locations"

    headers = {
        "X-API-Key": API_KEY,
        "Accept": "application/json",
    }

    params = {
        "coordinates": f"{LATITUDE},{LONGITUDE}",
        "radius": int(RADIUS_KM * 1000),
        "limit": 100,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=60,
    )

    print("Status:", response.status_code)

    response.raise_for_status()

    data = response.json()

    return data.get("results", [])


# =============================================================================
# PROCESS RESULTS
# =============================================================================

def process_locations(locations):

    rows = []

    for location in locations:

        location_id = location.get("id")

        name = location.get(
            "name",
            "Unknown"
        )

        coordinates = location.get(
            "coordinates",
            {}
        )

        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")

        if latitude is None or longitude is None:
            continue

        distance = haversine_distance(
            LATITUDE,
            LONGITUDE,
            latitude,
            longitude,
        )

        sensors = location.get(
            "sensors",
            []
        )

        for sensor in sensors:

            parameter = sensor.get(
                "parameter",
                {}
            )

            parameter_name = parameter.get(
                "name"
            )

            if parameter_name != "pm25":
                continue

            rows.append(
                {
                    "location_id": location_id,
                    "location_name": name,
                    "sensor_id": sensor.get("id"),
                    "parameter": parameter_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_km": round(
                        distance,
                        3
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("FINDING NEARBY PM2.5 SENSORS")
    print("=" * 70)

    print()
    print(
        f"Target sensor coordinates: "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Search radius: {RADIUS_KM} km"
    )

    print()
    print("Searching OpenAQ...")

    locations = find_locations()

    print(
        f"Locations returned: {len(locations)}"
    )

    df = process_locations(
        locations
    )

    if df.empty:

        print()
        print(
            "No nearby PM2.5 sensors were found."
        )

        return

    df = (
        df.sort_values(
            "distance_km"
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("NEARBY PM2.5 SENSORS")
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Found {len(df)} PM2.5 sensors."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()