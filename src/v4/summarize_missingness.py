from pathlib import Path
import pandas as pd


INPUT_FILE = Path("reports/v4/missingness_by_year.csv")


def main():

    df = pd.read_csv(INPUT_FILE)

    print("\n" + "=" * 80)
    print("PM2.5 MISSINGNESS BY STATION AND YEAR")
    print("=" * 80)

    # Keep only PM2.5
    pm25 = df[df["variable"] == "PM2.5"].copy()

    # Pivot into station × year table
    table = pm25.pivot(
        index="station_code",
        columns="year",
        values="missing_pct"
    )

    print("\nPM2.5 missingness (%):")
    print(table.round(2).to_string())

    # --------------------------------------------------------
    # Average PM2.5 missingness
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("AVERAGE PM2.5 MISSINGNESS")
    print("=" * 80)

    avg = (
        pm25.groupby(
            ["station_code", "station_name"]
        )["missing_pct"]
        .mean()
        .sort_values()
    )

    for (station_code, station_name), value in avg.items():

        print(
            f"{station_code} ({station_name}): "
            f"{value:.2f}%"
        )

    # --------------------------------------------------------
    # Worst year for each station
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("WORST PM2.5 YEAR FOR EACH STATION")
    print("=" * 80)

    for station_code in table.index:

        row = table.loc[station_code]

        worst_year = row.idxmax()
        worst_value = row.max()

        print(
            f"{station_code}: "
            f"{int(worst_year)} "
            f"({worst_value:.2f}% missing)"
        )

    # --------------------------------------------------------
    # Average missingness by year across stations
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("AVERAGE PM2.5 MISSINGNESS ACROSS ALL STATIONS")
    print("=" * 80)

    yearly_average = (
        pm25.groupby("year")["missing_pct"]
        .mean()
        .sort_index()
    )

    for year, value in yearly_average.items():

        print(
            f"{int(year)}: {value:.2f}%"
        )

    # --------------------------------------------------------
    # Stations above thresholds
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STATIONS ABOVE MISSINGNESS THRESHOLDS")
    print("=" * 80)

    for threshold in [5, 10, 15, 20]:

        print(
            f"\nYears with > {threshold}% missing PM2.5:"
        )

        found = False

        for _, row in pm25.iterrows():

            if row["missing_pct"] > threshold:

                print(
                    f"  {row['station_code']} - "
                    f"{int(row['year'])}: "
                    f"{row['missing_pct']:.2f}%"
                )

                found = True

        if not found:

            print("  None")

    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()