"""
PM2.5 health-risk interpretation.

Thresholds are based on the WHO 2021 Global Air Quality Guidelines
for the 24-hour mean PM2.5 concentration.

WHO 2021:
    AQG  = 15 µg/m³
    IT-4 = 25 µg/m³
    IT-3 = 37.5 µg/m³
    IT-2 = 50 µg/m³
    IT-1 = 75 µg/m³

The labels used by the application are application-defined
descriptions of these WHO levels. They are not official WHO
risk-category names.
"""


WHO_PM25_THRESHOLDS = {
    "AQG": 15.0,
    "IT4": 25.0,
    "IT3": 37.5,
    "IT2": 50.0,
    "IT1": 75.0,
}


RISK_LEVELS = [
    {
        "key": "aqg",
        "max_value": 15.0,
        "label": "WHO GUIDELINE",
        "short_label": "Guideline",
        "description": (
            "The predicted concentration is at or below the "
            "WHO 2021 24-hour PM2.5 air quality guideline."
        ),
        "threshold": "≤ 15 µg/m³",
    },
    {
        "key": "it4",
        "max_value": 25.0,
        "label": "ABOVE GUIDELINE",
        "short_label": "Above guideline",
        "description": (
            "The predicted concentration is above the WHO "
            "guideline and within Interim Target 4."
        ),
        "threshold": "> 15–25 µg/m³",
    },
    {
        "key": "it3",
        "max_value": 37.5,
        "label": "ELEVATED",
        "short_label": "Elevated",
        "description": (
            "The predicted concentration is within the WHO "
            "Interim Target 3 range."
        ),
        "threshold": "> 25–37.5 µg/m³",
    },
    {
        "key": "it2",
        "max_value": 50.0,
        "label": "HIGH",
        "short_label": "High",
        "description": (
            "The predicted concentration is within the WHO "
            "Interim Target 2 range."
        ),
        "threshold": "> 37.5–50 µg/m³",
    },
    {
        "key": "it1",
        "max_value": 75.0,
        "label": "VERY HIGH",
        "short_label": "Very high",
        "description": (
            "The predicted concentration is within the WHO "
            "Interim Target 1 range."
        ),
        "threshold": "> 50–75 µg/m³",
    },
    {
        "key": "above_it1",
        "max_value": float("inf"),
        "label": "ABOVE IT1",
        "short_label": "Above IT1",
        "description": (
            "The predicted concentration is above the highest "
            "WHO 2021 PM2.5 interim target."
        ),
        "threshold": "> 75 µg/m³",
    },
]


def classify_pm25(pm25):
    """
    Classify a PM2.5 concentration against the WHO 2021
    24-hour PM2.5 AQG and interim targets.

    Parameters
    ----------
    pm25 : float
        PM2.5 concentration in µg/m³.

    Returns
    -------
    dict
        Classification information for the application.
    """

    if pm25 is None:
        raise ValueError("PM2.5 value cannot be None.")

    try:
        pm25 = float(pm25)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "PM2.5 value must be numeric."
        ) from exc

    if pm25 < 0:
        raise ValueError(
            "PM2.5 concentration cannot be negative."
        )

    for level in RISK_LEVELS:
        if pm25 <= level["max_value"]:
            return {
                "value": pm25,
                "key": level["key"],
                "label": level["label"],
                "short_label": level["short_label"],
                "description": level["description"],
                "threshold": level["threshold"],
                "unit": "µg/m³",
            }

    # Defensive fallback.
    return {
        "value": pm25,
        "key": "above_it1",
        "label": "ABOVE IT1",
        "short_label": "Above IT1",
        "description": (
            "The predicted concentration is above the "
            "highest WHO 2021 PM2.5 interim target."
        ),
        "threshold": "> 75 µg/m³",
        "unit": "µg/m³",
    }


def get_thresholds():
    """
    Return the WHO PM2.5 thresholds used by the application.
    """
    return WHO_PM25_THRESHOLDS.copy()


def get_all_levels():
    """
    Return all application-defined interpretation levels.
    """
    return [level.copy() for level in RISK_LEVELS]