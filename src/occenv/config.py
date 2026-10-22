"""Shared constants.

Attributes:
    BIN: Pandas frequency string of the common time grid (10 minutes).
    TZ: Local timezone used for hour-of-day / day-of-week features.
    VAR_NAMES: Map from the raw sensor variable names to short column names.
    ENV_BASE: Raw environmental model features.
    ENV_ENG: Engineered environmental features (rates of change and rolling statistics).
    ENV_ALL: ``ENV_BASE + ENV_ENG``.
    ENV_CO2: CO2-only feature subset (drops variables dominated by weather).
"""
BIN = "10min"
TZ = "Australia/Melbourne"

VAR_NAMES = {
    "Battery": "battery_v",
    "Carbon dioxide": "co2_ppm",
    "Humidity": "humidity_pct",
    "Formaldehyde": "hcho_ugm3",
    "Pressure": "pressure_mb",
    "Temperature": "temp_c",
    "Light Volatile Organic Compounds": "lvoc_ppb",
    "Particulate matter 1": "pm1",
    "Particulate matter 4": "pm4",
    "Particulate matter 2.5": "pm2_5",
    "Particulate matter 10": "pm10",
}

ENV_BASE = ["co2_ppm", "temp_c", "humidity_pct", "pm2_5"]
ENV_ENG = ["co2_d30", "co2_roll1h", "co2_above_base", "temp_d30"]
ENV_ALL = ENV_BASE + ENV_ENG
ENV_CO2 = ["co2_ppm", "co2_d30", "co2_roll1h", "co2_above_base"]
