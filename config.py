"""Configuration for the CAISO congestion-relief node map build.

Single source of truth for probe node names, parameters, and windows.
Imported by every step script.
"""

PROBE_IMPORT = "ALAMT1G_7_B1"
PROBE_FLAT = "TH_NP15_GEN-APND"

MARKET = "RTM"

BATTERY_DURATION_HOURS = 4
ROUND_TRIP_EFFICIENCY = 0.85

MIDDAY_WINDOW = (10, 15)
EVENING_WINDOW = (17, 21)

EPSILON_FLATNESS = 3.0
CONCENTRATION_THRESHOLD = 0.5

VOLTAGE_TO_MVA = {
    69: 150,
    115: 250,
    230: 700,
    500: 2500,
}
