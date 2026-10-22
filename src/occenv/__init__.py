"""occenv: predicting zone occupancy from environmental sensor data.

Built for the MMA3001 individual project using the Monash Smart Infrastructure
(MSI) occupancy and environmental sensor extracts.

Modules:

- `occenv.config`: shared constants (time bin, timezone, variable names, feature sets)
- `occenv.reporting`: helper that echoes results and saves them to a text file
- `occenv.profiling`: chunked audit of the raw (large) CSV files
- `occenv.cleaning`: de-duplication, validity filtering and 10-minute regularisation
- `occenv.mapping`: inferring which occupancy space a sensor belongs to
- `occenv.modelling`: features, chronological validation, models and comparisons
- `occenv.performance`: timing, profiling and cost model (baseline vs improved code)
- `occenv.cli`: command line interface (`python -m occenv --help`)
"""
__docformat__ = "google"
__version__ = "0.1.0"
