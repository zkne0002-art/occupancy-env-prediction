"""occenv: predicting zone occupancy from environmental sensor data.

Built for the MMA3001 individual project using the Monash Smart Infrastructure
(MSI) occupancy and environmental sensor extracts.

Modules:
    * :mod:`occenv.config`     shared constants (time bin, timezone, variable names, feature sets)
    * :mod:`occenv.reporting`  small helper that echoes results and saves them to a text file
    * :mod:`occenv.profiling`  chunked audit of the raw (large) CSV files
    * :mod:`occenv.cleaning`   de-duplication, validity filtering and 10-minute regularisation
    * :mod:`occenv.mapping`    inferring which occupancy space a sensor belongs to
    * :mod:`occenv.modelling`  features, chronological validation, models and comparisons
    * :mod:`occenv.cli`        command line interface (``python -m occenv --help``)
"""
__version__ = "0.1.0"
