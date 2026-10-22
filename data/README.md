# Data

The data are **not** stored in this repository.

They are Monash Smart Infrastructure (MSI) building sensor extracts supplied for the MMA3001
unit, so ownership, licensing and privacy rest with the data provider. Place the files here
(or anywhere, and pass the paths on the command line):

| File | Content |
|------|---------|
| `5occupancySensor_MayToDec2024_9MRows.csv` | occupancy event log (about 9.1 M rows) |
| `5EnvSensor_MayToDec2024_180kRows.csv` | environmental readings (about 179 k rows) |
| `Sensor_ID_and_Locations.xlsx` | sensor and space metadata (reported unreliable) |

`python -m occenv prepare` writes cleaned 10-minute tables to `data/processed/`
(also ignored by git).
