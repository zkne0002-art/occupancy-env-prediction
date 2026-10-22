# occenv: predicting zone occupancy from environmental sensors

MMA3001 (Numerical Methods and Machine Learning) individual project, using the Monash Smart
Infrastructure (MSI) occupancy and environmental sensor data.

## Engineering problem

Buildings are ventilated and conditioned for assumed occupancy. If occupancy could be
inferred from cheap environmental sensors (mainly CO2), ventilation could follow actual use.
**Question:** do environmental measurements predict whether a zone is occupied *beyond what
the weekly schedule alone already tells us*?

Scope and limits: one working sensor with a usable signal, about 3.5 months of 10-minute
data, a zone (two inferred spaces) rather than a confirmed room, one chronological test window.

### Data problems this project had to treat as part of the problem
* **Sensor-to-space mapping is unreliable** (reported as a known cloud-sync issue with no
  corrected mapping available), so the mapping is inferred from the data and validated.
* **About 76 % of env rows are repeats** of (sensor, time); three of five sensors are dead
  (re-send one stale reading). The JSON timestamp, not `createdate`, is the measurement time.
* **Formaldehyde is mostly invalid** (error code and no value), so it is excluded.
* **Occupancy is an event log**, not a regular series, and must be time-weighted.

## Inputs and outputs

| | Description | Format / units / domain |
|---|---|---|
| Input 1 | Occupancy event log | CSV; `floorspaceid`, `headcount` (people, >= 0), `collecteddate` (timestamp with UTC offset) |
| Input 2 | Environmental readings | CSV; `sensorid`, `jsondata` (JSON list of variables, Unix-second UTC time, units per variable e.g. CO2 ppm, temperature C) |
| Input 3 (optional) | `Sensor_ID_and_Locations.xlsx` | only used by `profile` to report mapping coverage |
| Intermediate | Cleaned tables | `env_10min.csv.gz`, `occupancy_10min.csv.gz`: regular 10-minute UTC grid, NaN where invalid |
| Output | Result reports | text files in `results/`: mapping evidence, CV scores, held-out test metrics, paired AUC differences with 95 % intervals, permutation importance |

**Invalid or unsupported input handling:** missing required columns raise `ValueError` naming
the columns; unparseable JSON rows are counted and skipped; values with an error code become
NaN; isolated or repeated stale readings are removed; stale occupancy states become missing;
a sensor or space not found, a missing time overlap, or a one-class train/test period raise
`ValueError` (shown as a clear message by the command line interface).

## Method (short)
1. **Clean** both sources onto a 10-minute UTC grid (`occenv.cleaning`).
2. **Infer the mapping** with residual correlations (hour-of-week profile removed), a fair common
   window, lag scan, week-shift placebo test and stability checks (`occenv.mapping`).
3. **Model** `occupied` = zone headcount >= threshold. Features: schedule (hour, day of week)
   and environment (CO2, temperature, humidity, PM2.5, plus past-only rates and rolling stats).
4. **Validate chronologically**: first 70 % train, 1-day gap, last 30 % test; forward-chaining CV
   inside the training period only; day-block bootstrap for paired AUC differences.
5. **Compare**: schedule lookup (baseline), logistic regression and gradient boosting, each with
   schedule only vs schedule + environment (or CO2 only).

## Key results (author's runs on the MSI extract; reproduce with the commands below)

| Sensor | Threshold | Schedule-only AUC | + CO2 AUC | Difference (95 % CI) |
|---|---|---|---|---|
| `...869`, zone of 2 spaces | 1 | 0.832 | 0.905 | +0.073 (+0.034, +0.107) |
| `...869`, zone of 2 spaces | 2 | 0.831 | 0.925 | +0.094 (+0.055, +0.133) |
| `...326` (negative control) | 1 | 0.916 | 0.908 | -0.009 (-0.016, -0.001) |

(Logistic regression shown; gradient boosting is similar.) Reading: CO2 adds reliable skill
over the schedule for the sensor that shows an occupancy signal, and adds nothing for the
control sensor with none. The zone was chosen using training-period data only.

## Limitations
One working sensor; about 3.5 months; zone not a confirmed room; one 31-day test window at the
end of a semester; ground truth is itself a sensor count of unknown accuracy; public holidays
and semester breaks are not modelled; the placebo test has little power on short windows.

## Install and run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (use source .venv/bin/activate on Mac/Linux)
pip install -e ".[dev]"

python -m occenv profile  --occ <occupancy.csv> --env <env.csv> --xlsx <mapping.xlsx>
python -m occenv prepare  --occ <occupancy.csv> --env <env.csv> --out data/processed
python -m occenv screen
python -m occenv diagnose --end 2024-05-14 --exclude 3e5a14df
python -m occenv train --sensor 869 --spaces 0f3f37f8 941f0352 --thr 1 --out results/thr1.txt
python -m occenv train --sensor 326 --spaces 0f3f37f8 941f0352 --thr 1 --out results/control_326.txt
```

## Tests and documentation

```bash
pytest -v --junitxml=reports/test_report.xml        # test report (XML) ; add --cov=occenv for coverage
pdoc -o docs occenv                                  # HTML documentation of the source code
```

The tests use small synthetic data in the MSI format and include hand-calculated checks of the
occupancy time-weighting, a check that features use only past values, and a check that the
mapping diagnostics recover a known sensor-to-space link.

## Repository layout
```
src/occenv/   package (config, reporting, profiling, cleaning, mapping, modelling, cli)
tests/        pytest tests and synthetic data generator
data/         NOT committed (see data/README.md)
results/      text outputs of each run
docs/         generated HTML documentation
```

## AI use
AI assistance (Claude) was used for this project. The full disclosure and reflection are in the
written report. *(Complete the details before submission.)*

## Licence
MIT, see `LICENSE`.
