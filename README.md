# occenv: does CO2 predict zone occupancy beyond the weekly schedule?

MMA3001 (Numerical Methods and Machine Learning) individual project, using Monash Smart
Infrastructure (MSI) building sensor data.

**Answer in one paragraph.** For the one environmental sensor with a usable occupancy signal
(`...869`, about 105 days of 10-minute data), adding CO2 to a schedule-only logistic regression raised
held-out AUC from 0.832 to 0.905 (paired difference +0.073, 95% interval +0.034 to +0.107). A control
sensor with no occupancy signal showed no gain, so the pipeline does not manufacture improvements.
A logistic regression matched gradient boosting and fits about four times faster.

## Engineering problem

Ventilation is usually run to a timetable, while demand-controlled ventilation uses CO2 as a proxy for
people present. If CO2 only reproduces the weekly timetable it adds nothing. **Question:** do
environmental measurements predict whether a zone is occupied *beyond what the hour of the week
already tells us*? The code is an assessment tool, not a controller.

Scope and limits: one working sensor, about 3.5 months, a zone of two inferred spaces (not a confirmed
room), a binary occupancy definition, and one 31-day chronological test window.

### Data problems treated as part of the problem
* **The sensor-to-space mapping is unreliable** (reported as a known, unfixable cloud-sync issue), so it
  is inferred from the data and validated (`occenv.mapping`).
* **76% of environmental rows are repeats** of (sensor, time); three of five sensors are dead (they re-send
  one or two stale readings). The timestamp inside the JSON, not `createdate`, is the measurement time.
* **Formaldehyde is mostly invalid** (error code and no value, 78.5% of unique readings): excluded.
* **Occupancy is an event log.** 92% of events are labelled CurrentlyOccupied, but headcount is above zero
  for only about 35% of time, so the log is time-weighted onto a regular grid.

## Inputs and outputs

| | Description | Format / units / domain |
|---|---|---|
| Input 1 | Occupancy event log (about 9.1 M events, 5 spaces) | CSV: `floorspaceid`, `headcount` (people, >= 0), `collecteddate` (timestamp with UTC offset) |
| Input 2 | Environmental readings (about 179 k rows, 5 sensors) | CSV: `sensorid`, `jsondata` (11 variables, each with a Unix-second UTC time and units, e.g. CO2 ppm, temperature C) |
| Input 3 (optional) | `Sensor_ID_and_Locations.xlsx` | used only by `profile` to report mapping coverage |
| Intermediate | Cleaned tables | `data/processed/*.csv.gz`: regular 10-minute UTC grid, NaN where invalid |
| Outputs | Text reports and figures | `results/*.txt` and `report/figures/*.png` |

**Invalid or unsupported input handling:** missing required columns raise `ValueError` naming them;
unparseable JSON rows are counted and skipped; values with an error code become NaN; repeated and
isolated stale readings are removed; stale occupancy states become missing; an unknown sensor or space,
no time overlap, or a one-class train/test period raises a clear error (shown as a message by the CLI).

## Method

1. **Clean** both sources onto a 10-minute UTC grid (`occenv.cleaning`).
2. **Infer the mapping** from residual correlations (the mean hour-of-week profile removed), over a common
   window for all spaces, with a lag scan, a week-shift placebo test and a stability check (`occenv.mapping`).
   The zone choice was made with training-period data only.
3. **Model** `occupied` = zone headcount >= threshold, from schedule features (hour, day of week) and
   environmental features (CO2, temperature, humidity, PM2.5, plus past-only rates and rolling statistics).
4. **Validate chronologically**: first 70% train, 1-day gap, last 30% test; forward-chaining CV inside the
   training period only; the test set is scored once; uncertainty from a day-block bootstrap.
5. **Compare**: schedule lookup (baseline), logistic regression and histogram gradient boosting (200
   iterations, no early stopping), each with schedule only, schedule + CO2, and schedule + all environment.

## Key results (reproduce with the commands below)

| Sensor | Threshold | Schedule-only AUC | + CO2 AUC | Difference (95% interval) |
|---|---|---|---|---|
| `...869`, zone of 2 spaces | 1 | 0.832 | 0.905 | +0.073 (+0.034, +0.107) |
| `...869`, zone of 2 spaces | 2 | 0.831 | 0.925 | +0.094 (+0.055, +0.133) |
| `...326` (negative control) | 1 | 0.916 | 0.908 | -0.009 (-0.016, -0.001) |

Logistic regression shown; gradient boosting gives similar differences. Gradient boosting is not better
than logistic regression (intervals include zero, or slightly favour logistic with CO2 at threshold 1),
and an environment-only model scores AUC 0.726 even on the control, because environmental variables
follow a daily cycle. The fair test is therefore schedule versus schedule + environment.

## Performance (measured, identical results confirmed on the real data)

| Step | Baseline | Improved | Result |
|---|---|---|---|
| Duplicate counting, 2M rows | 3.35 s, 361 MB | 3.02 s, 79 MB | 1.1x time, 4.6x less memory |
| Timestamp parsing, 1M rows | 5.60 s | 1.51 s | 3.7x faster (about 8x more memory) |
| Environmental cleaning, full file | 7.36 s | 3.98 s | 1.85x faster |

Profiling (`python -m occenv benchmark`) showed timestamp parsing was 76% of occupancy cleaning time and
JSON parsing 70% of environmental cleaning time, so those were optimised first.

## Limitations

One working sensor; about 105 days; zone not a confirmed room; one end-of-semester test window; the ground
truth is another sensor's count of unknown accuracy; public holidays and semester breaks are not modelled;
the placebo test has little power on short windows; the control sensor has occasional CO2 spikes that were
not removed. Results depend on data volume (a 44-day experiment showed no environmental gain).

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows; on Mac/Linux: source .venv/bin/activate
pip install -e ".[dev]"

# data go in data/ (not committed, see data/README.md)
python -m occenv profile  --occ data/<occupancy.csv> --env data/<env.csv> --xlsx "data/<mapping.xlsx>"
python -m occenv prepare  --occ data/<occupancy.csv> --env data/<env.csv>
python -m occenv diagnose --end 2024-05-14 --exclude 3e5a14df --out results/mapping_train_only.txt
python -m occenv train --sensor 869 --spaces 0f3f37f8 941f0352 --thr 1 --out results/thr1.txt
python -m occenv train --sensor 869 --spaces 0f3f37f8 941f0352 --thr 2 --out results/thr2.txt
python -m occenv train --sensor 869 --spaces 0f3f37f8 941f0352 --thr 5 --out results/thr5.txt
python -m occenv train --sensor 869 --spaces 0f3f37f8 941f0352 3e5a14df --thr 1 --out results/sensitivity_3space.txt
python -m occenv train --sensor 326 --spaces 0f3f37f8 941f0352 --thr 1 --out results/control_326.txt
python -m occenv benchmark --occ data/<occupancy.csv> --env data/<env.csv> --nrows 2000000
python -m occenv figures   --occ data/<occupancy.csv> --env data/<env.csv>
```

Commands: `profile` (audit raw files), `prepare` (clean to 10-minute tables), `screen` and `diagnose`
(sensor-to-space mapping evidence), `train` (features, chronological validation, model comparison),
`benchmark` (timing, profiling and cost model), `figures` (every report figure). Use
`python -m occenv <command> --help` for options.

## Tests and documentation

```bash
pytest -v --junitxml=reports/test_report.xml      # 55 tests; add --cov=occenv for coverage
pdoc --docformat google -o docs occenv            # HTML documentation (open docs/index.html)
```

The tests use small synthetic data in the MSI format. They include hand-calculated checks of the
occupancy time-weighting, a check that features use only past values, equality checks that every
optimised implementation matches its baseline, and a check that the mapping diagnostics recover a known
sensor-to-space link.

## Repository layout

```
src/occenv/   package: config, reporting, profiling, cleaning, mapping, modelling, performance, figures, cli
tests/        pytest tests and the synthetic data generator
data/         NOT committed (see data/README.md)
results/      text output of every run
report/       figures used in the written report
reports/      pytest test report
docs/         generated HTML documentation
```

## AI use

Claude (Anthropic) was used throughout this project for code drafting, analysis design, debugging and
report drafting. All AI output was checked by running the test suite, comparing optimised code with its
baseline on the real data, and using a negative-control sensor. The full disclosure and reflection are in
the written report. *(Add any details you want recorded here.)*

## Licence

MIT, see `LICENSE`.
