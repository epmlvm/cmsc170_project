# DBSCAN weather forecasting — Los Baños, Laguna

Libraries used: Python standard library (`csv`, `os`, `re`, `json`, `math`,
`argparse`, `collections`, `datetime`, `platform`, `random`, `tempfile`, `unittest`), NumPy (arrays and
basic linear algebra), Matplotlib (figures). Nothing else.

## Run — week mode 

```
python run_experiment.py --data dataset/training --mode week
```

Leave-one-day-out: 7 rounds, each hides one day, trains on the other six, and
predicts the hidden day, so all 76 rows get a prediction from a model that never
saw their day. Rows within `--gap` hours (default 3) of the hidden day are also
left out of training. eps and min_pts are chosen inside each round by the same
leave-one-day-out on the training days only. Runs in a few seconds.

Outputs in `results/`: `summary.txt` (all score tables as plain text),
`predictions_LosBanos_Laguna.txt` (the original data file, unchanged, with each
method's predicted label and a yes/no correct column added at the end of every
row), `results.json` (same scores in JSON), `config.json`,
`scores_by_day.png`, `confusion_dbscan.png`, `kdistance.png`, `clusters.png`
(a descriptive fit on all rows, not used for scoring).

Real result on the Los Baños file (12–18 Nov 2015, 76 rows):

| Method | Accuracy | Macro-F1 | Unknown |
| --- | --- | --- | --- |
| DBSCAN | 0.092 | 0.064 | 0.395 |
| Hour-of-day climatology | 0.368 | 0.215 | 0 |
| Majority class | 0.013 | 0.005 | 0 |

Why DBSCAN did poorly here (worth explaining in the report):
- About 50 training rows for 9 labels: most of the feature space is too sparse
  for core points, so 40% of rows get "unknown".
- Day/night errors (ClearN predicted as ClearD): hour of day is 2 of 9 features,
  so a similar-looking row from the other half of the day is often nearer.
- The file has no cloud-cover variable, so Clear vs Cloud1 is barely separable
  from rain, temperature, humidity, wind and pressure.
- Climatology wins because the most common label per hour already gets the D/N
  letter right and needs no dense clusters.
- Days 6–7 have 2 rows each; their per-day scores mean little.
These results describe one forecast run's labels for one week, not real-weather
skill. Settings were not changed after seeing the results.



- Validation = whole provinces picked with `--seed` (default 0) until they hold
  `--val-share` (15%) of the towns. `split.json` lists every town's role.
- Training is capped at `--max-train 15000` rows (fixed random sample) so the
  grid search finishes; that took about 3 minutes on our test machine. Use
  `--max-train 8000` for a quicker run.
- Extra outputs: per-day scores in `results.json` and `scores_by_day.png`.
- Read the per-day numbers with care: days 1–2 have 24 rows each, days 3–5 have
  8, days 6–7 only 2. A day-7 score of 100% is 2 rows out of 2.
- Files whose names are not `TYPE_Town_Province.txt` are skipped with a message.

What the scores mean: how well DBSCAN, trained elsewhere, reproduces the forecast
model's own labels for Los Baños that week. It is not a test of predicting real
weather, and it is not a forecast of future days. Say so in the report.

## Run — other modes (need many daily runs)

```
python run_experiment.py --data dataset/training --mode forecast   #  hourly forecast
python run_experiment.py --data dataset/training --mode classify   # label of the current hour
python -m unittest discover -s tests -t .                          # unit tests
```

Useful options: `--out results_dir`, `--lead-min 168 --lead-max 288` (change to
145 for calendar days 7–12), `--stride 3` (every 3rd start hour, faster),
`--min-pts 3 5 8 12 20`, `--n-eps 12`, `--split 0.5 0.25 0.25` (bigger validation and test blocks for shorter datasets).

Outputs in `results/`: `results.json` (all scores, per class and per lead),
`config.json` (chosen eps/min_pts, features, versions), `split.json` (block
boundaries — give this to the other methods in your comparison), and figures:
`kdistance.png`, `validation_grid.png`, `scores_by_lead.png`,
`confusion_dbscan.png`, `clusters.png`.

## Files

| File | Job |
| --- | --- |
| `data_loader.py` | Find run folders, parse the Los Baños file, stitch the hourly timeline |
| `features.py` | Build examples; sin/cos for wind and hour; standardize with train stats |
| `crossval.py` | Week mode: leave-one-day-out with nested tuning |
| `split.py` | Time-ordered blocks (forecast/classify); town roles by province (towns) |
| `dbscan.py` | `DBSCAN.fit`, `DBSCAN.predict`, `k_distance`, cluster voting |
| `baselines.py` | Climatology, persistence, majority class |
| `report.py` | Writes `summary.txt` and the predictions text file |
| `metrics.py` | Accuracy, per-class P/R/F1, macro-F1, confusion matrix, skill |
| `run_experiment.py` | Runs everything with one command |



