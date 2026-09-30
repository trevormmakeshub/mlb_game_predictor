# Notes

## Bugs fixed

- README pointed at baseball_dataset.py, baseball_model.py, and baseball_prediction.py in the repo root. Those files live in legacy/. The README now points at legacy/ and at scripts/refresh_today.py.
- legacy/baseball_prediction.py hardcoded the 2024 season in the date filter, the schedule end date, and the team lookup. Those now use datetime.now().year.
- legacy/baseball_dataset.py described a 2000-2024 build and its year loop stopped at 2025. The loop now runs through the current year, and the retry loop runs only when the file is executed as a script.
- legacy/baseball_model.py and legacy/baseball_prediction.py read stats.csv and the pickles from the working directory at import. They now look next to the script, then in the working directory, and a missing csv exits instead of failing at import.
- Unused imports were removed from the three legacy scripts. The Ridge Classifier and the linear run model were not rewritten.
- mlb_predictor/data.py constructed a Supabase client at import, and importing the module required the supabase package. The import and the client now happen only when the client is used. No key was added.
- pip install -r requirements.txt failed on numpy==2.0.0 because Python 3.13 has no wheel for that pin and this machine has no C compiler. numpy, pandas, scikit-learn, and scipy were pinned to the installed wheels (2.5.3, 3.0.6, 1.9.1, 1.18.1). The retry succeeded.

## Data

The training table is league-wide completed games from 2025-03-01. Pre-2025 rows were dropped. The two-game file is only a slice of the full table.

Minimum date: 2025-03-18
Maximum date: 2026-09-30
Row count: 4911

Features: 13. September accuracy: 0.5397.

## Fields skipped

- FIP: FanGraphs pitching returned HTTP 403. Baseball Reference pitching 2025 has no FIP column.
- K%: FanGraphs pitching returned HTTP 403. Baseball Reference pitching 2025 has no K% column.
- BB%: FanGraphs pitching returned HTTP 403. Baseball Reference pitching 2025 has no BB% column.
- lineup wRC+ versus starter hand: FanGraphs team batting returned HTTP 403. pybaseball returned no wRC+ split versus pitcher hand.
- park factor: FanGraphs team batting returned HTTP 403. The pybaseball Lahman teams download failed (File is not a zip file).

Sources used: MLB Stats API box scores.

this model does not price sportsbook props and is not a bet.
