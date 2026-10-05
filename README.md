# MLB Game Predictor 2026
Current through 2026-10-04. `main` is the current copy.

Training table is league-wide completed games from 2025-03-18 through 2026-10-04. Pre-2025 rows were dropped. The model uses MLB Stats API box scores, Baseball Savant K% and BB%, and MLB team OPS versus LHP and RHP. Starter FIP is the prior pitching line. xFIP and park factor are not in this build. This model does not price sportsbook props and is not a bet.

## Run
```bash
python scripts/refresh_today.py
```

Outputs land in `artifacts/`.
