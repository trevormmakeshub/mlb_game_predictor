# MLB Game Predictor 2026
Current through 2026-10-28. The schedule was checked through 2026-11-07. `main` is the current copy.

Training table is 4946 league-wide completed games from 2025-03-18 through 2026-10-28. Pre-2025 rows were dropped. The model uses MLB Stats API box scores, Baseball Savant K% and BB%, and MLB team OPS versus LHP and RHP. Starter FIP is the prior pitching line. Starter xFIP is the 2025 Savant fly-ball count with the 2025 MLB pitching line, joined only onto 2026 games. Park factor is not in this build: Baseball Savant 2026 index_woba misses the Athletics, and the 2025 table also misses the Rays. This model does not price sportsbook props and is not a bet.

## Run
```bash
python scripts/refresh_today.py
```

Outputs land in `artifacts/`.
