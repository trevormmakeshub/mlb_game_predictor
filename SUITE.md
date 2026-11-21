# Sports suite status

Checked 2026-11-20. These are win probabilities and score projections. They are not live book prices and they are not bets.

| Sport | Repo | Last final | Next games | High confidence | Holdout accuracy | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| NFL | nfl_game_predictor | 2026-11-19 | 12 | 7 | 0.6463 from 2026-09-01 | 18 features. In-sample accuracy 0.6789. Points allowed and recent form were tested and not adopted. The points blend was adopted. Blend MAE 12.261. |
| NCAA | ncaa_football_game_predictor | 2026-11-19 | 80 | 21 | 0.7263 from 2026-08-01 | 12 features. In-sample accuracy 0.7113. Recent form and the points blend were adopted. Six games on 2026-11-20 were still not final. Next window runs through 2026-11-27. |
| NBA | nba_game_predictor | 2026-11-20 | 49 | 20 | 0.6552 from 2026-10-20 | 12 features. In-sample accuracy 0.6884 on 1531 rows. Recent form still beats the baseline on 232 holdout games. The points blend was adopted. stats.nba.com returned the team log and 473 player rows. |
| MLB | mlb_game_predictor | 2026-10-28 | 0 | 0 | 0.5475 from 2026-09-01 | 23 features. The schedule was checked through 2026-11-20. The 2026 season is over. The rs_14 blend was rejected. Total MAE 3.256. |
| NFL scores | nfl-scores | 2026-11-19 |  |  |  | 151 completed 2026 games. No model and no odds. |
| MLB predictions | mlb-predictions | 2026-10-28 |  |  |  | 2469 finals. The schedule was checked through 2026-11-20. No new finals. |
| MLB run line | mlb-runline | 2026-10-28 |  |  |  | 9889 finals from 2023-03-30. The schedule was checked through 2026-11-20. No new finals. |
