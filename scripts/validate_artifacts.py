"""Check the saved MLB table. This does not download or retrain."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "artifacts" / "training_table.csv"
CALIBRATION = ROOT / "artifacts" / "calibration.csv"
HIGH = ROOT / "artifacts" / "high_confidence.csv"


def latest_remaining() -> Path | None:
    matches = sorted((ROOT / "artifacts").glob("mlb_*_remaining_games.csv"))
    return matches[-1] if matches else None
REQUIRED_NEXT = (
    "date",
    "home",
    "away",
    "home_win_prob",
    "predicted_home_points",
    "predicted_away_points",
    "predicted_total",
    "confidence_tier",
    "pick_side",
)


def fail(message: str) -> None:
    print(f"fail {message}")


def main() -> int:
    problems = 0
    remaining_path = latest_remaining()
    for path in (TABLE, remaining_path, CALIBRATION, HIGH):
        if path is None or not path.exists():
            fail(f"missing {getattr(path, 'name', 'remaining games')}")
            problems += 1
    if problems:
        print(f"validation fail {problems}")
        return 1
    with TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["game_pk"] for row in rows]
    if len(ids) != len(set(ids)):
        fail("duplicate game_pk")
        problems += 1
    dates = [row["game_date"][:10] for row in rows]
    if not dates:
        fail("training table is empty")
        problems += 1
    else:
        if min(dates) < "2025-01-01":
            fail(f"training table starts at {min(dates)}")
            problems += 1
        if max(dates) > date.today().isoformat():
            fail(f"training table date {max(dates)} is after today")
            problems += 1
        missing_scores = sum(1 for row in rows if row.get("away_runs") in ("", None) or row.get("home_runs") in ("", None))
        if missing_scores:
            fail(f"{missing_scores} rows lack runs")
            problems += 1
    with remaining_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        upcoming = list(reader)
    for column in REQUIRED_NEXT:
        if column not in header:
            fail(f"remaining games missing {column}")
            problems += 1
    if any(row.get("date", "") > date.today().isoformat() and row.get("home_win_prob") for row in upcoming):
        pass
    with CALIBRATION.open(encoding="utf-8", newline="") as handle:
        bins = list(csv.DictReader(handle))
    floor = next((row for row in bins if row.get("split") == "holdout" and row.get("bin") == "50-55"), None)
    if floor is None:
        fail("calibration is missing the holdout 50-55 bin")
        problems += 1
    elif floor.get("home_win_rate") and float(floor["home_win_rate"]) < 0.50 and floor.get("calibrated") == "yes":
        fail("50-55 holdout bin is marked yes below the floor")
        problems += 1
    print(f"training_rows {len(rows)}")
    print(f"min_date {min(dates) if dates else ''}")
    print(f"max_date {max(dates) if dates else ''}")
    print(f"remaining {len(upcoming)}")
    if problems:
        print(f"validation fail {problems}")
        return 1
    print("validation pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
