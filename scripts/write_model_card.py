"""Write holdout calibration from the saved training table. No download."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "artifacts" / "training_table.csv"
CALIBRATION = ROOT / "artifacts" / "calibration.csv"
HIGH = ROOT / "artifacts" / "high_confidence.csv"
REMAINING = ROOT / "artifacts" / "mlb_2026-11-07_remaining_games.csv"
HOLDOUT = "2026-09-01"
FEATURES = [
    "home_starter_rest_days",
    "home_starter_last_ip",
    "home_starter_fip",
    "home_bullpen_era_14",
    "home_bullpen_ip_yesterday",
    "home_rs_14",
    "home_ra_14",
    "away_starter_rest_days",
    "away_starter_last_ip",
    "away_starter_fip",
    "away_bullpen_era_14",
    "away_bullpen_ip_yesterday",
    "away_rs_14",
    "away_ra_14",
    "home_won_game1",
    "home_starter_xfip",
    "away_starter_xfip",
    "home_starter_k_pct",
    "away_starter_k_pct",
    "home_starter_bb_pct",
    "away_starter_bb_pct",
    "home_lineup_ops_vs_hand",
    "away_lineup_ops_vs_hand",
]
REMAINING_COLUMNS = [
    "date", "away", "home", "starter_away", "starter_home", "home_win_prob",
    "predicted_away_runs", "predicted_home_runs", "predicted_away_points", "predicted_home_points",
    "predicted_total", "pick_side", "pick_team", "confidence", "confidence_tier",
    "model_side", "data_cutoff",
]


def probabilities(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med = train[FEATURES].median()
    filled_train = train[FEATURES].fillna(med).fillna(0.0)
    mu = filled_train.mean()
    sd = filled_train.std(ddof=0).replace(0, 1.0)
    x_train = (filled_train - mu) / sd
    x_test = (test[FEATURES].fillna(med).fillna(0.0) - mu) / sd
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(x_train, train["home_win"].astype(int))
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(ridge.decision_function(x_train).reshape(-1, 1), train["home_win"].astype(int))
    probability = calibrator.predict_proba(ridge.decision_function(x_test).reshape(-1, 1))[:, 1]
    predicted = ridge.predict(x_test)
    return probability, predicted


def run_mae(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    med = train[FEATURES].median()
    filled_train = train[FEATURES].fillna(med).fillna(0.0)
    mu = filled_train.mean()
    sd = filled_train.std(ddof=0).replace(0, 1.0)
    x_train = (filled_train - mu) / sd
    x_test = (test[FEATURES].fillna(med).fillna(0.0) - mu) / sd
    home = LinearRegression().fit(x_train, train["home_runs"])
    away = LinearRegression().fit(x_train, train["away_runs"])
    pred_home = home.predict(x_test)
    pred_away = away.predict(x_test)
    actual = test["home_runs"].to_numpy() + test["away_runs"].to_numpy()
    regression = pred_home + pred_away
    blend = 0.5 * regression + 0.5 * (test["home_rs_14"].to_numpy() + test["away_rs_14"].to_numpy())
    return float(np.mean(np.abs(regression - actual))), float(np.mean(np.abs(blend - actual)))


def bins_for(frame: pd.DataFrame, split: str) -> list[dict]:
    edges = (("50-55", 0.50, 0.55), ("55-60", 0.55, 0.60), ("60-65", 0.60, 0.65), ("65-plus", 0.65, 1.01))
    rows = []
    for name, low, high in edges:
        group = frame[(frame["probability"] >= low) & (frame["probability"] < high)]
        count = int(len(group))
        if count == 0:
            rate = ""
            calibrated = "no"
            note = "no rows"
        else:
            rate = float(group["home_win"].mean())
            if count < 20:
                calibrated = "no"
                note = "count under 20"
            elif rate >= low:
                calibrated = "yes"
                note = "observed home-win rate is at least the bin floor"
            else:
                calibrated = "no"
                note = "observed home-win rate is below the bin floor"
        rows.append(
            {
                "split": split,
                "bin": name,
                "count": count,
                "home_win_rate": rate,
                "calibrated": calibrated,
                "note": note,
            }
        )
    return rows


def main() -> None:
    frame = pd.read_csv(TABLE)
    frame = frame.dropna(subset=["home_win"]).copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    train = frame[frame["game_date"] < HOLDOUT]
    test = frame[frame["game_date"] >= HOLDOUT]
    holdout_prob, holdout_pick = probabilities(train, test)
    in_sample_prob, _ = probabilities(frame, frame)
    holdout = test[["home_win"]].copy()
    holdout["probability"] = holdout_prob
    sampled = frame[["home_win"]].copy()
    sampled["probability"] = in_sample_prob
    rows = bins_for(sampled, "in_sample") + bins_for(holdout, "holdout")
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "bin", "count", "home_win_rate", "calibrated", "note"])
        writer.writeheader()
        writer.writerows(rows)
    accuracy = float((holdout_pick == test["home_win"].astype(int).to_numpy()).mean())
    calibrated_accuracy = float(((holdout_prob >= 0.5).astype(int) == test["home_win"].astype(int).to_numpy()).mean())
    total_mae, blend_mae = run_mae(train, test)
    print(f"train_rows {len(train)}")
    print(f"test_rows {len(test)}")
    print(f"holdout_accuracy {accuracy:.4f}")
    print(f"calibrated_threshold_accuracy {calibrated_accuracy:.4f}")
    print(f"total_mae {total_mae:.3f}")
    print(f"blend_mae {blend_mae:.3f}")
    print(f"features {len(FEATURES)}")
    with HIGH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date", "away", "home", "home_win_prob", "pick_side", "pick_team",
                "confidence", "confidence_tier", "predicted_away_points", "predicted_home_points", "predicted_total",
            ],
        )
        writer.writeheader()
    with REMAINING.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=REMAINING_COLUMNS).writeheader()
    print("remaining 0")


if __name__ == "__main__":
    main()
