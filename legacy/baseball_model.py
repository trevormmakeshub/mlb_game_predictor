import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent


def stats_path() -> Path:
    for path in (HERE / "stats.csv", Path.cwd() / "stats.csv"):
        if path.is_file():
            return path
    raise SystemExit(
        "stats.csv is missing from legacy/ and the working directory. "
        "Pre-2025 rows were dropped. Run python scripts/refresh_today.py"
    )


def main() -> None:
    data = pd.read_csv(stats_path())
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    data["Runs/Game"] = data["Total Runs"] / data["Total Games"]
    data = data.drop_duplicates(keep="first")

    hitting_stats = [
        "AVG", "OBP", "SLG", "WRC+", "WAR", "K Percentage", "BB Percentage", "BSR",
        "AVG/5 Players", "OBP/5 Players", "SLG/5 Players", "WAR/5 Players", "WRC+/5 Players",
        "K Percentage/5 Players", "BB Percentage/5 Players", "AVG/Week", "OBP/Week", "SLG/Week",
        "WAR/Week", "WRC+/Week", "K Percentage/Week", "BB Percentage/Week", "Runs/Game",
    ]
    pitching_stats = [
        "Opposing K/9", "Opposing HR/9", "Opposing BB/9", "ERA", "Opposing War",
        "Opposing K/9/5 Players", "Opposing BB/9/5 Players", "ERA/5 Players", "Opposing WAR/5 Players",
        "Opposing K/9/Week", "Opposing BB/9/Week", "ERA/Week", "Opposing WAR/Week",
    ]

    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    features = hitting_stats + pitching_stats
    drop_cols = ["Runs Scored", "Win?", "Date", "Offensive Team", "Defensive Team", "Total Games", "Total Runs", "RBIs"]
    X_train = train.drop(drop_cols, axis=1)[features]
    X_test = test.drop(drop_cols, axis=1)[features]
    y_train = train["Win?"]
    y_test = test["Win?"]

    win_model = RidgeClassifier()
    win_model.fit(X_train, y_train)
    predictions = win_model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    with open(HERE / "win_model.pkl", "wb") as file:
        pickle.dump(win_model, file)

    run_model = LinearRegression()
    y_train = train["Runs Scored"]
    y_test = test["Runs Scored"]
    run_model.fit(X_train, y_train)
    predictions = run_model.predict(X_test)
    with open(HERE / "run_model.pkl", "wb") as file:
        pickle.dump(run_model, file)

    print(mean_absolute_error(y_test, predictions))
    print(mean_squared_error(y_test, predictions))
    print(r2_score(y_test, predictions))


if __name__ == "__main__":
    main()
