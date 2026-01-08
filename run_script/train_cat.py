import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data

TARGET = "Calories_Burned"
CAT_COLS = ["Gender", "Weight_Status"]

BEST_PARAMS = {
    "iterations": 2581,
    "learning_rate": 0.024124340682842315,
    "depth": 5,
    "l2_leaf_reg": 0.11161639330831707,
    "subsample": 0.9177269468895219,
    "loss_function": "RMSE",
    "verbose": 0,
    "random_seed": 42,
    "cat_features": CAT_COLS
}

def main():
    train_df = load_data(os.path.join(project_root, "data", "raw", "train.csv"))
    train_df = clean_data(train_df, is_train=True)

    test_df = load_data(os.path.join(project_root, "data", "raw", "test.csv"))
    test_df = clean_data(test_df, is_train=False)

    X_train = train_df.drop(columns=["ID", TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=["ID"])

    model = CatBoostRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    test_pred = np.round(test_pred)
    test_pred = np.clip(test_pred, y_train.min(), y_train.max())

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        TARGET: test_pred
    })

    output_path = os.path.join(project_root, "outputs", "submission_cat_round.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f" Saved submission → {output_path}")

if __name__ == "__main__":
    main()
