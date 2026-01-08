# run_script/train_cat_final_last.py

import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features

SEED = 42
TARGET = "Calories_Burned"
CAT_COLS = [ "Weight_Status"]

CAT_PARAMS = {
    "iterations": 2581,
    "learning_rate": 0.024124340682842315,
    "depth": 5,
    "l2_leaf_reg": 0.11161639330831707,
    "subsample": 0.9177269468895219,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "verbose": 0,
    "cat_features": CAT_COLS
}

def main():
    # -------- Load & clean --------
    train_df = load_data(os.path.join(project_root, "data/raw/train.csv"))
    train_df = clean_data(train_df, is_train=True)

    test_df = load_data(os.path.join(project_root, "data/raw/test.csv"))
    test_ids = test_df["ID"].copy()

    # -------- Feature engineering --------
    train_fe = generate_features(train_df)
    test_fe = generate_features(test_df)

    X_train = train_fe.drop(columns=[ TARGET])
    y_train = train_fe[TARGET]

    X_test = test_fe.copy()

    # -------- Train CatBoost --------
    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    # ==================================================
    # ⭐ Final post-processing (핵심)
    # 1) 물리적 최대치 기반 "약한 clip"
    # 2) round
    # ==================================================
    max_possible = (
        test_fe["Weight_kg"]
        * test_fe["Exercise_Duration"]
        * 0.30      # ❗ 강하지 않은 상한 (핵심)
    )

    test_pred = np.clip(test_pred, 0, max_possible)
    test_pred = np.round(test_pred)

    # -------- Save --------
    submission = pd.DataFrame({
        "ID": test_ids,
        TARGET: test_pred.astype(int)
    })

    out_path = os.path.join(project_root, "outputs/submission_cat_final_last.csv")
    submission.to_csv(out_path, index=False)

    print(f"\n🚀 FINAL SUBMISSION SAVED → {out_path}")

if __name__ == "__main__":
    main()
