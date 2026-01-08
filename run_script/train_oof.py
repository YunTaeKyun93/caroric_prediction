
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data

SEED = 42
N_FOLDS = 5
TARGET = "Calories_Burned"
CAT_COLS = ["Gender", "Weight_Status"]

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
    train_df = load_data(os.path.join(project_root, "data/raw/train.csv"))
    train_df = clean_data(train_df, is_train=True)

    test_df = load_data(os.path.join(project_root, "data/raw/test.csv"))
    test_ids = test_df["ID"].copy()

    X = train_df.drop(columns=["ID", TARGET])
    y = train_df[TARGET]
    y_log = np.log1p(y)

    X_test = test_df.drop(columns=["ID"])

    # -------- CV (log space) --------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"🚀 Fold {fold+1}")

        model = CatBoostRegressor(**CAT_PARAMS)
        model.fit(X.iloc[tr_idx], y_log.iloc[tr_idx])

        oof[val_idx] = model.predict(X.iloc[val_idx])

    cv_rmse = np.sqrt(mean_squared_error(y_log, oof))
    print(f"\n✅ CV RMSE (log target): {cv_rmse:.5f}")

    # -------- Train full --------
    final_model = CatBoostRegressor(**CAT_PARAMS)
    final_model.fit(X, y_log)

    test_pred = np.expm1(final_model.predict(X_test))
    test_pred = np.round(test_pred)

    submission = pd.DataFrame({
        "ID": test_ids,
        TARGET: test_pred
    })

    out_path = os.path.join(project_root, "outputs/submission_cat_log_optuna.csv")
    submission.to_csv(out_path, index=False)
    print(f"\n📦 Saved → {out_path}")

if __name__ == "__main__":
    main()
