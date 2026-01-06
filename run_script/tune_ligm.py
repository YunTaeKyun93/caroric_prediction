import os
import sys
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline

def objective(trial):
    train_path = os.path.join(project_root, "data", "raw", "train.csv")
    df = load_data(train_path)
    df = clean_data(df, is_train=True)
    df = generate_features(df)
    
    target = "Calories_Burned"
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150), # LGBM 핵심 (트리 복잡도)
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    model = LGBMRegressor(**params)
    pipeline = get_pipeline(model)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    
    return rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("\n" + "="*50)
    print(f" Best LGBM RMSE: {study.best_value:.4f}")
    print(" Best Parameters:")
    print(study.best_params)
    print("="*50)