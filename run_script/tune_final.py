import os
import sys
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline

PARAMS_SAVE_PATH = os.path.join(project_root, "models", "best_params.json")

def get_data():
    train_path = os.path.join(project_root, "data", "raw", "train.csv")
    df = load_data(train_path)
    df = clean_data(df, is_train=True)
    df = generate_features(df)
    
    target = "Calories_Burned"
    X = df.drop(columns=[target])
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial, model_name):
    X_train, X_val, y_train, y_val = get_data()

    if model_name == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "n_jobs": -1,
            "random_state": 42
        }
        model = XGBRegressor(**params)

    elif model_name == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1
        }
        model = LGBMRegressor(**params)

    elif model_name == "cat":
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42
        }
        model = CatBoostRegressor(**params)

    pipeline = get_pipeline(model)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

def main():
    if os.path.exists(PARAMS_SAVE_PATH):
        with open(PARAMS_SAVE_PATH, 'r') as f:
            best_params_all = json.load(f)
    else:
        best_params_all = {}

    models_to_tune = ["xgb", "lgbm", "cat"]
    

    for model_name in models_to_tune:
        print(f"\n>> Tunning {model_name.upper()}!!@!@!@")
        
        n_trials = 20 if model_name == "cat" else 30
        
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
        
        print(f" {model_name.upper()} Best RMSE: {study.best_value:.4f}")
        
        best_params_all[model_name] = study.best_params

    os.makedirs(os.path.dirname(PARAMS_SAVE_PATH), exist_ok=True)
    with open(PARAMS_SAVE_PATH, 'w') as f:
        json.dump(best_params_all, f, indent=4)
        
    print("\n" + "="*50)
    print(f" All best parameters saved to: {PARAMS_SAVE_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()