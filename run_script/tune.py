import os
import sys
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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
    
    X = df.drop(columns=["Calories_Burned"])
    y = df["Calories_Burned"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000), # 트리를 몇 개나 심을지
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True), # 얼마나 꼼꼼하게 볼지
        "max_depth": trial.suggest_int("max_depth", 4, 10), # 트리의 깊이
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10), # 노이즈에 얼마나 민감할지
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), # 데이터 샘플링 비율
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # 컬럼 샘플링 비율
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True), # L1 규제
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True), # L2 규제
        "n_jobs": -1,
        "random_state": 42,
    }

    model = XGBRegressor(**params)
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
    print(" Tuning Finished!")
    print(f" Best RMSE: {study.best_value:.4f}")
    print(" Best Parameters:")
    print(study.best_params)
    print("="*50)
    