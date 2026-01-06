import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline
from src.modeling.models import get_xgb_model, get_lgbm_model
from src.validation.metrics import rmse_score

CONFIG = {
    "seed": 42,
    "test_size": 0.2,
    "path": {
        "train": os.path.join(project_root, "data", "raw", "train.csv"),
    }
}

def main():

    raw_df = load_data(CONFIG['path']['train'])
    clean_df = clean_data(raw_df, is_train=True)
    fe_df = generate_features(clean_df)

    target = "Calories_Burned"
    X = fe_df.drop(columns=[target])
    y = fe_df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['seed']
    )

    xgb_pipeline = get_pipeline(get_xgb_model())
    xgb_pipeline.fit(X_train, y_train)
    
    lgbm_pipeline = get_pipeline(get_lgbm_model())
    lgbm_pipeline.fit(X_train, y_train)

    xgb_pred = xgb_pipeline.predict(X_val)
    lgbm_pred = lgbm_pipeline.predict(X_val)
    
    xgb_score = rmse_score(y_val, xgb_pred)
    lgbm_score = rmse_score(y_val, lgbm_pred)
    
    print(f"\n Single Model RMSE")
    print(f"    XGBoost : {xgb_score:.4f}")
    print(f"    LightGBM: {lgbm_score:.4f}")

    best_score = float('inf')
    best_ratio = 0.0
    
    for i in range(21):
        w_xgb = i * 0.05
        w_lgbm = 1.0 - w_xgb
        
        blend_pred = (xgb_pred * w_xgb) + (lgbm_pred * w_lgbm)
        score = rmse_score(y_val, blend_pred)
        
        if score < best_score:
            best_score = score
            best_ratio = w_xgb
            
    print("="*40)
    print(f" Best Ensemble RMSE: {best_score:.4f}")
    print(f" Optimal Ratio -> XGB: {best_ratio:.2f} / LGBM: {1-best_ratio:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()