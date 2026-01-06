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
from src.modeling.models import get_xgb_model, get_lgbm_model, get_catboost_model
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

    cat_pipeline = get_pipeline(get_catboost_model())
    cat_pipeline.fit(X_train, y_train)

    xgb_pred = xgb_pipeline.predict(X_val)
    lgbm_pred = lgbm_pipeline.predict(X_val)
    cat_pred = cat_pipeline.predict(X_val)
    
    print(f"\n Single Model RMSE")
    print(f"   XGBoost : {rmse_score(y_val, xgb_pred):.4f}")
    print(f"   LightGBM: {rmse_score(y_val, lgbm_pred):.4f}")
    print(f"   CatBoost: {rmse_score(y_val, cat_pred):.4f}")

    best_score = float('inf')
    best_ratio = (0, 0, 0)
    
    steps = [i/10 for i in range(11)]
### 여기서 
    for wx in steps:
        for wl in steps:
            wc = 1.0 - wx - wl
            if wc < 0 or wc > 1.0:
                continue
            
            blend_pred = (xgb_pred * wx) + (lgbm_pred * wl) + (cat_pred * wc)
            score = rmse_score(y_val, blend_pred)
            
            if score < best_score:
                best_score = score
                best_ratio = (wx, wl, wc)

    print("="*50)
    print(f"Best Triple Ensemble RMSE: {best_score:.4f}")
    print(f"Optimal Ratio -> XGB: {best_ratio[0]:.1f} / LGBM: {best_ratio[1]:.1f} / CAT: {best_ratio[2]:.1f}")
    print("="*50)



### 여기서 

if __name__ == "__main__":
    main()



    # 2.0 => 단독 
    # 앙상블 1.6
    # 트리플 1.0
    # 스태킹  0.6 => optuna

    # ROund , PolynomialFeatures, All_data