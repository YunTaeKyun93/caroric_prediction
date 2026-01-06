import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline
from src.modeling.models import get_xgb_model, get_lgbm_model, get_catboost_model

CONFIG = {
    "seed": 42,
    "n_folds": 5,  
    "path": {
        "train": os.path.join(project_root, "data", "raw", "train.csv"),
        "test": os.path.join(project_root, "data", "raw", "test.csv"),
        "submission": os.path.join(project_root, "outputs", "submission_stacking.csv")
    }
}

def get_oof_preds(model_func, X, y, X_test, n_folds=5):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train = np.zeros((X.shape[0],))
    oof_test = np.zeros((X_test.shape[0],))
    test_preds_fold = np.zeros((X_test.shape[0], n_folds))
    
    print(f"Running {n_folds}-Fold ")
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline = get_pipeline(model_func())
        pipeline.fit(X_tr, y_tr)
        
        oof_train[val_idx] = pipeline.predict(X_val)
        
        test_preds_fold[:, i] = pipeline.predict(X_test)
        
    oof_test = test_preds_fold.mean(axis=1)
    
    return oof_train, oof_test

def main():

    train_df = load_data(CONFIG['path']['train'])
    train_df = clean_data(train_df, is_train=True)
    train_fe = generate_features(train_df)
    
    test_df = load_data(CONFIG['path']['test'])
    test_ids = test_df["ID"]
    test_fe = generate_features(test_df)

    target = "Calories_Burned"
    X = train_fe.drop(columns=[target])
    y = train_fe[target]
    X_test = test_fe 


 
    
    print(" XGBoost")
    xgb_oof_train, xgb_oof_test = get_oof_preds(get_xgb_model, X, y, X_test)
    
    print("LightGBM")
    lgbm_oof_train, lgbm_oof_test = get_oof_preds(get_lgbm_model, X, y, X_test)
    
    print("CatBoost")
    cat_oof_train, cat_oof_test = get_oof_preds(get_catboost_model, X, y, X_test)


    
    X_meta_train = pd.DataFrame({
        "xgb": xgb_oof_train,
        "lgbm": lgbm_oof_train,
        "cat": cat_oof_train
    })
    
    X_meta_test = pd.DataFrame({
        "xgb": xgb_oof_test,
        "lgbm": lgbm_oof_test,
        "cat": cat_oof_test
    })
    
    meta_model = Ridge(alpha=10.0)
    meta_model.fit(X_meta_train, y)
    
    final_pred = meta_model.predict(X_meta_test)
    
    print("\n Stacking Weights (Ridge):")
    print(f"XGBoost : {meta_model.coef_[0]:.4f}")
    print(f"LightGBM: {meta_model.coef_[1]:.4f}")
    print(f"CatBoost: {meta_model.coef_[2]:.4f}")
    
    oof_pred_final = meta_model.predict(X_meta_train)
    oof_score = np.sqrt(mean_squared_error(y, oof_pred_final))
    print(f"\nFinal CV Score (Stacking OOF): {oof_score:.4f}")


    submission = pd.DataFrame({
        "ID": test_ids,
        "Calories_Burned": final_pred
    })
    os.makedirs(os.path.dirname(CONFIG['path']['submission']), exist_ok=True)
    submission.to_csv(CONFIG['path']['submission'], index=False)
    
    print(f"Saved: {CONFIG['path']['submission']}")

if __name__ == "__main__":
    main()