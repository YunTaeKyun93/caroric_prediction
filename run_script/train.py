# run_script/train.py
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.modeling.trainer import train_model
from src.validation.metrics import rmse_score

CONFIG = {
    "seed": 42,
    "test_size": 0.2,
    "target": "Calories_Burned",
    "model_params": {
        "n_estimators": 2575,
        'learning_rate': 0.03457740369903334,
        'min_child_weight': 8,
        "max_depth": 4,
        'reg_alpha': 0.02266666885129714,
       'subsample': 0.7379452499074695,
       'colsample_bytree': 0.6968572816213251,
        'reg_lambda': 2.417957718328564e-05,
        "random_state": 42,
        "n_jobs": -1
    },
    "path": {
        "train": os.path.join(project_root, "data", "raw", "train.csv"),
        "model_save": os.path.join(project_root, "models", "xgb_model.pkl")
    }
}

def main():
    print("스타또")

    print(f"{CONFIG['path']['train']}")
    raw_df = load_data(CONFIG['path']['train'])

    print("전처리중")
    clean_df = clean_data(raw_df, is_train=True)
    fe_df = generate_features(clean_df)


    
    X = fe_df.drop(columns=[CONFIG['target']])
    y = fe_df[CONFIG['target']]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['seed']
    )

    pipeline = train_model(X_train, y_train, params=CONFIG['model_params'])

    val_pred = pipeline.predict(X_val)
    score = rmse_score(y_val, val_pred)
    print(f"\nFinal Validation RMSE: {score:.4f}")

    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        
        num_cols = preprocessor.transformers_[0][2]
        print(preprocessor.transformers_)
      
        cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
        print(cat_cols)
        feature_names = list(num_cols) + list(cat_cols)
        
       
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        
        print(feat_imp_df.head(10))
    except Exception as e:
        print(f"⚠️ Feature importance extraction failed: {e}")

    os.makedirs(os.path.dirname(CONFIG['path']['model_save']), exist_ok=True)
    joblib.dump(pipeline, CONFIG['path']['model_save'])
    print("끝끝")

if __name__ == "__main__":
    main()