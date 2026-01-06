# run_script/make_submission.py
import os
import sys
import pandas as pd
import numpy as np

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline
from src.modeling.models import get_xgb_model, get_lgbm_model, get_catboost_model

# ==========================================
# [설정] 찾은 최적 비율을 여기에 입력하세요!
# ==========================================
WEIGHTS = {
    "xgb": 0.2,
    "lgbm": 0.1,
    "cat": 0.7
}

CONFIG = {
    "path": {
        "train": os.path.join(project_root, "data", "raw", "train.csv"),
        "test": os.path.join(project_root, "data", "raw", "test.csv"),
        "submission": os.path.join(project_root, "outputs", "submission_final_ensemble.csv")
    }
}

def main():
    print("🚀 Start Final Training for Submission...")
    print(f"⚖️  Ensemble Weights -> XGB: {WEIGHTS['xgb']}, LGBM: {WEIGHTS['lgbm']}, CAT: {WEIGHTS['cat']}")

    # 1. 데이터 로드 (Train Full Data & Test Data)
    print("🛠️  Processing All Data...")
    
    # Train 데이터 로드 및 전처리
    train_df = load_data(CONFIG['path']['train'])
    train_df = clean_data(train_df, is_train=True)
    train_fe = generate_features(train_df)
    
    # Test 데이터 로드 및 전처리
    test_df = load_data(CONFIG['path']['test'])
    test_ids = test_df["ID"] # ID 백업
    test_fe = generate_features(test_df) # clean_data 적용 X

    # X, y 준비
    target = "Calories_Burned"
    X_train_full = train_fe.drop(columns=[target])
    y_train_full = train_fe[target]
    
    # 2. 모델 전체 데이터로 재학습 (Retrain on Full Data)
    print("\n🤖 Retraining Models on 100% Data...")
    
    # XGBoost
    print("   >> Fitting XGBoost...")
    xgb_pipeline = get_pipeline(get_xgb_model())
    xgb_pipeline.fit(X_train_full, y_train_full)
    
    # LightGBM
    print("   >> Fitting LightGBM...")
    lgbm_pipeline = get_pipeline(get_lgbm_model())
    lgbm_pipeline.fit(X_train_full, y_train_full)
    
    # CatBoost
    print("   >> Fitting CatBoost...")
    cat_pipeline = get_pipeline(get_catboost_model())
    cat_pipeline.fit(X_train_full, y_train_full)

    # 3. 예측 (Inference)
    print("\n🔮 Predicting Test Data...")
    xgb_pred = xgb_pipeline.predict(test_fe)
    lgbm_pred = lgbm_pipeline.predict(test_fe)
    cat_pred = cat_pipeline.predict(test_fe)

    # 4. 앙상블 (Weighted Blending)
    final_pred = (xgb_pred * WEIGHTS['xgb']) + \
                 (lgbm_pred * WEIGHTS['lgbm']) + \
                 (cat_pred * WEIGHTS['cat'])

    # 5. 제출 파일 저장
    print("📝 Saving Submission...")
    submission = pd.DataFrame({
        "ID": test_ids,
        "Calories_Burned": final_pred
    })
    
    os.makedirs(os.path.dirname(CONFIG['path']['submission']), exist_ok=True)
    submission.to_csv(CONFIG['path']['submission'], index=False)
    
    print("="*50)
    print(f"✨ Final Submission Saved: {CONFIG['path']['submission']}")
    print("🔥 Go submit and hit the leaderboard!")
    print("="*50)

if __name__ == "__main__":
    main()