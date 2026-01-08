import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# -------------------------------------------------
# Path
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# -------------------------------------------------
# Internal modules
# -------------------------------------------------
from src.features.build_features import load_data, clean_data, generate_features
from src.preprocessing.pipeline import get_pipeline
from src.modeling.models import (
    get_xgb_model,
    get_lgbm_model,
    get_catboost_with_features_model
)

# -------------------------------------------------
# Config
# -------------------------------------------------
CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "train_path": os.path.join(project_root, "data", "raw", "train.csv"),
    "test_path": os.path.join(project_root, "data", "raw", "test.csv"),
    "output_path": os.path.join(project_root, "outputs", "submission_stacking.csv"),
}

TARGET = "Calories_Burned"
CAT_COLS = ["Gender", "Weight_Status"]

np.random.seed(CONFIG["seed"])

# -------------------------------------------------
# OOF function (CatBoost 분기 완전 분리)
# -------------------------------------------------
def get_oof_preds(model_func, X, y, X_test, is_cat=False):

    kf = KFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"]
    )

    oof_train = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), CONFIG["n_folds"]))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold + 1}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]

        if is_cat:
            # ✅ CatBoost는 pipeline 절대 사용 ❌
            model = model_func()
            model.fit(X_tr, y_tr)

            oof_train[val_idx] = model.predict(X_val)
            test_preds[:, fold] = model.predict(X_test)

            # Feature importance (첫 fold만)
            if fold == 0:
                fi = pd.DataFrame({
                    "feature": X_tr.columns,
                    "importance": model.get_feature_importance()
                }).sort_values("importance", ascending=False)

                print("\n🔥 CatBoost Feature Importance (Top 20)")
                print(fi.head(20))
                print("-" * 60)

        else:
            # ✅ XGB / LGBM only
            pipeline = get_pipeline(model_func())
            pipeline.fit(X_tr, y_tr)

            oof_train[val_idx] = pipeline.predict(X_val)
            test_preds[:, fold] = pipeline.predict(X_test)

    return oof_train, test_preds.mean(axis=1)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    # -----------------------------
    # Load & clean
    # -----------------------------
    train_df = load_data(CONFIG["train_path"])
    train_df = clean_data(train_df, is_train=True)

    test_df = load_data(CONFIG["test_path"])
    test_ids = test_df["ID"]

    y = train_df[TARGET]

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    # XGB / LGBM용
    train_fe = generate_features(train_df)
    test_fe = generate_features(test_df)

    X = train_fe.drop(columns=[TARGET])
    X_test = test_fe

    # CatBoost용 (raw categorical 유지)
    X_cat = train_df.drop(columns=[TARGET, "ID"])
    X_cat_test = test_df.drop(columns=["ID"])

    # -----------------------------
    # Base models
    # -----------------------------
    print("\n🚀 XGBoost")
    xgb_tr, xgb_te = get_oof_preds(
        get_xgb_model, X, y, X_test
    )

    print("\n🚀 LightGBM")
    lgbm_tr, lgbm_te = get_oof_preds(
        get_lgbm_model, X, y, X_test
    )

    print("\n🚀 CatBoost")
    cat_model_func = get_catboost_with_features_model(CAT_COLS)
    cat_tr, cat_te = get_oof_preds(
        cat_model_func,
        X_cat,
        y,
        X_cat_test,
        is_cat=True
    )

    # -----------------------------
    # Stacking
    # -----------------------------
    X_meta_train = pd.DataFrame({
        "xgb": xgb_tr,
        "lgbm": lgbm_tr,
        "cat": cat_tr
    })

    X_meta_test = pd.DataFrame({
        "xgb": xgb_te,
        "lgbm": lgbm_te,
        "cat": cat_te
    })

    meta = Ridge(alpha=10.0)
    meta.fit(X_meta_train, y)

    oof_pred = meta.predict(X_meta_train)
    cv_score = np.sqrt(mean_squared_error(y, oof_pred))

    print("\n🎯 Stacking Weights")
    for col, w in zip(X_meta_train.columns, meta.coef_):
        print(f"{col}: {w:.4f}")

    print(f"\n✅ Final CV Score (Stacking OOF): {cv_score:.5f}")

    # -----------------------------
    # Final prediction
    # -----------------------------
    final_pred = meta.predict(X_meta_test)

    submission = pd.DataFrame({
        "ID": test_ids,
        TARGET: np.round(final_pred)
    })

    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    submission.to_csv(CONFIG["output_path"], index=False)

    print(f"\n📁 Saved → {CONFIG['output_path']}")

# -------------------------------------------------
if __name__ == "__main__":
    main()
