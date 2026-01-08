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
from src.modeling.models import (
    get_xgb_model,
    get_lgbm_model,
    get_catboost_with_features_model
)

CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "path": {
        "train": os.path.join(project_root, "data", "raw", "train.csv"),
        "test": os.path.join(project_root, "data", "raw", "test.csv"),
        "submission": os.path.join(
            project_root,
            "outputs",
            "submission_stacking_round_only.csv"
        )
    }
}

np.random.seed(CONFIG["seed"])

TARGET = "Calories_Burned"
CAT_COLS = ["Gender", "Weight_Status"]


def get_oof_preds(
    model_func,
    X,
    y,
    X_test,
    model_name=None,
    cat_cols=None
):
    kf = KFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"]
    )
    oof_train = np.zeros(len(X))
    test_preds = np.zeros((len(X_test), CONFIG["n_folds"]))

    print(f"Running OOF → {model_name}")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y.iloc[tr_idx]

        if model_name == "cat":
            model = model_func(cat_cols)
            model.fit(X_tr, y_tr)
            oof_train[val_idx] = model.predict(X_val)
            test_preds[:, fold] = model.predict(X_test)
        else:
            pipeline = get_pipeline(model_func())
            pipeline.fit(X_tr, y_tr)
            oof_train[val_idx] = pipeline.predict(X_val)
            test_preds[:, fold] = pipeline.predict(X_test)

    return oof_train, test_preds.mean(axis=1)


def main():
    train_df = load_data(CONFIG["path"]["train"])
    train_df = clean_data(train_df, is_train=True)

    test_df = load_data(CONFIG["path"]["test"])
    test_df = clean_data(test_df, is_train=False)
    test_ids = test_df["ID"].copy()

    y = train_df[TARGET]

    train_fe = generate_features(train_df)
    test_fe = generate_features(test_df)

    X = train_fe.drop(columns=[TARGET])
    X_test = test_fe

    X_cat = train_df.drop(columns=[TARGET, "ID"])
    X_cat_test = test_df.drop(columns=["ID"])

    assert all(col in X_cat.columns for col in CAT_COLS), "❌ cat column missing"

    xgb_tr, xgb_te = get_oof_preds(get_xgb_model, X, y, X_test, model_name="xgb")
    lgbm_tr, lgbm_te = get_oof_preds(get_lgbm_model, X, y, X_test, model_name="lgbm")
    cat_tr, cat_te = get_oof_preds(
        get_catboost_with_features_model,
        X_cat,
        y,
        X_cat_test,
        model_name="cat",
        cat_cols=CAT_COLS
    )

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

    kf_meta = KFold(
        n_splits=CONFIG["n_folds"],
        shuffle=True,
        random_state=CONFIG["seed"]
    )
    oof_meta = np.zeros(len(y))

    for tr_idx, val_idx in kf_meta.split(X_meta_train):
        meta = Ridge(alpha=10.0)
        meta.fit(X_meta_train.iloc[tr_idx], y.iloc[tr_idx])
        oof_meta[val_idx] = meta.predict(X_meta_train.iloc[val_idx])

    final_cv = np.sqrt(mean_squared_error(y, oof_meta))
    print(f"\n✅ Final CV Score (Stacking OOF): {final_cv:.5f}")

    final_meta = Ridge(alpha=10.0)
    final_meta.fit(X_meta_train, y)

    print("\nStacking Weights:")
    print(dict(zip(X_meta_train.columns, final_meta.coef_)))

    final_pred = final_meta.predict(X_meta_test)

    final_pred = np.round(final_pred)

    submission = pd.DataFrame({
        "ID": test_ids,
        TARGET: final_pred
    })

    os.makedirs(os.path.dirname(CONFIG["path"]["submission"]), exist_ok=True)
    submission.to_csv(CONFIG["path"]["submission"], index=False)
    print(f"Saved → {CONFIG['path']['submission']}")


if __name__ == "__main__":
    main()
