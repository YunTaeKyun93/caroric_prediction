import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def run_kfold(
    X,
    y,
    pipeline,
    n_splits=5,
    random_state=42,
    verbose=True
):
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rmses = []


    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)

        rmse = root_mean_squared_error(y_valid, preds)
        rmses.append(rmse)

        if verbose:
            print(f"Fold {fold} RMSE: {rmse:.4f}")

    return np.mean(rmses), np.std(rmses)
