import numpy as np
from sklearn.metrics import mean_squared_error
"""
RMSE 계산기용
"""
def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)