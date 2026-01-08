import os
import json
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(os.path.dirname(current_dir)) 
PARAMS_PATH = os.path.join(project_root, "models", "best_params.json")

def load_params(model_name):
    """JSON 파일에서 최적 파라미터를 불러오는 함수"""
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, 'r') as f:
                data = json.load(f)
                # print(data)
            return data.get(model_name, {})
        except Exception as e:
            print(f" Failed  load params  {model_name}: {e}")
            return {}
    return {}

def get_xgb_model(params=None):
    
    if params is None:
       
        tuned_params = load_params("xgb")
        
        default_params = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1
        }
        default_params.update(tuned_params)
        params = default_params
        
    return XGBRegressor(**params)

def get_lgbm_model(params=None):
    if params is None:
        tuned_params = load_params("lgbm")
        default_params = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        default_params.update(tuned_params)
        params = default_params
        
    return LGBMRegressor(**params)

def get_catboost_model(params=None):
    if params is None:
        tuned_params = load_params("cat")
        default_params = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42,
        }
        default_params.update(tuned_params)
        params = default_params
        
    return CatBoostRegressor(**params)

def get_catboost_with_features_model(cat_features):
    tuned_params = load_params("cat")
    default_params = {
        "iterations": 2000,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "cat_features": cat_features
    }
    default_params.update(tuned_params)

    def _model():
        return CatBoostRegressor(**default_params)

    return _model