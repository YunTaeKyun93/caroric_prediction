from xgboost import XGBRegressor
from src.preprocessing.pipeline import get_pipeline

def train_model(X_train, y_train, params=None):
    """
    파라미터를 받아 모델을 생성하고 학습시킨 뒤,
    학습된 파이프라인을 반환합니다.
    """
    if params is None:
        params = {}
        
    model = XGBRegressor(**params)
    
    pipeline = get_pipeline(model)
    
    pipeline.fit(X_train, y_train)
    
    return pipeline