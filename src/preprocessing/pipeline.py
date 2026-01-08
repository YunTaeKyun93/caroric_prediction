from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector 
def get_pipeline(model):

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=['object', 'category']))
        ],
        remainder='drop' 
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def get_cat_pipeline(model):
    pipeline = Pipeline([
        ("model", model)
    ])
    return pipeline