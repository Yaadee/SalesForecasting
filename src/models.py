from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def build_model_pipeline():
    pipeline = Pipeline([
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return pipeline
