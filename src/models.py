from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import time

def build_model():
    pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    return grid_search

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    return mse, mae, r2

def save_model(model, scaler, filename):
    joblib.dump({'model': model, 'scaler': scaler}, filename)
