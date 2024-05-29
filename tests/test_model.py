import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_model import train_and_save_model
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model_file = 'app/model_test.pkl'
        self.scaler_file = 'app/scaler_test.pkl'
        train_and_save_model(self.model_file, self.scaler_file)
        self.model = joblib.load(self.model_file)
        self.scaler = joblib.load(self.scaler_file)
    
    def test_model_performance(self):
        # Load a sample test set
        df = pd.read_csv('data/train.csv').sample(100)
        df = handle_missing_values(df)
        df = convert_categorical_to_numeric(df)
        df = extract_date_features(df)
        df = feature_engineering(df)
        
        features = ['Sales', 'Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen']
        df[features] = self.scaler.transform(df[features])
        
        X_test = df.drop(['Sales', 'Date'], axis=1)
        y_test = df['Sales']
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        self.assertTrue(mae < 1000, f"Model performance is not satisfactory, MAE: {mae}")

if __name__ == '__main__':
    unittest.main()
