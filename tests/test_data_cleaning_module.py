import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_cleaning import load_and_merge_data, handle_missing_values, convert_categorical_to_numeric, extract_date_features

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.df = load_and_merge_data('data/train.csv', 'data/store.csv')
    
    def test_handle_missing_values(self):
        df_cleaned = handle_missing_values(self.df)
        self.assertFalse(df_cleaned.isnull().sum().any(), "There are still missing values in the dataframe")
    
    def test_convert_categorical_to_numeric(self):
        df_cleaned = convert_categorical_to_numeric(self.df)
        self.assertTrue(df_cleaned['StateHoliday'].dtype == 'int32', "StateHoliday column is not converted to numeric")
        self.assertTrue(df_cleaned['StoreType'].dtype == 'int32', "StoreType column is not converted to numeric")
        self.assertTrue(df_cleaned['Assortment'].dtype == 'int32', "Assortment column is not converted to numeric")
    
    def test_extract_date_features(self):
        df_cleaned = extract_date_features(self.df)
        self.assertTrue('Year' in df_cleaned.columns, "Year column not extracted from Date")
        self.assertTrue('Month' in df_cleaned.columns, "Month column not extracted from Date")
        self.assertTrue('Day' in df_cleaned.columns, "Day column not extracted from Date")
        self.assertTrue('WeekOfYear' in df_cleaned.columns, "WeekOfYear column not extracted from Date")
        self.assertTrue('DayOfWeek' in df_cleaned.columns, "DayOfWeek column not extracted from Date")

if __name__ == '__main__':
    unittest.main()
