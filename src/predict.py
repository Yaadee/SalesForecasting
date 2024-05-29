import pandas as pd
import joblib
from data_cleaning import load_test_data, handle_missing_values, convert_categorical_to_numeric, extract_date_features
from preprocess import feature_engineering

def predict_sales(test_path, store_path, model_path, scaler_path):
    # Load and preprocess test data
    df_test = load_test_data(test_path, store_path)
    df_test = handle_missing_values(df_test)
    df_test = convert_categorical_to_numeric(df_test)
    df_test = extract_date_features(df_test)
    df_test = feature_engineering(df_test)
    
    # Load the trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale the features
    features = ['Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen']
    df_test[features] = scaler.transform(df_test[features])
    
    # Predict sales
    X_test = df_test.drop(['Date', 'Id'], axis=1)
    df_test['Sales'] = model.predict(X_test)
    
    # Prepare submission file
    submission = df_test[['Id', 'Sales']]
    submission.to_csv('data/sample_submission.csv', index=False)
    print("Prediction and submission file saved.")

# Example usage
predict_sales('data/test.csv', 'data/store.csv', 'app/model.pkl', 'app/scaler.pkl')
