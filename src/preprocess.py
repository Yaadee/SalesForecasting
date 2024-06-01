import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date'])
    store = pd.read_csv(store_path)
    return train, test, store

def merge_data(train, test, store):
    train = pd.merge(train, store, on='Store', how='left')
    test = pd.merge(test, store, on='Store', how='left')
    return train, test

def handle_missing_values(df):
    df.fillna(0, inplace=True)
    return df

def feature_engineering(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Weekday'] = df['Date'].dt.weekday
    df['Weekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['MonthStart'] = (df['Date'].dt.day < 10).astype(int)
    df['MonthMid'] = ((df['Date'].dt.day >= 10) & (df['Date'].dt.day <= 20)).astype(int)
    df['MonthEnd'] = (df['Date'].dt.day > 20).astype(int)

    # Additional features
    df['DaysToHoliday'] = (df['StateHoliday'] != '0').astype(int)
    df['DaysAfterHoliday'] = (df['StateHoliday'] != '0').astype(int)
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['PromoOpen'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    df['PromoOpen'] = df['PromoOpen'].apply(lambda x: x if x > 0 else 0)
    return df

def preprocess_data(train, test, store):
    # Merge data with store information
    train, test = merge_data(train, test, store)

    # Handle missing values
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    

    # Feature engineering
    train = feature_engineering(train)
    test = feature_engineering(test)
   
    
    # Check if 'Sales' column exists
    if 'Sales' not in train.columns:
        raise KeyError("'Sales' column not found in training data.")

    # Select features
    features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 
                'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'Weekend',
                'MonthStart', 'MonthMid', 'MonthEnd', 'DaysToHoliday', 'DaysAfterHoliday',
                'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']

    # Convert categorical features to numerical
    train = pd.get_dummies(train, columns=['StateHoliday'])
    test = pd.get_dummies(test, columns=['StateHoliday'])
   

    # # Ensure same columns in train and test
    # train, test = train.align(test, join='inner', axis=1)
    # print("after compare test and train\n",train.columns)
    # print(test.columns)

    # Separate features and target
    X_train = train[features]
    y_train = train['Sales']  # Ensure 'Sales' column exists
    X_test = test[features]
    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, scaler

