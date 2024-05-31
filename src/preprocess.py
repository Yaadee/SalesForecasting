# from sklearn.preprocessing import StandardScaler

# def feature_engineering(df):
#     df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
#                              (df['Month'] - df['CompetitionOpenSinceMonth'])
#     df['PromoOpen'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
#                       (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
#     df.fillna(0, inplace=True)
#     return df

# def scale_features(df, features):
#     scaler = StandardScaler()
#     df[features] = scaler.fit_transform(df[features])
#     return df, scaler

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

    # Select features
    features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'Weekend',
                'MonthStart', 'MonthMid', 'MonthEnd', 'DaysToHoliday', 'DaysAfterHoliday',
                'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    
    # Convert categorical features to numerical
    train = pd.get_dummies(train, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])
    test = pd.get_dummies(test, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])

    # Ensure same columns in train and test
    train, test = train.align(test, join='inner', axis=1, fill_value=0)

    # Separate features and target
    X_train = train[features]
    y_train = train['Sales']
    X_test = test[features]

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, scaler

