import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, filename='logs/data_cleaning.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_and_merge_data(train_path, store_path):
    data = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    df = pd.merge(data, store, on='Store')
    logging.info('Data loaded and merged')
    return df

def handle_missing_values(df):
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna(0, inplace=True)
    logging.info('Missing values handled')
    return df

def convert_categorical_to_numeric(df):
    label_encoder = LabelEncoder()
    df['StateHoliday'] = label_encoder.fit_transform(df['StateHoliday'])
    df['StoreType'] = label_encoder.fit_transform(df['StoreType'])
    df['Assortment'] = label_encoder.fit_transform(df['Assortment'])
    logging.info('Categorical variables converted to numeric')
    return df

def extract_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    logging.info('New features extracted from date column')
    return df
