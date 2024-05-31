# import pandas as pd
import logging

from sklearn.calibration import LabelEncoder
# from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, filename='logs/data_cleaning.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')

# # Function to load and merge data
# def load_and_merge_data(train_path, store_path):
#     train = pd.read_csv(train_path)
#     store = pd.read_csv(store_path)
#     df = pd.merge(train, store, on='Store')
#     logging.info('Data loaded and merged')
#     return df

# # Function to handle missing values
# def handle_missing_values(df):
#     df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
#     df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
#     df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
#     df['Promo2SinceWeek'].fillna(0, inplace=True)
#     df['Promo2SinceYear'].fillna(0, inplace=True)
#     df['PromoInterval'].fillna('None', inplace=True)
#     logging.info('Missing values handled')
#     return df

# # Function to convert categorical columns to numeric
# def convert_categorical_to_numeric(df):
#     # Identify categorical columns
#     categorical_columns = df.select_dtypes(include=['category', 'object']).columns
#     logging.info(f'Categorical columns identified: {categorical_columns}')
    
#     # Initialize label encoder
#     label_encoder = LabelEncoder()
    
#     # Convert categorical columns and cast to int32
#     for column in categorical_columns:
#         df[column] = label_encoder.fit_transform(df[column].astype(str)).astype('int32')
#         logging.info(f'Column {column} converted to numeric and cast to int32')
    
#     return df

# # Function to extract features from the date column
# def extract_date_features(df):
#     df['Date'] = pd.to_datetime(df['Date'])
#     df['Year'] = df['Date'].dt.year
#     df['Month'] = df['Date'].dt.month
#     df['Day'] = df['Date'].dt.day
#     df['WeekOfYear'] = df['Date'].dt.isocalendar().week
#     df['DayOfWeek'] = df['Date'].dt.dayofweek
#     logging.info('New features extracted from date column')
#     return df

# # Main data processing function
# def preprocess_data(train_file, store_file):
#     df = load_and_merge_data(train_file, store_file)
#     df = handle_missing_values(df)
#     df = convert_categorical_to_numeric(df)
#     df = extract_date_features(df)
#     logging.info('Data preprocessing complete')
#     return df

import pandas as pd
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, filename='logs/data_cleaning.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Function to convert categorical columns to numeric
def convert_categorical_to_numeric(df):
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    logging.info(f'Categorical columns identified: {categorical_columns}')
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    # Convert categorical columns and cast to int32
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column].astype(str)).astype('int32')
        logging.info(f'Column {column} converted to numeric and cast to int32')
    
    return df

def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    return train, test, store

def merge_data(train, test, store):
    train = pd.merge(train, store, on='Store')
    test = pd.merge(test, store, on='Store')
    return train, test

def handle_missing_values(df):
    df.fillna(df.median(), inplace=True)
    return df

