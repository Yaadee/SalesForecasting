import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, filename='logs/data_cleaning.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to load and merge data
def load_and_merge_data(train_path, store_path):
    train = pd.read_csv(train_path, parse_dates=['Date'])
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, on='Store')
    logging.info('Data loaded and merged')
    return df

# Function to handle missing values
def handle_missing_values(df):
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    logging.info('Missing values handled')
    return df

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

# Function to extract features from the date column
def extract_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    logging.info('New features extracted from date column')
    return df

# Function to preprocess data
def preprocess_data(train_path, test_path, store_path):
    train, test, store = load_data(train_path, test_path, store_path)
    train, test = merge_data(train, test, store)
    
    # Extract date features first before handling missing values and encoding
    train = extract_date_features(train)
    test = extract_date_features(test)
    
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    train = convert_categorical_to_numeric(train)
    test = convert_categorical_to_numeric(test)
    
    logging.info('Data preprocessing complete')
    return train, test

# Function to load data
def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date'])
    store = pd.read_csv(store_path)
    return train, test, store

# Function to merge data
def merge_data(train, test, store):
    train = pd.merge(train, store, on='Store', how='left')
    test = pd.merge(test, store, on='Store', how='left')
    return train, test

# Main function
def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    store_path = 'data/store.csv'

    # Preprocess data
    train, test = preprocess_data(train_path, test_path, store_path)

    # Save processed data
    train.to_csv('data/processed/processed_train_data.csv', index=False)
    test.to_csv('data/processed/processed_test_data.csv', index=False)
    logging.info('Processed data saved to CSV files')

    # Exploratory data analysis can be added here if needed
    # analyze_sales_trends(train)
    # analyze_day_of_week_sales(train)
    # analyze_store_types_and_assortment(train)
    # analyze_promotions_and_competitions(train)
    # analyze_customer_visits(train)

if __name__ == "__main__":
    main()
