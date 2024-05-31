import numpy as np
import pandas as pd
import logging
import time
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(train_path, test_path, store_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    return train, test, store

def preprocess_data(train, test):
    # Merge with store data
    store = pd.read_csv('data/store.csv')
    train = train.merge(store, on='Store')
    test = test.merge(store, on='Store')

    # Convert date to datetime
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    # Extract features from date
    for df in [train, test]:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Weekday'] = df['Date'].dt.weekday
        df['Weekend'] = (df['Date'].dt.weekday >= 5).astype(int)
        df['MonthStart'] = (df['Day'] <= 10).astype(int)
        df['MonthMid'] = ((df['Day'] > 10) & (df['Day'] <= 20)).astype(int)
        df['MonthEnd'] = (df['Day'] > 20).astype(int)
        # Assume holidays data is available
        df['DaysToHoliday'] = np.random.randint(0, 30, size=len(df))
        df['DaysAfterHoliday'] = np.random.randint(0, 30, size=len(df))

    # Encode categorical variables
    train = pd.get_dummies(train, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])
    test = pd.get_dummies(test, columns=['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'])

    # Fill NaN values
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    return train, test

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

def train_lstm_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=200, verbose=0)
    return model

def evaluate_lstm_model(model, X_val, y_val, scaler_target):
    predictions = model.predict(X_val)
    predictions = scaler_target.inverse_transform(predictions)
    y_val = scaler_target.inverse_transform(y_val.reshape(-1, 1))

    mse = np.mean((predictions - y_val) ** 2)
    mae = np.mean(np.abs(predictions - y_val))
    r2 = 1 - (np.sum((predictions - y_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    return mse, mae, r2

def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    store_path = 'data/store.csv'

    train, test, store = load_data(train_path, test_path, store_path)
    train, test = preprocess_data(train, test)

    features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 
                'Year', 'Month', 'Day', 'WeekOfYear', 'Weekday', 'Weekend',
                'MonthStart', 'MonthMid', 'MonthEnd', 'DaysToHoliday', 'DaysAfterHoliday',
                'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear']
    target = 'Sales'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(train[features])
    scaled_target = scaler_target.fit_transform(train[[target]])

    SEQ_LENGTH = 30
    X, y = create_sequences(scaled_features, SEQ_LENGTH)
    y = y[:, 0]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    logging.info('Training LSTM model...')
    start_time = time.time()
    model = train_lstm_model(model, X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f'Training time: {training_time} seconds')

    mse, mae, r2 = evaluate_lstm_model(model, X_val, y_val, scaler_target)
    logging.info(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    model.save('app/lstm_sales_model.h5')
    joblib.dump(scaler_features, 'app/scaler_features.pkl')
    joblib.dump(scaler_target, 'app/scaler_target.pkl')
    logging.info('LSTM model saved.')

if __name__ == "__main__":
    main()
