import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import time
from preprocess import load_data, preprocess_data

logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    store_path = 'data/store.csv'

    train, test, store = load_data(train_path, test_path, store_path)
    X_train, y_train, X_test, scaler = preprocess_data(train, test, store)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    logging.info('Training RandomForestRegressor model...')
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f'Training time: {training_time} seconds')

    y_pred = model.predict(X_val)
    mse = np.mean((y_pred - y_val)**2)
    mae = np.mean(abs(y_pred - y_val))
    r2 = model.score(X_val, y_val)
    logging.info(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Save the model and scaler
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    joblib.dump((model, scaler), f'app/model_{timestamp}.pkl')
    logging.info('Model saved.')

    # Optionally save the test set for future predictions
    np.save(f'app/X_test_{timestamp}.npy', X_test)
    logging.info('Test set saved.')

if __name__ == "__main__":
    main()
