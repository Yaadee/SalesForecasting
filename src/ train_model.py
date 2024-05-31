import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess_data
from models import build_model, evaluate_model, save_model
import logging
import time

logging.basicConfig(filename='logs/training.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    store_path = 'data/store.csv'

    train, test, store = load_data(train_path, test_path, store_path)
    X_train, y_train, X_test, scaler = preprocess_data(train, test, store)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model()

    logging.info('Training RandomForestRegressor model...')
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f'Training time: {training_time} seconds')

    # Evaluate the model
    mse, mae, r2 = evaluate_model(model, X_val, y_val)
    logging.info(f'MSE: {mse}, MAE: {mae}, R2: {r2}')

    # Save the model
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_model(model, scaler, f'app/model_{timestamp}.pkl')
    logging.info('Model saved.')

if __name__ == "__main__":
    main()
