from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load RandomForest model and scalers
rf_model = joblib.load('app/model.pkl')
scaler_features = joblib.load('app/scaler_features.pkl')
scaler_target = joblib.load('app/scaler_target.pkl')

# Load LSTM model and scalers
lstm_model = tf.keras.models.load_model('app/lstm_sales_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    store = data.get('Store')
    day_of_week = data.get('DayOfWeek')
    promo = data.get('Promo')
    school_holiday = data.get('SchoolHoliday')
    year = data.get('Year')
    month = data.get('Month')
    day = data.get('Day')
    week_of_year = data.get('WeekOfYear')
    weekday = data.get('Weekday')
    weekend = data.get('Weekend')
    month_start = data.get('MonthStart')
    month_mid = data.get('MonthMid')
    month_end = data.get('MonthEnd')
    days_to_holiday = data.get('DaysToHoliday')
    days_after_holiday = data.get('DaysAfterHoliday')
    competition_distance = data.get('CompetitionDistance')
    competition_open_since_month = data.get('CompetitionOpenSinceMonth')
    competition_open_since_year = data.get('CompetitionOpenSinceYear')
    promo2 = data.get('Promo2')
    promo2_since_week = data.get('Promo2SinceWeek')
    promo2_since_year = data.get('Promo2SinceYear')

    input_data = np.array([[
        store, day_of_week, promo, school_holiday, year, month, day, 
        week_of_year, weekday, weekend, month_start, month_mid, month_end, 
        days_to_holiday, days_after_holiday, competition_distance, 
        competition_open_since_month, competition_open_since_year, promo2, 
        promo2_since_week, promo2_since_year
    ]])

    # Preprocess input data
    input_data = scaler_features.transform(input_data)

    # Make prediction with RandomForest model
    rf_prediction = rf_model.predict(input_data)

    # Make prediction with LSTM model
    lstm_input = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    lstm_prediction = lstm_model.predict(lstm_input)
    lstm_prediction = scaler_target.inverse_transform(lstm_prediction)

    result = {
        'RandomForestPrediction': rf_prediction[0],
        'LSTMPrediction': lstm_prediction[0][0]
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
