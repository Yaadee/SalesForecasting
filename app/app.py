from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('app/model.pkl')
scaler = joblib.load('app/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    
    # Feature engineering and scaling
    input_data = feature_engineering(input_data)
    features = ['Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen']
    input_data[features] = scaler.transform(input_data[features])
    
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
