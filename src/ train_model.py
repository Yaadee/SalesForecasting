import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from data_cleaning import load_and_merge_data, handle_missing_values, convert_categorical_to_numeric, extract_date_features
from preprocess import feature_engineering, scale_features
from models import build_model_pipeline

# Load and preprocess data
df = load_and_merge_data('data/train.csv', 'data/store.csv')
df = handle_missing_values(df)
df = convert_categorical_to_numeric(df)
df = extract_date_features(df)
df = feature_engineering(df)
df, scaler = scale_features(df, ['Sales', 'Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen'])

# Split data
X = df.drop(['Sales', 'Date'], axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline = build_model_pipeline()
pipeline.fit(X_train, y_train)

# Evaluate model
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Mean Absolute Error: {mae}')

# Save model
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
joblib.dump(pipeline, f'app/model_{timestamp}.pkl')
