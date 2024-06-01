import pandas as pd
from data_cleaning import load_and_merge_data, handle_missing_values, convert_categorical_to_numeric, extract_date_features
from exploratory_analysis import analyze_sales_trends, analyze_day_of_week_sales, analyze_store_types_and_assortment, analyze_promotions_and_competitions, analyze_customer_visits
from feature_engineering import create_new_features

def main():
    # Load and merge data
    df = load_and_merge_data('data/train.csv', 'data/store.csv')

    # Handle missing values
    df = handle_missing_values(df)

    # Convert categorical to numeric
    df = convert_categorical_to_numeric(df)

    # Extract date features
    df = extract_date_features(df)

    # Feature engineering
    df = create_new_features(df)

    # Save processed data
    df.to_csv('data/processed/processed_data.csv', index=False)

    # Exploratory data analysis
    analyze_sales_trends(df)
    analyze_day_of_week_sales(df)
    analyze_store_types_and_assortment(df)
    analyze_promotions_and_competitions(df)
    analyze_customer_visits(df)

if __name__ == "__main__":
    main()
