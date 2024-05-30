import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def load_processed_data(filepath):
    return pd.read_csv(filepath)

def plot_skewness_kurtosis(df, column):
    # Calculate skewness and kurtosis
    skewness = df[column].skew()
    kurtosis = df[column].kurt()

    # Plot histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(df[column], kde=True)
    
    plt.title(f'Histogram of {column}\nSkewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    plt.show()

def main():
    # Load processed data
    df = load_processed_data('data/processed/processed_data.csv')

    # Columns to analyze
    columns_to_analyze = ['Sales', 'Customers', 'CompetitionDistance', 'PromoOpen', 'CompetitionOpen']

    # Plot skewness and kurtosis for each column
    for column in columns_to_analyze:
        plot_skewness_kurtosis(df, column)

if __name__ == "__main__":
    main()
