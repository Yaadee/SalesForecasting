import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_sales_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], bins=50, kde=True)
    plt.title('Sales Distribution')
    plt.show()

def plot_sales_trends_over_time(df):
    plt.figure(figsize=(15, 6))
    df.groupby('Date')['Sales'].sum().plot()
    plt.title('Total Sales Over Time')
    plt.show()

def plot_sales_during_holidays(df):
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, x='StateHoliday', y='Sales')
    plt.title('Sales during Holidays')
    plt.show()

def analyze_sales_trends(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Date', y='Sales')
    plt.title('Sales Trends Over Time')
    plt.show()

def analyze_day_of_week_sales(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='DayOfWeek', y='Sales')
    plt.title('Sales by Day of the Week')
    plt.show()

def analyze_store_types_and_assortment(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='StoreType', y='Sales')
    plt.title('Sales by Store Type')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Assortment', y='Sales')
    plt.title('Sales by Assortment')
    plt.show()

def analyze_promotions_and_competitions(df, store):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Promo', y='Sales')
    plt.title('Sales by Promotion')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales')
    plt.title('Sales vs. Competition Distance')
    plt.show()

def analyze_customer_visits(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales')
    plt.title('Sales vs. Customer Visits')
    plt.show()

if __name__ == "__main__":
    # Load your datasets
    train_path = "data/train.csv"
    train = pd.read_csv(train_path, parse_dates=['Date'])
    
    store_path = "data/store.csv"
    store = pd.read_csv(store_path)

    # Merge datasets if necessary
    train = pd.merge(train, store, on='Store')

    # Analyze data
    analyze_customer_visits(train)
    plot_sales_trends_over_time(train)
    plot_sales_distribution(train)
    plot_sales_during_holidays(train)
    analyze_promotions_and_competitions(train, store)
    analyze_store_types_and_assortment(train)
    analyze_sales_trends(train)
