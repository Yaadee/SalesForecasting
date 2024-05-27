import matplotlib.pyplot as plt
import seaborn as sns

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
