# Rossmann Pharmaceuticals Sales Forecasting

## Overview

This project aims to forecast the sales of Rossmann Pharmaceuticals stores using both traditional machine learning and deep learning approaches. The repository includes data preprocessing, model training, and deployment of the model using Flask.

## Repository Structure

- `data/`: Contains the training and test data files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model building.
- `src/`: Source code for data cleaning, preprocessing, model training, and utility functions.
- `app/`: Flask application for serving the model predictions.
- `logs/`: Log files for tracking data cleaning and other processes.
- `README.md`: Project overview and setup instructions.
- `requirements.txt`: Required Python packages.
- `.gitignore`: Files and directories to ignore in the Git repository.

## Setup Instructions

1. Clone the repository:

    ```sh
    git clone https://github.com/Yaadee/SalesForecasting.git
    cd RossmannSalesForecasting
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Prepare the data and train the models:

    ```sh
    python src/train_model.py
    python src/train_lstm.py
    ```

4. Start the Flask application:

    ```sh
    python app/app.py
    ```

5. Use the Flask API to get sales predictions:

    ```sh
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Store": 1, "DayOfWeek": 4, "Customers": 300, "Open": 1, "Promo": 1, "StateHoliday": 0, "SchoolHoliday": 0, "Year": 2015, "Month": 7, "Day": 31, "WeekOfYear": 31, "CompetitionDistance": 200, "CompetitionOpenSinceMonth": 9, "CompetitionOpenSinceYear": 2008, "Promo2": 1, "Promo2SinceWeek": 13, "Promo2SinceYear": 2010, "PromoInterval": 0, "StoreType": 1, "Assortment": 1}'
    ```

## Contributing

Feel free to open issues or submit pull requests with improvements or bug fixes. Any contributions are highly appreciated!
