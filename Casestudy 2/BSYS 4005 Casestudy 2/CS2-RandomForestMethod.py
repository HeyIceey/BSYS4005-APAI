# 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

def run_forecast():
    print("Made by: Group 9 - Himanish Tripathi, Ho Ngoc An, Hasan Sufi")
    

    try:
        # Load the history and the required output format
        df_history = pd.read_csv("1CS2_HistoryData.csv")
        df_output_format = pd.read_csv("1CS2-ExampleOutput.csv")
    except FileNotFoundError:
        print("Error: Files not found. Check your folder.")
        return

    # ---------------------------------------------------------
    # 2. FIX DATES
    # ---------------------------------------------------------
    # Excel stores dates as numbers (days since Dec 30, 1899).
    # We convert this number into a readable Date format (YYYY-MM-DD).
    base_date = datetime.datetime(1899, 12, 30)
    df_history['Date'] = df_history['Report_TransactionEffectiveDate'].apply(
        lambda x: base_date + datetime.timedelta(days=x)
    )

    # ---------------------------------------------------------
    # 3. GROUP BY DAY
    # ---------------------------------------------------------
    # We don't care about individual transactions, just the DAILY TOTAL.
    daily_data = df_history.groupby('Date')['TransactionAmount'].sum().reset_index()
    
    # Set Date as the index so we can fill in missing days
    daily_data.set_index('Date', inplace=True)
    daily_data = daily_data.resample('D').sum().fillna(0)

    # ---------------------------------------------------------
    # 4. CREATE "FEATURES" (Explainable Variables)
    # ---------------------------------------------------------
    # We break the date into parts so the model can learn patterns.
    # e.g., "Is it a Monday?" or "Is it December?"
    def create_features(df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek  # 0=Mon, 6=Sun (Catches weekly cycles)
        df['month'] = df.index.month          # 1-12 (Catches yearly seasonality)
        df['dayofmonth'] = df.index.day       # 1-31 (Catches paydays/end-of-month)
        df['year'] = df.index.year            # 2014-2020 (Catches long-term growth)
        return df

    df_features = create_features(daily_data)
    
    # We are predicting 'TransactionAmount'
    X = df_features.drop(columns=['TransactionAmount'])
    y = df_features['TransactionAmount']

    # ---------------------------------------------------------
    # 5. TRAIN MODEL (Random Forest)
    # ---------------------------------------------------------
    print("Training Random Forest model...")
    # n_estimators=100 means we build 100 decision trees.
    # random_state=42 ensures we get the same result every time we run it.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # ---------------------------------------------------------
    # 6. PREDICT FUTURE
    # ---------------------------------------------------------
    print("Forecasting 2021 Q1...")
    # Prepare the future dates from the template file
    forecast_dates = pd.to_datetime(df_output_format['Date'])
    forecast_df = pd.DataFrame(index=forecast_dates)
    forecast_df = create_features(forecast_df)

    # Ask the model to predict
    predictions = model.predict(forecast_df)

    # ---------------------------------------------------------
    # 7. SAVE OUTPUT
    # ---------------------------------------------------------
    output_df = df_output_format.copy()
    output_df['GroupName'] = 'Group 9'
    output_df['TotalPredictedAmount'] = predictions.round(2)

    output_df.to_csv('CS2-RandomForestResult.csv', index=False)
    print("Done! Saved to 'CS2-Forecast-Submission-Group9.csv'")

if __name__ == "__main__":
    run_forecast()