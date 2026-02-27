import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import datetime

def run_forecast():
    print("Loading data...")
    # 1. Load Data
    try:
        df_history = pd.read_csv("1CS2_HistoryData.csv")
        df_output_format = pd.read_csv("1CS2-ExampleOutput.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the CSV files are in the same directory as this script.")
        return

    # 2. Date Conversion & Aggregation
    print("Processing dates...")
    # Excel serial date conversion (Base date 1899-12-30)
    base_date = datetime.datetime(1899, 12, 30)
    df_history['Date'] = df_history['Report_TransactionEffectiveDate'].apply(
        lambda x: base_date + datetime.timedelta(days=x)
    )

    # Aggregate by Date (Sum transaction amounts per day)
    daily_data = df_history.groupby('Date')['TransactionAmount'].sum().reset_index()
    daily_data.set_index('Date', inplace=True)
    
    # Resample to ensure every day exists (fill missing days with 0)
    daily_data = daily_data.resample('D').sum().fillna(0)

    # 3. Feature Engineering
    print("Engineering features...")
    def create_features(df):
        df = df.copy()
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
        return df

    df_features = create_features(daily_data)
    target_col = 'TransactionAmount'

    # 4. Model Training
    print("Training Gradient Boosting Model (this might take a moment)...")
    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    # Model Parameters (Tuned for general time-series stability)
    model = GradientBoostingRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X, y)

    # 5. Generate Forecast
    print("Generating forecast for 2021 Q1...")
    forecast_dates = pd.to_datetime(df_output_format['Date'])
    forecast_df = pd.DataFrame(index=forecast_dates)
    forecast_df = create_features(forecast_df)

    predictions = model.predict(forecast_df)

    # 6. Save Output
    output_df = df_output_format.copy()
    output_df['TotalPredictedAmount'] = predictions
    output_df['TotalPredictedAmount'] = output_df['TotalPredictedAmount'].round(2)

    output_filename = 'CS2-GradientBoostingResult.csv'
    output_df.to_csv(output_filename, index=False)
    print(f"SUCCESS! Output saved to: {output_filename}")

if __name__ == "__main__":
    run_forecast()