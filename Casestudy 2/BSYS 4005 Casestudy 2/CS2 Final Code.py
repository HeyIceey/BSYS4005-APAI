# BSYS 4005 Applied AIM - Casestudy 2 - January 27, 2026
# CS2: Himanish Tripathi, Ho Ngoc An, Hasan Sufi

# Imports
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor # found this, "AI Model & Suggestion"
import datetime # calendar 
import time # literally for one silly function :)

print("Made by: Group 9 - Himanish Tripathi, Ho Ngoc An, Hasan Sufi")
time.sleep(2)

# 1 Read
def run_forecast():
    try: # just in case of missing source files
        # checks the history file & output file's csv format requirement

        df_history = pd.read_csv("1CS2_HistoryData.csv") # history file
        df_output_format = pd.read_csv("1CS2-ExampleOutput.csv") # literally the template for output
    except FileNotFoundError: # just in case but not necessary
        print("File not in same folder. ERROR.")
        return
   

# 2 Dates
    # change format to yyyy-mm-dd | # Excel date system starts from Dec 30, 1899 (intersting)
    base_date = datetime.datetime(1899, 12, 30) 
    
    # the column "Report_TransactionEffectiveDate" is in days, so we convert it to actual dates
    df_history['Date'] = df_history['Report_TransactionEffectiveDate'].apply(
        lambda x: base_date + datetime.timedelta(days=x)
    )

# 3 Aggerate: Dailys
    # individual transactions < DAILY TOTAL (so adds up) like a summary each day
    # net flow basically
    # group.by('Date') puts all a same date together, 
    # then sum() adds up the amounts for that date (deposit+withdrawal=netflow)
    
    daily_data = df_history.groupby('Date')['TransactionAmount'].sum().reset_index() 
    # ------------- RECHECK <->

    # .set_index makes 'Date' the "ID" for each row
    daily_data.set_index('Date', inplace=True)

    # .resample('D') makes sure every day is represented (even if no transactions)
    # .fillna(0) fills in missing days with 0 transactions (Eg. weekends)
    daily_data = daily_data.resample('D').sum().fillna(0)

# 4 The "AI" --------------- MAYBE RECHECK
    # we break the date into parts so the model can learn patterns. (Like: is it monday, is it december? etc)
    def create_features(df):
        df = df.copy() # so original stays intact
        df['dayofweek'] = df.index.dayofweek  # 0=Mon, 6=Sun (picks weekly cycles)
        df['month'] = df.index.month          # 1-12 (picks yearly "seasonality")
        df['dayofmonth'] = df.index.day       # 1-31 (picks end-of-month)
        df['year'] = df.index.year            # 2014-2020 (Catches long-term growth)
        return df

    df_features = create_features(daily_data)
    
    # We are predicting 'TransactionAmount'
    X = df_features.drop(columns=['TransactionAmount'])
    y = df_features['TransactionAmount']

# RandomForest Model [Train]
    print("Training Random Forest model...")
    # n_estimators=100 means 100 decision trees.
    # random_state=42 ensures same result every time we run it.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

# Predict of 2021 Q1
    print("Forecasting 2021 Q1...")
    time.sleep(1)

    forecast_dates = pd.to_datetime(df_output_format['Date'])
    forecast_df = pd.DataFrame(index=forecast_dates)
    forecast_df = create_features(forecast_df)

    # Ask the model to predict
    predictions = model.predict(forecast_df)

# Save & Make CSV
    output_df = df_output_format.copy()
    output_df['GroupName'] = 'Group 9' #renames it here
    output_df['TotalPredictedAmount'] = predictions.round(2)

    output_df.to_csv('CS2-FinalResults.csv', index=False)
    print("Done! Saved to 'CS2-Forecast-Submission-Group9.csv'")

if __name__ == "__main__":
    run_forecast()