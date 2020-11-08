""" 1.0 Load and clean historical BTC-USD price data """
import pandas as pd
import numpy as np
import calendar
import datetime
import math

file_name = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
df_in = pd.read_csv(file_name, header=0)
df_in = df_in.rename({"Volume_(Currency)": "Volume"}, axis=1)
cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
df_in = df_in[cols]

# Convert unix to datetime
df_in.Timestamp = pd.to_datetime(df_in.Timestamp, unit='s')

# Filter to dates of interest
start_date = '2017-01-01'
end_date = '2020-05-31'
df_in = df_in.loc[df_in.Timestamp.between(start_date, end_date)].set_index('Timestamp')

# Re-sample to required frequency
selected_freq = "D"
df = df_in.resample(selected_freq).agg({'Open':'first',
                              'High':'max',
                              'Low':'min',
                              'Close':'last',
                              'Volume': 'sum'})
df = df.reset_index()

# Temporary work around to match naming of other file parts
df = df.rename({"Timestamp": "Date"}, axis=1)


# TEMPORARY WORK AROUND TO GET INTO FORMAT REQUIRED FOR LINEAR_RL_TRADER.PY
df = df[['Close']]

# Output files to csv
df.to_csv('data/btc_ohlc_1d.csv', index=False)

print("Data prepared")



