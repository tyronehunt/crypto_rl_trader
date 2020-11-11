import pandas as pd
import numpy as np
import calendar
import datetime
import math
import talib


def convert_to_ohlc(file_in= 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv',
                    start='2017-01-01', end='2020-05-31', freq="D",
                    output_path='data/btc_ohlc_1d.csv'):
    """ 1.0 Load and clean historical BTC-USD price data """
    df_in = pd.read_csv(file_in, header=0)
    df_in = df_in.rename({"Volume_(Currency)": "Volume"}, axis=1)
    cols = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    df_in = df_in[cols]

    # Convert unix to datetime
    df_in.Timestamp = pd.to_datetime(df_in.Timestamp, unit='s')

    # Filter to dates of interest
    start_date = start
    end_date = end
    df_in = df_in.loc[df_in.Timestamp.between(start_date, end_date)].set_index('Timestamp')

    # Re-sample to required frequency
    selected_freq = freq
    df = df_in.resample(selected_freq).agg({'Open':'first',
                                  'High':'max',
                                  'Low':'min',
                                  'Close':'last',
                                  'Volume': 'sum'})
    df = df.reset_index()

    # Match naming of other file parts
    df = df.rename({"Timestamp": "Date"}, axis=1)

    # Output files to csv
    df.to_csv(output_path, index=False)
    print("Data prepared")


def get_data(data_filepath):
    # returns a T x 3 list of (daily close) close prices
    df = pd.read_csv(data_filepath)

    # TEMPORARY WORK AROUND TO GET INTO FORMAT REQUIRED FOR LINEAR_RL_TRADER.PY
    df = df[['Close', 'Volume']]
    df = add_indicators(df)

    df = df.dropna(subset=['MACD'])
    return df.values


def add_indicators(df):
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df


