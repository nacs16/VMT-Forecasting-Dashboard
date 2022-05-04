import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.api import VAR

def anomaly_detect_V1(vmt_series, dq_check_threshold=4.5, printing=True):
    """
    Detect anomalies in a daily time series with weekly seasonality.
    For each day, calculate mean and std of all other same-weekday values.
    Then check the value of each dayagainst the average of other same-weekdays
    ignoring out-of-pattern periods that we know are accurate (COVID lockdown dip)
    If the value is 4.5 (configurable) standard deviations outside the same-weekday mean, consider it an anomaly.
    After that first round of checks, recheck the days surrounding any detected anomalies with a smaller threshold.
    
    Parameters:
    vmt_series - a pandas series with datetime index freq=D
    dq_check_threshold - number of standard deviations to consider an anomaly
    printing - boolean, whether to print details about detected anomalies
    
    Returns:
    A list of dates which were detected as anomalies
    """
    unique_weekdays_in_series = vmt_series.index.strftime("%A").to_series().unique()
    weekday_means, weekday_stds = {}, {}
    
    IGNORE_DATES = []
    
    # COVID Lockdown Period
    for ignore_date in pd.date_range(start="2020-03-12", end="2020-06-20"):
        IGNORE_DATES.append(ignore_date)
    ignore_dates_filter = (~vmt_series.index.isin(IGNORE_DATES))
    quantile_threshold_filter = ((vmt_series > vmt_series.quantile(0.05)) & (vmt_series < vmt_series.quantile(0.95)))
    anomalies = []
    
    for weekday in unique_weekdays_in_series:
        weekday_means[weekday] = {}
        weekday_stds[weekday] = {}
        for weekday_date in vmt_series[(vmt_series.index.strftime("%A") == weekday)].index:

            window = datetime.timedelta(days=7*4) # Changed to 4 weeks, was 8 weeks originally
            window_filter = ((vmt_series.index >= weekday_date-window) & (vmt_series.index <= weekday_date+window))
            
            weekday_means[weekday][weekday_date] = vmt_series[ignore_dates_filter & quantile_threshold_filter & window_filter & (vmt_series.index.strftime("%A") == weekday) & (vmt_series.index != weekday_date)].mean()
            weekday_stds[weekday][weekday_date] = vmt_series[ignore_dates_filter & quantile_threshold_filter & window_filter & (vmt_series.index.strftime("%A") == weekday) & (vmt_series.index != weekday_date)].std()    
            
    # Round 1
    for day in vmt_series.index:
        if day in IGNORE_DATES:
            continue
        dq_mean = weekday_means[day.strftime("%A")][day]
        dq_std = weekday_stds[day.strftime("%A")][day]

        dq_check = abs(vmt_series[day] - dq_mean) / dq_std

        if (dq_check > dq_check_threshold):
            if printing:
                print("Anomaly on {}! {} std out of pattern".format(day.strftime("%Y-%m-%d"), round(dq_check,1)))
            anomalies.append(day)
            
    # Round 2, only on days surrounding anomalies detected during round 1 and the threshold is half as big
    # The purpose of round 2 checks is to account for mid-day outage periods that are only a little out of pattern
    for day in [d for d in [d-datetime.timedelta(days=1) for d in anomalies] + [d+datetime.timedelta(days=1) for d in anomalies] if d not in anomalies]:
        if day not in vmt_series.index:
            continue
        dq_mean = weekday_means[day.strftime("%A")][day]
        dq_std = weekday_stds[day.strftime("%A")][day]

        dq_check = abs(vmt_series[day] - dq_mean) / dq_std

        if (dq_check > dq_check_threshold/2):
            if printing:
                print("Round 2 anomaly on {}! {} std out of pattern".format(day.strftime("%Y-%m-%d"), round(dq_check,1)))
            anomalies.append(day)
    return list(pd.Series(anomalies).sort_values()), weekday_means, weekday_stds

def fix_vmt_anomalies(vmt_series, bad_dates):
    
    fix_this_series = vmt_series.copy() # Make a copy of the pandas series that needs interpolation
    
    for bad_date in sorted(bad_dates):
        fix_this_series.loc[bad_date] = np.nan # Set each bad day to np.nan so that the interpolate function works on it
    for bad_date in sorted(bad_dates):
        imputed_val = fix_this_series[fix_this_series.index.strftime("%A") == bad_date.strftime("%A")].interpolate(method='linear').loc[bad_date]
        fix_this_series.loc[bad_date] = imputed_val
    return fix_this_series

def clean_series(series, impute_missing_days=True, fix_anomalies=True):
    """Assumed to be a daily series with datetime index freq='1D'"""
    clean_this = series.copy()

    # Step 1: Check for missing dates
    if impute_missing_days:
        missing_days = []
        for d in pd.date_range(start=series.index.min(), end=series.index.max()):
            if d not in clean_this.index:
                missing_days.append(d)
        if len(missing_days) > 0:
            clean_this = clean_this.append(pd.Series([np.nan]*len(missing_days), index=missing_days)).sort_index().interpolate()
    # Step 2: Ensure there are no duplicate days by resampling to 1D sums
    clean_this = clean_this.resample('1D').sum()
    # Step 3: Detect and replace anomalies
    if fix_anomalies:
        bad_dates, _, _ = anomaly_detect_V1(clean_this, printing=False)
        clean_this = fix_vmt_anomalies(clean_this, bad_dates)
    return clean_this

def standard_rescale(series):
    mu = series.mean()
    s = series.std()
    return series.apply(lambda x: (x-mu) / s)

def invert_diff(df_original, df_diff, periods=1):
    df_undiff = df_diff.copy()
    for c in df_diff.columns:        
        df_undiff[c] = df_original[c].shift(periods) + df_diff[c]
    return df_undiff

def score_forecast(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
    mae = np.mean(np.abs(forecast - actual))
    rmse = np.mean((forecast - actual)**2)**.5
    return {'mape':mape, 'mae':mae, 'rmse':rmse}

def run_VAR_forecast(df, target_metric):
    """Returns a 30-Day Forecast dataframe"""

    df.to_pickle("df_passed_to_run_VAR_forecast.pkl")

    df_diff = df.diff(7)
    nobs=30
    df_diff_train, df_diff_test = df_diff.iloc[0:-nobs].copy(), df_diff.iloc[-30:].copy()

    model = VAR(df_diff_train, freq='d')
    lag_order = model.select_order().selected_orders['aic'] # ['aic', 'bic', 'hqic', 'fpe']
    trend_parameter = 'c'
    model_fit = model.fit(lag_order, trend=trend_parameter)
    forecast = pd.DataFrame(model_fit.forecast(y=df_diff_train[-lag_order:].values, steps=nobs), index=df_diff_test.index, columns=df_diff_test.columns)
    forecast_undiff = invert_diff(df, forecast, periods=7)[target_metric]
    return forecast_undiff, score_forecast(forecast_undiff, df[-nobs:][target_metric])


