from enum import Enum
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import math
import seaborn as sns


def kelly(price, std):
    kelly_values = []

    for tau in np.arange(0, 1.05, 0.05):
        price = price.squeeze()
        position_in_contracts = tau / (price * std)
        return_price_points = price.diff() * position_in_contracts.shift(1)
        perc_return = return_price_points / 1
        account = (1 + perc_return).cumprod()
        kelly_values.append((tau, account.iloc[-1]))

    kelly_df = pd.DataFrame(kelly_values, columns=['Tau', 'Final_Account_Value'])
    return kelly_df


def risktargeting(price: pd.DataFrame, target_tau: float, multiplier: float, capital: float, instrument_risk: pd.Series, weight: float, IDM: float,  contracts: int = 4):

    
    position_in_contracts= capital * weight * IDM * target_tau / (multiplier* price * instrument_risk)
    minimum_capital = (contracts * multiplier * price * instrument_risk)/ (IDM*weight*target_tau)
    
    
    return position_in_contracts, minimum_capital

def position_buyhold(price: pd.DataFrame, target_tau: float, multiplier: float, capital: float, instrument_risk: pd.Series, weight: float, IDM: float,  contracts: int = 4):

    
    position_in_contracts= capital * weight * IDM * target_tau / (multiplier* price * instrument_risk)
    minimum_capital = (contracts * multiplier * price * instrument_risk)/ (IDM*weight*target_tau)
    
    
    return position_in_contracts, minimum_capital

def variable_ewm_std(price: pd.DataFrame, ewm_window: int):
    perc_return= price.pct_change()
    ann_recent_exp_std = perc_return.ewm(span=ewm_window).std()*(252**0.5)

    ## Weight with ten year vol
    ten_year_vol = ann_recent_exp_std.rolling(252 * 10, min_periods=1).mean() #get the average for ten years
    weighted_vol = 0.3 * ten_year_vol + 0.7 * ann_recent_exp_std
   
    return weighted_vol

def calculate_turnover(position: pd.Series, average_position: pd.Series ):
    daily_trades_to_average = position.diff().abs()/average_position.shift(1)

    #we compare absolute changes to average position and take a mean of those. 
    #thats the average change of positions compared to having average position
    average_daily = daily_trades_to_average.mean()
    (daily_trades_to_average.rolling(window=180).mean()*252).plot(title="Average Turnover over past 6 months rolling")
    annualised_turnover = average_daily * 252
    
    return annualised_turnover

def strategy_buyhold(price: pd.DataFrame, capital: float, position_in_contracts_held: pd.Series):
    return_price_points = (price - price.shift(1)) * position_in_contracts_held.shift(1)
    perc_return = return_price_points / capital

    ann_mean = perc_return.mean() * 252
    ann_std = perc_return.std() * (252 ** 0.5)
    sharpe_ratio = ann_mean / ann_std
    skew_at_freq = perc_return.skew()

    # Drawdowns and skew
    cum_perc_return = (1 + perc_return).cumprod() - 1
    max_cum_perc_return = cum_perc_return.cummax()
    drawdowns = (cum_perc_return - max_cum_perc_return) / (1 + max_cum_perc_return)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.min()
    NORMAL_DISTR_RATIO = norm.ppf(0.01) / norm.ppf(0.3)

    def demeaned_remove_zeros(x):
        x[x == 0] = np.nan
        return x - x.mean()

    def calculate_quant_ratio_lower(x): 
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(0.01) / x_dm.quantile(0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    def calculate_quant_ratio_upper(x): 
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(1 - 0.01) / x_dm.quantile(1 - 0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    # Plotting the cumulative returns



    # Return percentage return and statistics dictionary
    return (
        perc_return,
        dict(
            ann_mean=ann_mean,
            ann_std=ann_std,
            sharpe_ratio=sharpe_ratio,
            skew=skew_at_freq,
            avg_drawdown=avg_drawdown,
            max_drawdown=max_drawdown,
            quant_ratio_lower=calculate_quant_ratio_lower(perc_return),
            quant_ratio_upper=calculate_quant_ratio_upper(perc_return)
        )
    )


def forecast(price:pd.DataFrame, slow_span:float, fast_span:float):
    slow_ewma=price.ewm(span=slow_span,min_periods=2).mean()
    fast_ewma=price.ewm(span=fast_span,min_periods=2).mean()
    trend_filter = fast_ewma - slow_ewma
    raw_forecast = trend_filter / (price*price.pct_change().std())
    forecast_scaler=10/(abs(raw_forecast).mean())
    scaled_forecast=raw_forecast*forecast_scaler
    return forecast_scaler, scaled_forecast

def risktarget_trendfilterlsforecast(price: pd.DataFrame,capped_forecast:pd.Series, target_tau: float, multiplier: float, capital: float, instrument_risk: pd.Series, weight: float, IDM: float, contracts: int = 4):

    position_in_contracts =  capped_forecast * capital * weight * IDM * target_tau / (10 * multiplier* price * instrument_risk)
    minimum_capital = (contracts * multiplier * price * instrument_risk)/ (IDM*weight*target_tau)

    average_position = capital * weight * IDM * target_tau / ( multiplier* price * instrument_risk)
    return position_in_contracts, minimum_capital,average_position

def strategy_results_trendforecast(price: pd.DataFrame, capital: float, position_in_contracts_held: pd.DataFrame):
    return_price_points = (price - price.shift(1)) * position_in_contracts_held.shift(1)
    perc_return = return_price_points / capital

    ann_mean = perc_return.mean() * 252
    ann_std = perc_return.std() * (252 ** 0.5)
    sharpe_ratio = ann_mean / ann_std
    skew_at_freq = perc_return.skew()

    # Drawdowns and skew
    cum_perc_return = (1 + perc_return).cumprod() - 1
    max_cum_perc_return = cum_perc_return.cummax()
    drawdowns = (cum_perc_return - max_cum_perc_return) / (1 + max_cum_perc_return)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.min()
    NORMAL_DISTR_RATIO = norm.ppf(0.01) / norm.ppf(0.3)

    def demeaned_remove_zeros(y):
        x = y.copy()
        x[x == 0] = np.nan
        return x - x.mean()

    def calculate_quant_ratio_lower(x): 
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(0.01) / x_dm.quantile(0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    def calculate_quant_ratio_upper(x): 
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(1 - 0.01) / x_dm.quantile(1 - 0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    return (
        perc_return, 
        dict(
            ann_mean=ann_mean,
            ann_std=ann_std,
            sharpe_ratio=sharpe_ratio,
            skew=skew_at_freq,
            avg_drawdown=avg_drawdown,
            max_drawdown=max_drawdown,
            quant_ratio_lower=calculate_quant_ratio_lower(perc_return),
            quant_ratio_upper=calculate_quant_ratio_upper(perc_return)
        )
    )

def apply_buffering(position: pd.DataFrame, buffer_value: float, average_position: pd.DataFrame):
    buffer = abs(average_position) * buffer_value
    upper_buffer = (position + buffer).round()
    lower_buffer = (position - buffer).round()
    
    current_position = 1
    buffered_position_list = []
    
    for idx in range(len(position.index)):
        if current_position > upper_buffer.iloc[idx, 0]:
            buffered_position_list.append(upper_buffer.iloc[idx, 0])
        elif current_position < lower_buffer.iloc[idx, 0]:
            buffered_position_list.append(lower_buffer.iloc[idx, 0])
        else:
            buffered_position_list.append(current_position)
        
        current_position = buffered_position_list[-1]
    
    # Preserve original column name
    original_col_name = position.columns[0]
    
    # Create DataFrame with the same column name
    buffered_position = pd.DataFrame(buffered_position_list, index=position.index, columns=[original_col_name])
    
    return buffered_position


def jumbo_stats(merged_returns):
    ann_mean = merged_returns.mean() * 252
    ann_std = merged_returns.std() * (252 ** 0.5)
    sharpe_ratio = ann_mean / ann_std
    skew_at_freq = merged_returns.skew()

    # Drawdowns and skew
    cum_perc_return = (1 + merged_returns).cumprod() - 1
    max_cum_perc_return = cum_perc_return.cummax()
    drawdowns = (cum_perc_return - max_cum_perc_return) / (1 + max_cum_perc_return)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.min()
    NORMAL_DISTR_RATIO = norm.ppf(0.01) / norm.ppf(0.3)

    def demeaned_remove_zeros(y):
        x = y.copy()
        x[x == 0] = np.nan
        return x - x.mean()

    def calculate_quant_ratio_lower(x):
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(0.01) / x_dm.quantile(0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    def calculate_quant_ratio_upper(x):
        x_dm = demeaned_remove_zeros(x)
        raw_ratio = x_dm.quantile(1 - 0.01) / x_dm.quantile(1 - 0.3)
        return raw_ratio / NORMAL_DISTR_RATIO

    return (
        dict(
            ann_mean=ann_mean,
            ann_std=ann_std,
            sharpe_ratio=sharpe_ratio,
            skew=skew_at_freq,
            avg_drawdown=avg_drawdown,
            max_drawdown=max_drawdown,
            quant_ratio_lower=calculate_quant_ratio_lower(merged_returns),
            quant_ratio_upper=calculate_quant_ratio_upper(merged_returns)
        )
    )