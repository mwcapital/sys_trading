import functions
from enum import Enum
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import math
import seaborn as sns



price = pd.read_excel('data.xlsx',  header=0, parse_dates=[0], index_col=0,sheet_name='COPX')

price.index.name = None
if isinstance(price, pd.DataFrame):
    print("This is a DataFrame")
elif isinstance(price, pd.Series):
    print("This is a Series")

#plot price
plt.figure()
plt.plot(price)
plt.title("price")
plt.show()
#calculate vanilla std 
daily_std=price.pct_change().std()*(252**0.5)
print(f"Daily standard deviation: {daily_std}")
#plot kelly criterion which is roughly half the Sharpe ratio in the center
functions.kellygraph(price=price,recent_std=daily_std)

#ewm std estimate 

# EWM STD Estimate
instrument_risk_ewm = functions.variable_ewm_std(price=price, ewm_window=32)

plt.figure()
plt.plot(instrument_risk_ewm.tail(50))
plt.title("Recent Vol Estimate (40 days)")
plt.show()

plt.figure()
plt.plot(instrument_risk_ewm)
plt.title("Overall Weighted Vol")
plt.show()

######################## Buy and Hold Strategy#############################
position_hold, minimum_capital_req_hold = functions.position_buyhold(
    price=price, target_tau=0.22, multiplier=1, capital=100000, instrument_risk=instrument_risk_ewm, weight=1, IDM=1
)
position_hold.plot(title="Position Over Time with EWM Vol (Buy and Hold)")

perc_return, stats=functions.strategy_buyhold(
    price=price, multiplier=1, capital=100000, position_in_contracts_held=position_hold, title="Buy and Hold Backtest"
)
print("Statistics for Buy and Hold")
print(stats)



##################trend following with diff filters###############################

print("\n" + "="*50 + "\n")

# List of filter pairs (slow_span, fast_span)
filter_pairs = [
    (2, 8),
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256)
]
returns_df =  pd.DataFrame()
# Loop through each filter pair
for fast_span, slow_span in filter_pairs:
    print(f"Running for filter pair: Fast Span = {fast_span}, Slow Span = {slow_span}")
    
    # Forecast
    forecast_scaler, scaled_forecast = functions.forecast(price=price, slow_span=slow_span, fast_span=fast_span)
    capped_forecast = scaled_forecast.clip(-20, 20)
    
    # Print average of the capped forecast
    print(f"Average of the capped Forecast: {abs(capped_forecast).mean()}")
    
    # Plot capped forecast
    capped_forecast.plot(title=f"Scaled Forecast (Fast={fast_span}, Slow={slow_span})")
    plt.show()
    
    # Position calculation
    position, minimum_capital, average_position = functions.risktarget_trendfilterlsforecast(
        price=price,
        capped_forecast=capped_forecast,
        target_tau=0.2,
        multiplier=1,
        capital=100000,
        instrument_risk=instrument_risk_ewm,
        weight=1,
        IDM=1
    )
    
    # Plot position
    position.plot(title=f"Position with Forecast (Fast={fast_span}, Slow={slow_span})")
    plt.show()
    
    # Strategy results
    perc_return, stats= functions.strategy_results_trendforecast(
        price=price,
        multiplier=1,
        capital=100000,
        position_in_contracts_held=position,
        title=f"Portfolio with Filter (Fast={fast_span}, Slow={slow_span})"
    )
    # Ensure perc_return has the correct index and add to DataFrame
    returns_df[f"Fast={fast_span}, Slow={slow_span}"] = perc_return
    print(f"Statistics for filter pair (Fast={fast_span}, Slow={slow_span}):")
    print(stats)
    print("\n" + "="*50 + "\n")

# Compute correlation matrix
correlation_matrix = returns_df.corr()
# Plot heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Strategy Returns")
plt.show()

######################now run only for the selected forecasts ######################

import functions
from enum import Enum
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import math
import seaborn as sns


# List of selected filter pairs (manually chosen after observing results)
selected_filter_pairs = [
    (10,100),
     # Add/remove pairs as needed
]

# Dictionary to store capped forecast values for selected pairs
capped_forecasts_dict = {}

# Loop through each selected filter pair
for fast_span, slow_span in selected_filter_pairs:
    print(f"Processing filter pair: Fast Span = {fast_span}, Slow Span = {slow_span}")
    
    # Forecast calculation
    forecast_scaler, scaled_forecast = functions.forecast(price=price, slow_span=slow_span, fast_span=fast_span)
    capped_forecast = scaled_forecast.clip(-20, 20)
    
    # Store the capped forecast in the dictionary
    capped_forecasts_dict[f"Fast={fast_span}_Slow={slow_span}"] = capped_forecast
    
    # Print average of the capped forecast
    print(f"Average of the capped Forecast: {abs(capped_forecast).mean()}")
    
    # Plot the capped forecast
    capped_forecast.plot(title=f"Scaled Forecast (Fast={fast_span}, Slow={slow_span})")
    plt.show()

# Get the number of selected forecasts (n)
n = len(capped_forecasts_dict)

# Check if n is greater than 0 to avoid division by zero

# Calculate weights (1/n for each forecast)
weight = 1 / n
    
# Initialize a list to store the weighted forecasts
weighted_forecasts = []
    
# Loop through each capped forecast in the dictionary and apply the weight
for key, capped_forecast in capped_forecasts_dict.items():
    weighted_forecast = capped_forecast * weight
    weighted_forecasts.append(weighted_forecast)
    
# Sum up all weighted forecasts to get the combined forecast
combined_forecast = sum(weighted_forecasts)
    
# Apply clipping to the combined forecast
combined_forecast = (10/abs(combined_forecast).mean())*combined_forecast.clip(-20, 20)
    
# Plot the combined forecast
combined_forecast.plot(title="Combined Weighted Forecast (Clipped)")
plt.show()
    
# Calculate and print the absolute mean of the combined forecast
abs_mean_combined_forecast = abs(combined_forecast).mean()
print(f"Absolute mean of the combined forecast: {abs_mean_combined_forecast}")



#decide on the FDM???? for combined_forecast


#overall result for the instrument##############################
# Position calculation
position_overall, minimum_capital_overall, average_position_overall = functions.risktarget_trendfilterlsforecast(
    price=price,
    capped_forecast=combined_forecast,
    target_tau=0.2,
    multiplier=1,
    capital=100000,
    instrument_risk=instrument_risk_ewm,
    weight=1,
    IDM=1
    )
position_overall_buffered=functions.apply_buffering(position=position_overall,buffer_value=0.3,average_position=average_position_overall)
# Plot position
position_overall.plot(title=f"Position with Combined Forecast")
plt.show()


#compare normal and buffered positions
plt.figure()
plt.plot(position_overall.tail(40))
plt.plot(position_overall_buffered.tail(40))
plt.legend()
plt.show()    

# Strategy results
perc_return, stats= functions.strategy_results_trendforecast(
    price=price,
    multiplier=1,
    capital=100000,
    position_in_contracts_held=position_overall_buffered,
    title=("Portfolio with Multiple Combined Filters)")
    )
    
# Print statistics
print("Statistics for combined Filter")
print(stats)
print("\n" + "="*50 + "\n")
