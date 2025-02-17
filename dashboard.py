import streamlit as st
import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
# Streamlit Layout
st.title("Trading Strategy Dashboard")

# Upload Data File
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_name = st.selectbox("Select a sheet", xls.sheet_names)
    price = pd.read_excel(xls, sheet_name=sheet_name, header=0, parse_dates=[0], index_col=0)
else:
    st.stop()


# Plot Price
st.subheader("Price Chart")
fig = px.line(price, title="Price Chart")
st.plotly_chart(fig, use_container_width=True,key="price_chart")

# Calculate Standard Deviation
daily_std = price.pct_change().std() * (252 ** 0.5)
st.write(f"**Daily Standard Deviation:** {daily_std}")


# Kelly Criterion Graph
st.subheader("Kelly Criterion Visualization")
kelly_df = functions.kelly(price=price, std=daily_std)
# Scatter Plot for Kelly Criterion
fig = px.scatter(kelly_df, x='Tau', y='Final_Account_Value', title="Tau vs Final Account Value")
st.plotly_chart(fig, use_container_width=True)



# EWM STD Estimate
st.subheader("Exponential Weighted Moving STD")
ewm_window = st.slider("Select EWM Window", min_value=10, max_value=100, value=32, step=1)
instrument_risk_ewm = functions.variable_ewm_std(price=price, ewm_window=ewm_window)
st.line_chart(instrument_risk_ewm)

# Plot Recent Vol Estimate
st.subheader("Recent Vol Estimate (40 days)")
st.line_chart(instrument_risk_ewm.tail(50))


# Buy and Hold Strategy
st.subheader("Buy and Hold Strategy")
position_hold, minimum_capital_req_hold = functions.position_buyhold(
    price=price, target_tau=st.slider("Select target tau", min_value=0.1, max_value=0.5, value=0.2, step=0.01),
    multiplier=st.number_input("Select Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                               key=f'multiplier_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
    capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000,
                            key=f'capital_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
    instrument_risk=instrument_risk_ewm, weight=1, IDM=1
)
st.markdown("Position for Buy and Hold")
st.line_chart(position_hold)

perc_return, stats = functions.strategy_buyhold(
    price=price,
    capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000,
                            key=f'capital_buyhold_{np.random.randint(0, 10000)}'),
    position_in_contracts_held=position_hold,
)
buyandhold = (1 + perc_return).cumprod()
st.markdown("Buy and Hold Backtest")
st.line_chart(buyandhold)

st.write("**Buy and Hold Statistics:**")
st.write(stats)


# Trend Following Strategies
st.subheader("Trend Following Strategies")
filter_options = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
filter_pairs = st.multiselect("Select Trend Filters", filter_options, default=filter_options)
returns_df = pd.DataFrame()

for fast_span, slow_span in filter_pairs:
    st.write(f"**Processing filter pair: Fast={fast_span}, Slow={slow_span}**")
    forecast_scaler, scaled_forecast = functions.forecast(price=price, slow_span=slow_span, fast_span=fast_span)
    capped_forecast = scaled_forecast.clip(-20, 20)

    # Plot Forecast
    st.markdown(f"Capped Forecast for (Fast={fast_span}, Slow={slow_span})")
    st.line_chart(capped_forecast)

    # Position Calculation
    position, _, _ = functions.risktarget_trendfilterlsforecast(
        price=price, capped_forecast=capped_forecast,
        target_tau=st.slider("Select Target Tau", min_value=0.1, max_value=0.5, value=0.2, step=0.01,
                             key=f'target_tau_{np.random.randint(0, 10000)}'),
        multiplier=st.number_input("Select Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                   key=f'multiplier_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
        capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000,
                                key=f'capital_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
        instrument_risk=instrument_risk_ewm, weight=1, IDM=1
    )
    st.markdown(f"Position (Fast={fast_span}, Slow={slow_span})")
    st.line_chart(position)


    # Calculate the strategy results (this function already plots cumulative returns)
    perc_return, stats = functions.strategy_results_trendforecast(
        price=price,
        capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000,
                                key=f'capital_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
        position_in_contracts_held=position
    )

    # Capture the existing Matplotlib figure and display it in Streamlit
    result_cumulated = (1 + perc_return).cumprod()
    st.markdown(f"Portfolio with Filter (Fast={fast_span}, Slow={slow_span}")
    st.line_chart(result_cumulated)

    st.write("**Statistics:**")
    st.write(stats)

    returns_df[f"Fast={fast_span}, Slow={slow_span}"] = perc_return




# Correlation Matrix
st.subheader("Correlation Matrix of Strategy Returns")
correlation_matrix = returns_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
st.pyplot(fig)

# Combined Forecast
st.subheader("Combined Weighted Forecast")
combined_filter_options = [(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)]
selected_filter_pairs = st.multiselect("Select Combined Forecast Filters", combined_filter_options,
                                       default=[(2, 8), (4, 16)])
capped_forecasts_dict = {}

for fast_span, slow_span in selected_filter_pairs:
    forecast_scaler, scaled_forecast = functions.forecast(price=price, slow_span=slow_span, fast_span=fast_span)
    capped_forecast = scaled_forecast.clip(-20, 20)
    capped_forecasts_dict[f"Fast={fast_span}_Slow={slow_span}"] = capped_forecast

n = len(capped_forecasts_dict)
weight = 1 / n
weighted_forecasts = [capped_forecast * weight for capped_forecast in capped_forecasts_dict.values()]
combined_forecast = sum(weighted_forecasts)
combined_forecast = ((10 / abs(combined_forecast).mean()) * combined_forecast).clip(-20, 20) #bring average to 10 and clip it

st.markdown(f"**Capped Forecast for {selected_filter_pairs}")
st.line_chart(combined_forecast)

# Calculate and display the absolute mean of the combined forecast
abs_mean_combined_forecast = abs(combined_forecast).mean()
st.write(f"**Absolute mean of the combined forecast:** {abs_mean_combined_forecast}")

# Final Strategy
st.subheader("Final Strategy Analysis")
position_overall, _, average_position_overall = functions.risktarget_trendfilterlsforecast(
    price=price, capped_forecast=combined_forecast,
    target_tau=st.slider("Select Target Tau", min_value=0.1, max_value=0.5, value=0.2, step=0.01,
                         key=f'target_tau_{np.random.randint(0, 10000)}'),
    multiplier=st.number_input("Select Multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                               key=f'multiplier_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
    capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000,
                            key=f'capital_{st.session_state.get("unique_id", 0)}_{np.random.randint(0, 10000)}'),
    instrument_risk=instrument_risk_ewm, weight=1, IDM=1
)
position_overall_buffered = functions.apply_buffering(position=position_overall, buffer_value=st.slider("Select buffer value", min_value=0.0, max_value=1.0, value=0.3, step=0.05), average_position=average_position_overall)
st.subheader("Buffered Position Over Time")
fig, ax = plt.subplots()
ax.plot(position_overall_buffered, label="Buffered Position")
ax.set_title("Buffered Position Over Time")
ax.legend()
st.pyplot(fig)

st.line_chart(position_overall_buffered)

# Compare Normal and Buffered Positions with Streamlit Line Chart
comparison_df = pd.DataFrame({
    'Normal': position_overall.tail(40).squeeze(),
    'Buffered': position_overall_buffered.tail(40).squeeze()
}, index=position_overall.tail(40).index)

st.line_chart(comparison_df)

# Calculate the final strategy results
perc_return, stats = functions.strategy_results_trendforecast(
    price=price,
    capital=st.number_input("Select Capital", min_value=1000, max_value=1000000, value=100000, step=1000, key=f'capital_final_{np.random.randint(0, 10000)}'),
    position_in_contracts_held=position_overall_buffered
)
# Capture the existing Matplotlib figure and display it in Streamlit
result_combined_cumulated = (1 + perc_return).cumprod()
st.markdown(f"Portfolio with {selected_filter_pairs}")
st.line_chart(result_combined_cumulated)

st.write("**Final Strategy Statistics:**")
st.write(stats)
