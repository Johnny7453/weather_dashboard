import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from pandas.plotting import autocorrelation_plot

# Initialize Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SHOW_ANALYSIS_PLOTS = False

# Load data with the correct delimiter
df = pd.read_csv('../data/temperature.csv', delimiter=',')

# Ensure the Date column is treated as a datetime object
df['Date'] = pd.to_datetime(df['Date'], yearfirst=True, utc=True, format='ISO8601')

# Set the Date column as the index
df.set_index('Date', inplace=True)

# Plot the aurocorrelation_plot for analysis
if(SHOW_ANALYSIS_PLOTS):
    autocorrelation_plot(df["Temperature"])
    plt.show()

# Fit an ARIMA model
model = ARIMA(df['Temperature'], order=(20, 2, 2))
model_fit = model.fit()

# Forecast future values
n_periods = 90  # Number of periods (days) to forecast
forecast_result = model_fit.get_forecast(steps=n_periods)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()


# Create a DataFrame for the forecasted values
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=n_periods, freq='1h')

forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower CI': conf_int.iloc[:, 0],
    'Upper CI': conf_int.iloc[:, 1]
}, index=forecast_dates)

# Combine historical and forecast data
combined_df = pd.concat([df, forecast_df], axis=0)

# Create a time series plot
fig = go.Figure()

# Add Historical Temperature trace
fig.add_trace(go.Scatter(
    # x=combined_df["Date"], y=combined_df['Temperature'],
    x=combined_df.index, y=combined_df['Temperature'],
    name='Historical Temperature',
    line=dict(color='blue')
))

# Add Forecasted Temperature trace
fig.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df['Forecast'],
    name='Forecasted Temperature',
    line=dict(color='red', dash='dash')
))

# Add Confidence Interval
fig.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df['Lower CI'],
    fill=None,
    mode='lines',
    line=dict(color='lightgrey'),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df['Upper CI'],
    fill='tonexty',
    mode='lines',
    line=dict(color='lightgrey'),
    name='Confidence Interval'
))

fig.update_layout(
    title='Temperature Forecast',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Temperature'),
    template='plotly_white'
)

# Define the layout of the Dash app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Temperature Forecast Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='temperature-forecast-graph', figure=fig), className="mb-4")
    ]),
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
