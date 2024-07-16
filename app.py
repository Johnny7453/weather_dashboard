from flask import Flask
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from collections import OrderedDict
import joblib

# Initialize Flask
server = Flask(__name__)

# Initialize Dash
app = Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Example data
df = pd.read_csv('data/temperature.csv', delimiter=',')
prediction_df = pd.read_csv('data/temperature_prediction_data.csv', delimiter=',')

# Checks
# Print the first few rows of the DataFrame
# print(df.head())

# Check the data types of the columns
# print(df.dtypes)

df['Date'] = pd.to_datetime(df['Date'], yearfirst=True, utc=True, format='ISO8601')
prediction_df['Date'] = pd.to_datetime(prediction_df['Date'], yearfirst=True, utc=True, format='ISO8601')

# Create lag features for temperature in the new data
prediction_df['Temp_Lag1'] = prediction_df['Temperature'].shift(1)
prediction_df['Temp_Lag2'] = prediction_df['Temperature'].shift(2)

# Drop rows with NaN values created by lag features
prediction_df = prediction_df.dropna()

# Filter out dates in prediction_df that are already in df
prediction_df = prediction_df[~prediction_df['Date'].isin(df['Date'])]

# Define features for prediction
X_new = prediction_df[['Moisture', 'Rain', 'Temp_Lag1', 'Temp_Lag2']]

# Load the trained model
model = joblib.load('modelling/temperature_model.pkl')

# Make predictions on the new data
prediction_df['Predicted_Temperature'] = model.predict(X_new)


# Combine the original data with the new predictions
combined_df = pd.concat([df, prediction_df], ignore_index=True)


# Create a function for making predictions
def predict_temperature(moisture, rain, temp_lag1, temp_lag2):
    features = pd.DataFrame({
        'Moisture': [moisture],
        'Rain': [rain],
        'Temp_Lag1': [temp_lag1],
        'Temp_Lag2': [temp_lag2]
    })
    prediction = model.predict(features)

    return prediction[0]

# Create a time series plot with two y-axes
fig = go.Figure()

# Add Temperature trace
fig.add_trace(go.Scatter(
    x=combined_df['Date'], y=combined_df['Temperature'],
    name='Temperature',
    yaxis='y1',
    line=dict(color='red')
))

# Add Predicted Temperature trace
fig.add_trace(go.Scatter(
    x=combined_df['Date'], y=combined_df['Predicted_Temperature'],
    name='Predicted Temperature',
    yaxis='y1',
    line=dict(color='orange', dash='dash')
))

# Add Humidity trace with a second y-axis
fig.add_trace(go.Scatter(
    x=combined_df['Date'], y=combined_df['Moisture'],
    name='Humidity',
    yaxis='y2',
    line=dict(color='blue')
))

# Update layout for second y-axis
fig.update_layout(
    title='Temperature and Humidity Over Time',
    xaxis=dict(title='Date', type='date', tickformat='%Y-%m-%d %H:%M'),
    yaxis=dict(
        title='Temperature (°C)',
        titlefont=dict(color='red'),
        tickfont=dict(color='red')
    ),
    yaxis2=dict(
        title='Humidity (%)',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1.2),
    template='plotly_white'
)


# Create a bar graph for Moisture data
bar_fig_moisture = px.bar(combined_df, x='Date', y='Moisture', title='Moisture', text = 'Moisture')
bar_fig_moisture.update_layout(yaxis_title='Moisture (%)', xaxis=dict(title='Date', type='date', tickformat='%Y-%m-%d %H:%M'))

# Create a bar graph for Rain data
bar_fig_rain = px.bar(combined_df, x='Date', y='Rain', title='Amount of Rain', text = 'Rain')
bar_fig_rain.update_layout(yaxis_title='Rain (mm)', xaxis=dict(title='Date and Time', type='date', tickformat='%Y-%m-%d %H:%M'))

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Weather Dashboard"), className="mb-2")
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph', figure=fig), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Predict temperature"),
            html.Label('Moisture:'),
            dcc.Input(id='input-moisture', type='number', value=df.iloc[-1]['Moisture'], min=0, max=100),
            html.Label('Rain:'),
            dcc.Input(id='input-rain', type='number', value=df.iloc[-1]['Rain'], min=0, max=30),
            html.Label('Temp Lag1:'),
            dcc.Input(id='input-temp-lag1', type='number', value=df.iloc[-1]['Temperature'], min=-50, max=60),
            html.Label('Temp Lag2:'),
            dcc.Input(id='input-temp-lag2', type='number', value=df.iloc[-2]['Temperature'], min=-50, max=60),
            html.Button('Predict Temperature', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output')
        ])
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='moisture-bar-graph', figure=bar_fig_moisture), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='rain-bar-graph', figure=bar_fig_rain), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(html.Button('Show/Hide Data', id='toggle-table-button', n_clicks=0), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in combined_df.columns],
            data=combined_df.to_dict('records'),
            editable=True,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',
                'whiteSpace': 'normal'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Temperature', 'filter_query': '{Temperature} >= 25'},
                    'backgroundColor': 'tomato',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Moisture', 'filter_query': '{Moisture} >= 80'},
                    'backgroundColor': 'lightblue',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Rain', 'filter_query': '{Rain} >= 0.8'},
                    'backgroundColor': 'blue',
                    'color': 'white'
                },
                {
                    'if': {
                        'column_type': 'any'  # 'text' | 'any' | 'datetime' | 'numeric'
                    },
                    'textAlign': 'center'
                },
                {
                    'if': {
                        'state': 'active'  # 'active' | 'selected'
                    },
                'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                'border': '1px solid rgb(0, 116, 217)'
                }

            ]
        ),
        id='table-container',
        style={'display': 'none'}  # Initially hide the table
        )
    ])
], className="container")


# Define the callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-moisture', 'value'), State('input-rain', 'value'), 
     State('input-temp-lag1', 'value'), State('input-temp-lag2', 'value')]
)
def update_prediction(n_clicks, moisture, rain, temp_lag1, temp_lag2):
    if n_clicks > 0:
        predicted_temperature = predict_temperature(moisture, rain, temp_lag1, temp_lag2)
        return f'The predicted temperature is {predicted_temperature:.2f}°C'
    return ''

# Define the callback to toggle the table visibility
@app.callback(
    Output('table-container', 'style'),
    [Input('toggle-table-button', 'n_clicks')],
    [State('table-container', 'style')]
)
def toggle_table(n_clicks, current_style):
    if n_clicks % 2 == 1:
        # Show the table
        current_style['display'] = 'block'
    else:
        # Hide the table
        current_style['display'] = 'none'
    return current_style

if __name__ == '__main__':
    app.run_server(debug=True)
