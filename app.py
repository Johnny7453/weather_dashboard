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

# Checks
# Print the first few rows of the DataFrame
print(df.head())

# Check the data types of the columns
print(df.dtypes)

df['Date'] = pd.to_datetime(df['Date'], yearfirst=True, utc=True, format='ISO8601')

# Load the trained model
# model = joblib.load('modelling/temperature_model.pkl')

# Create a function for making predictions
def predict_temperature(moisture, rain, temp_lag1, temp_lag2):
    prediction = model.predict([[moisture, rain, temp_lag1, temp_lag2]])
    return prediction[0]

# Create a time series plot with two y-axes
fig = go.Figure()

# Add Temperature trace
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Temperature'],
    name='Temperature',
    yaxis='y1',
    line=dict(color='red')
))

# Add Humidity trace with a second y-axis
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Moisture'],
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
bar_fig_moisture = px.bar(df, x='Date', y='Moisture', title='Moisture', text = 'Moisture')
bar_fig_moisture.update_layout(yaxis_title='Moisture (%)', xaxis=dict(title='Date', type='date', tickformat='%Y-%m-%d %H:%M'))

# Create a bar graph for Rain data
bar_fig_rain = px.bar(df, x='Date', y='Rain', title='Amount of Rain', text = 'Rain')
bar_fig_rain.update_layout(yaxis_title='Rain (mm)', xaxis=dict(title='Date and Time', type='date', tickformat='%Y-%m-%d %H:%M'))

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Weather Dashboard"), className="mb-2")
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph', figure=fig), className="mb-4")
    ]),
    # dbc.Row([
    #     dbc.Col([
    #         html.H4("Predict temperature"),
    #         html.Label('Moisture:'),
    #         dcc.Input(id='input-moisture', type='number', value=30, min=0, max=100),
    #         html.Label('Rain:'),
    #         dcc.Input(id='input-rain', type='number', value=0, min=0, max=30),
    #         html.Label('Temp Lag1:'),
    #         dcc.Input(id='input-temp-lag1', type='number', value=20, min=-50, max=60),
    #         html.Label('Temp Lag2:'),
    #         dcc.Input(id='input-temp-lag2', type='number', value=19),
    #         html.Button('Predict Temperature', id='predict-button', n_clicks=0),
    #         html.Div(id='prediction-output')
    #     ])
    # ]),
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
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
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
# @app.callback(
#     Output('prediction-output', 'children'),
#     [Input('predict-button', 'n_clicks')],
#     [State('input-moisture', 'value'), State('input-rain', 'value'), 
#      State('input-temp-lag1', 'value'), State('input-temp-lag2', 'value')]
# )
# def update_prediction(n_clicks, moisture, rain, temp_lag1, temp_lag2):
#     if n_clicks > 0:
#         predicted_temperature = predict_temperature(moisture, rain, temp_lag1, temp_lag2)
#         return f'The predicted temperature is {predicted_temperature:.2f}°C'
#     return ''

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
