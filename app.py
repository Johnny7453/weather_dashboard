from flask import Flask
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Initialize Flask
server = Flask(__name__)

# Initialize Dash
app = Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Example data
df = pd.read_csv('temperature.csv', delimiter=';')

df['Date'] = pd.to_datetime(df['Date'])


# Melt the dataframe to long format
df_long = pd.melt(df, id_vars=['Date'], value_vars=['Temperature', 'Moisture'], 
                  var_name='Variable', value_name='Value')


# Create a time series plot with both Temperature and Moisture
fig = px.line(df_long, x='Date', y='Value', color='Variable', title='Temperature and Moisture Over Time')


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Weather Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.H2("Temperature"), className="mb-2")
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph', figure=fig), className="mb-4")
    ]),
], className="container")

if __name__ == '__main__':
    app.run_server(debug=True)
