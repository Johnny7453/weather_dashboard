from flask import Flask
from dash import Dash, html, dcc
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

# Create a sample plot
#fig = px.bar(df, x='Date', y='Temperature')

# Create a time series plot
fig = px.line(df, x='Date', y='Temperature', title='Temperature Over Time')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Weather Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.H2("Temperature"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig), className="mb-4")
    ]),
], className="container")

if __name__ == '__main__':
    app.run_server(debug=True)
