from flask import Flask
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# Initialize Flask
server = Flask(__name__)

# Initialize Dash
app = Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Example data
df = pd.read_csv('temperature.csv', delimiter=';')

df['Date'] = pd.to_datetime(df['Date'])

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
    xaxis=dict(title='Date'),
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


# Create a bar graph for Rain data
bar_fig = px.bar(df, x='Date', y='Rain', title='Amount of Rain', text = 'Rain')
bar_fig.update_layout(yaxis_title='Rain (mm)')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Weather Dashboard"), className="mb-2")
    ]),
     dbc.Row([
        dbc.Col(dcc.Graph(id='time-series-graph', figure=fig), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='rain-bar-graph', figure=bar_fig), className="mb-4")
    ]),
], className="container")

if __name__ == '__main__':
    app.run_server(debug=True)
