import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Model Monitoring Dashboard"),
    dcc.Interval(id="interval-component", interval=5*1000, n_intervals=0),
    dcc.Graph(id="live-update-graph")
])


@app.callback(
    Output("live-update-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_graph_live(n):
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [0.80, 0.81, 0.72, 0.75] # test
    }

    df = pd.DataFrame(data)
    fig = px.bar(df, x="Metric", y="Value", title="Model Performance Metrics")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)