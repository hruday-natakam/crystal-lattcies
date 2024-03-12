import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)

# Define lattice parameters
a = 1
b = 1
theta = np.pi / 3  # 60 degrees

# Define lattice vectors
v1 = [a, 0, 0]
v2 = [a * np.cos(theta), a * np.sin(theta), 0]

# Create grid points
x_values = np.arange(-5, 6, 1)
y_values = np.arange(-5, 6, 1)
grid_points = np.array([[i * v1[0] + j * v2[0], i * v1[1] + j * v2[1], 0] for i in x_values for j in y_values])

# Create app layout
app.layout = html.Div([
    html.Div([
        html.H2("Periodic Lattice Visualization"),
        dcc.Graph(id='lattice-plot',
                  config={'editable': True, 'edits': {'shapePosition': True}}),  # Enable shape editing
        html.Div(id='hover-info')
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='selected-point-plot')
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        html.P("Select lattice shape:"),
        dcc.Dropdown(
            id='shape-dropdown',
            options=[
                {'label': 'Square', 'value': 'square'},
                {'label': 'Hexagon', 'value': 'hexagon'}
            ],
            value='square'
        )
    ], style={'width': '50%', 'margin': 'auto', 'text-align': 'center', 'margin-top': '20px'}),
    html.Div([
        html.Button('Reset Zoom', id='reset-zoom-btn', n_clicks=0)
    ], style={'width': '50%', 'margin': 'auto', 'text-align': 'center', 'margin-top': '20px'})
])

# Callback to update lattice plot, hover info, and selected point plot
@app.callback(
    [Output('lattice-plot', 'figure'),
     Output('hover-info', 'children'),
     Output('selected-point-plot', 'figure')],
    [Input('lattice-plot', 'hoverData'),
     Input('shape-dropdown', 'value'),
     Input('reset-zoom-btn', 'n_clicks')],
    [State('lattice-plot', 'relayoutData')]
)
def update_plot(hover_data, shape, reset_zoom_btn_clicks, relayout_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        prop_id = None
    else:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if prop_id == 'reset-zoom-btn':
        relayout_data = None

    if hover_data:
        hover_point = hover_data['points'][0]
        hover_x = hover_point['x']
        hover_y = hover_point['y']
        hover_info = f"Selected Point: ({hover_x}, {hover_y})"
    else:
        hover_x = None
        hover_y = None
        hover_info = "Hover over a point to see coordinates."

    # Create lattice plot based on selected shape
    if shape == 'square':
        lattice_points = grid_points
    elif shape == 'hexagon':
        lattice_points = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * v1[0] + j * v2[0]
                y = i * v1[1] + j * v2[1]
                if (-3 <= x <= 3) and (-3 <= y <= 3):  # Restrict to hexagon shape
                    lattice_points.append([x, y, 0])
        lattice_points = np.array(lattice_points)
    else:
        lattice_points = grid_points

    # Create lattice plot
    lattice_trace = go.Scatter3d(x=lattice_points[:, 0], y=lattice_points[:, 1], z=lattice_points[:, 2], mode='markers',
                                 marker=dict(size=8, color='rgba(255, 0, 0, 0.8)'), hoverinfo='text',
                                 text=[f'Point: ({x:.2f}, {y:.2f}, {z:.2f})' for x, y, z in lattice_points])

    lattice_fig = go.Figure(data=[lattice_trace], layout=go.Layout(title='Periodic Lattice',
                                                                    scene=dict(xaxis=dict(title='X'),
                                                                               yaxis=dict(title='Y'),
                                                                               zaxis=dict(title='Z')),
                                                                    hovermode='closest',
                                                                    uirevision='True'))
    if relayout_data:
        if 'autosize' in relayout_data:
            del relayout_data['autosize']
        lattice_fig.update_layout(relayout_data)

    # Create selected point plot with vectors
    selected_point_trace = go.Scatter3d(x=[hover_x] if hover_x else [], y=[hover_y] if hover_y else [], z=[0],
                                        mode='markers', marker=dict(size=10, color='rgba(0, 0, 255, 0.8)'),
                                        name='Selected Point', hoverinfo='text',
                                        text=[f'Selected Point: ({hover_x}, {hover_y})'] if hover_x else [])

    # Calculate vectors v1, v2, and v0
    if hover_x is not None and hover_y is not None:
        v1_vector = np.array([hover_x, hover_y, 0]) + np.array(v1)
        v2_vector = np.array([hover_x, hover_y, 0]) + np.array(v2)
        v0_vector = np.array([hover_x, hover_y, 0]) + np.array(v2) - np.array(v1)

        # Add vectors as traces
        vectors_trace = [go.Scatter3d(x=[hover_x, v1_vector[0]], y=[hover_y, v1_vector[1]], z=[0, v1_vector[2]],
                                      mode='lines', line=dict(color='red'), name='v1'),
                         go.Scatter3d(x=[hover_x, v2_vector[0]], y=[hover_y, v2_vector[1]], z=[0, v2_vector[2]],
                                      mode='lines', line=dict(color='green'), name='v2'),
                         go.Scatter3d(x=[hover_x, v0_vector[0]], y=[hover_y, v0_vector[1]], z=[0, v0_vector[2]],
                                      mode='lines', line=dict(color='blue'), name='v0')]

        selected_point_trace['text'].append(f'v1: ({v1[0]}, {v1[1]}, {v1[2]})')
        selected_point_trace['text'].append(f'v2: ({v2[0]}, {v2[1]}, {v2[2]})')
        selected_point_trace['text'].append(f'v0: ({v0_vector[0]}, {v0_vector[1]}, {v0_vector[2]})')

        selected_point_fig = go.Figure(data=[selected_point_trace] + vectors_trace, layout=go.Layout(title='Selected Point and Vectors',
                                                                                                     scene=dict(xaxis=dict(title='X'),
                                                                                                                yaxis=dict(title='Y'),
                                                                                                                zaxis=dict(title='Z'))))
    else:
        selected_point_fig = go.Figure(data=[selected_point_trace], layout=go.Layout(title='Selected Point',
                                                                                     scene=dict(xaxis=dict(title='X'),
                                                                                                yaxis=dict(title='Y'),
                                                                                                zaxis=dict(title='Z'))))

    return lattice_fig, hover_info, selected_point_fig


if __name__ == '__main__':
    app.run_server(debug=True)
