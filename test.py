import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

# Data for plots - coordinates for x and y
x = np.array([1, 0.5, 0, 0, 1])
y = np.array([0, 0.5, 1, 0, 0])
z = np.array([0, 0, 0, 0, 0])  # z-coordinates for the initial data

# Create Dash App
app = dash.Dash(__name__)

# HTML layout of the app
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='lattices-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='coordinates-plot', style={'width': '50%', 'display': 'inline-block'})
    ]),
    dcc.Input(id='sigma-input', type='number', value=1, style={'marginTop': '20px'})
])

# Callback function to update the 'coordinates-plot' when hover data on 'lattices-plot' or when sigma-input changes
@app.callback(
    Output('coordinates-plot', 'figure'),
    [Input('lattices-plot', 'hoverData'),
     Input('sigma-input', 'value')]
)
def update_plot(hoverData, sigma):
    fig = go.Figure()  # Initialize empty Plotly figure
    origin = np.array([0, 0, 0])  # Use a 3D origin point

    if hoverData is not None:
        hover_x = hoverData['points'][0]['x']
        hover_y = hoverData['points'][0]['y']

        # Calculate vector components based on the hover data and sigma
        r12 = (sigma / 3) * hover_y
        r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
        r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)

        v1 = np.sqrt(r12**2 + r01**2)
        v2 = np.sqrt(r12**2 + r02**2)
        ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))

        dir1 = np.array([v1, 0, 0])  # Define direction vector
        dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang), 0])  # Define direction vector

        # Generate lattice points
        for i in range(-5, 6):
            for j in range(-5, 6):
                point = origin + i * dir1 + j * dir2
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode='markers', marker=dict(size=5, color='black')))

        # Draw the square and vectors from the origin
        square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])
        fig.add_trace(go.Scatter3d(x=square_corners[:, 0], y=square_corners[:, 1], z=square_corners[:, 2], mode='lines', fill='toself', line=dict(color='cyan')))
        fig.add_trace(go.Scatter3d(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], z=[origin[2], origin[2]], mode='lines+markers', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter3d(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], z=[origin[2], origin[2]], mode='lines+markers', line=dict(color='blue', width=2)))

    fig.update_layout(
        title='3D Lattice from Hovered Point',
        scene=dict(
            xaxis=dict(range=[-1, 1], title='x'),
            yaxis=dict(range=[-1, 1], title='y'),
            zaxis=dict(range=[-1, 1], title='z')
        ),
        showlegend=False
    )
    return fig

# Callback function to update the 'lattices-plot' when sigma-input changes
@app.callback(
    Output('lattices-plot', 'figure'),
    [Input('sigma-input', 'value')]
)
def update_lattices_plot(sigma):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', fill='toself', fillcolor='yellow', marker=dict(size=10, color='blue')))
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    grid_z = np.zeros_like(grid_x)  # Create a flat z-axis grid
    fig.add_trace(go.Scatter3d(x=grid_x, y=grid_y, z=grid_z, mode='markers', marker=dict(size=1, color='rgba(255, 255, 255, 0)'), hoverinfo='none'))
    fig.update_layout(
        title='3D Lattices',
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z')
        ),
        hovermode='closest',
        showlegend=False
    )
    return fig

# Main execution point
if __name__ == '__main__':
    app.run_server(debug=True)
