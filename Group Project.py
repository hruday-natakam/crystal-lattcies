import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Data for plots - coordinates for x and y
x = np.array([1, 0.5, 0, 0, 1])
y = np.array([0, 0.5, 1, 0, 0])

# Create Dash App
app = dash.Dash(__name__)

# HTML layout  
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='lattices-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='coordinates-plot', style={'width': '50%', 'display': 'inline-block'})
    ]),
    dcc.Input(id='sigma-input', type='number', value=1, step=0.1, style={'marginTop': '20px'}),
    html.Label('Range of Lattice Points:', style={'marginLeft': '10px'}),
    dcc.Input(id='range-input', type='number', value=4, min=1, max=20, step=1, style={'margin': '10px'}),
    html.Button('Rotate', id='rotate-button', n_clicks=0, style={'margin': '10px'})  # Rotate button
])

# Function to rotate points by 90 degrees
def rotate_90(x, y):
    """ Rotate a point counterclockwise by 90 degrees around the origin. """
    return -y, x

# Callback function to update the 'coordinates-plot' when hover data on 'lattices-plot', sigma-input changes, or the rotate button is clicked
@app.callback(
    Output('coordinates-plot', 'figure'),
    [Input('lattices-plot', 'hoverData'),
     Input('sigma-input', 'value'),
     Input('range-input', 'value'),
     Input('rotate-button', 'n_clicks')],
    [State('coordinates-plot', 'figure')]
)
def update_plot(hoverData, sigma, range_val, rotate_clicks, existing_figure):
    fig = go.Figure()  # initialize empty plotly figure
    origin = np.array([0, 0])  # 2D origin point
    
    # Initialize default directions
    initial_dir1 = np.array([1, 0])  # Default initial direction for vector 1
    initial_dir2 = np.array([0, 1])  # Default initial direction for vector 2
    dir1 = initial_dir1.copy()  # Working copy of initial_dir1
    dir2 = initial_dir2.copy()  # Working copy of initial_dir2

    # Check for hover data and calculate directional vectors
    if hoverData:
        hover_x = hoverData['points'][0]['x']
        hover_y = hoverData['points'][0]['y']
        mirror_x = 1 - hover_y
        mirror_y = 1 - hover_x
        if hover_x + hover_y > 1:
            hover_x, hover_y = mirror_x, mirror_y

        # Calculation based on hover data and sigma
        r12 = (sigma / 3) * hover_y
        r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
        r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)
        v1 = np.sqrt(r12**2 + r01**2)
        v2 = np.sqrt(r12**2 + r02**2)
        ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))
        dir1 = np.array([v1, 0])
        dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang)])

    # Apply rotations based on the number of button clicks
    for _ in range(rotate_clicks % 4):
        dir1 = np.array(rotate_90(dir1[0], dir1[1]))
        dir2 = np.array(rotate_90(dir2[0], dir2[1]))

    # Draw the initial or rotated vectors and shapes
    square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])
    fig.add_trace(go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', name='square'))
    fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', name='vector1'))

    # Generate lattice points
    lattice_points = []
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            point = origin + i * dir1 + j * dir2
            lattice_points.append([point[0], point[1]])
    lattice_points = np.array(lattice_points)
    fig.add_trace(go.Scatter(x=lattice_points[:, 0], y=lattice_points[:, 1], mode='markers', marker=dict(size=5, color='black'), name='Lattice Points'))

    # Initialize the frames list for animation
    frames = []
    num_frames = 40  # Number of animation frames
    final_dir1 = dir1 * 1.5  # Example final direction adjustment
    final_dir2 = dir2 * 1.5  # Adjust according to actual need

    dir1_steps = np.linspace(initial_dir1, final_dir1, num_frames)
    dir2_steps = np.linspace(initial_dir2, final_dir2, num_frames)

    for i in range(num_frames):
        frame_dir1 = dir1_steps[i]
        frame_dir2 = dir2_steps[i]

        # Compute the new corners of the square for this frame
        frame_corners = np.array([origin,
                                  origin + frame_dir1,
                                  origin + frame_dir1 + frame_dir2,
                                  origin + frame_dir2,
                                  origin])

        # Create the data for the frame, including the vectors and the square
        frame_data = [
            go.Scatter(x=frame_corners[:, 0], y=frame_corners[:, 1], mode='lines', fill='toself', fillcolor='cyan'),
            go.Scatter(x=[origin[0], origin[0] + frame_dir1[0]], y=[origin[1], origin[1] + frame_dir1[1]], mode='lines+markers', line=dict(color='green', width=2)),
            go.Scatter(x=[origin[0], origin[0] + frame_dir2[0]], y=[origin[1], origin[1] + frame_dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)),
            go.Scatter(x=[origin[0], origin[0] - frame_dir1[0] - frame_dir2[0]], y=[origin[1], origin[1] - frame_dir1[1] - frame_dir2[1]], mode='lines+markers', line=dict(color='red', width=2)),
        ]

        # Add the frame to the animation
        frames.append(go.Frame(data=frame_data, name=f'frame{i}'))
    fig.frames = frames

    # Animation controls
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}], 'label': 'Pause', 'method': 'animate'},
                {'args': [["rotate"], {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True}], 'label': 'Structure', 'method': 'animate'}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    # Set fixed axes ranges
    fig.update_layout(
        title='2D Lattice from Hovered Point',
        xaxis=dict(range=[-1, 1], autorange=False),
        yaxis=dict(range=[-1, 1], autorange=False),
        showlegend=False
    )
    return fig

@app.callback(
    Output('lattices-plot', 'figure'),
    [Input('sigma-input', 'value')]
)
def update_lattices_plot(sigma):
    fig = go.Figure()

    # Original scatter plot with markers
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', fill='toself', fillcolor='yellow', marker=dict(size=10, color='blue')))

    # Create a grid of points
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))  # Adjust the range and density as needed
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    # Add the grid as an invisible scatter plot (matching the background color)
    # Setting opacity to 0 makes the points invisible but still able to capture hover events
    fig.add_trace(go.Scatter(x=grid_x, y=grid_y, mode='markers', marker=dict(size=1, color='rgba(255, 255, 255, 0)'), hoverinfo='none'))

    # Adjust x and y axis range
    fig.update_xaxes(range=[min(x), max(x)])
    fig.update_yaxes(range=[min(y), max(y)])

    fig.update_layout(
        title='Lattices',
        xaxis=dict(title='x'),
        yaxis=dict(title='y'),
        hovermode='closest',
        showlegend=False
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)