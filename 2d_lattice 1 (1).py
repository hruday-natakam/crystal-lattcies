#Dash Core Components (graphs and sliders), INPUT OUTPUT for callbacks, plotly for visualisation
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
# from plotly.subplots import make_subplots

# Data for plots - coordinates for x y
x = np.array([1, 0.5, 0, 0, 1]) 
y = np.array([0, 0.5, 1, 0, 0])

# Import external stylesheets
app = dash.Dash(__name__, external_stylesheets=['/style.css'])

app.layout = html.Div(className='container', children=[
    html.Div(className='header', children=[
        html.H1('Lattice Visualization Tool'),
        html.P('Explore and visualize lattices in 2D space.'),
    ]),
    html.Div(className='description', children=[
        html.P('A lattice is a regular arrangement of points in space. Lattices are commonly used in crystallography and for defining the positions of atoms in crystals. The left graph allows you to select a point to generate a lattice, while the right graph displays the lattice generated from the selected point.'),
    ]),
    html.Div(className='graph-row', children=[
        html.Div(className='graph-column', children=[
            html.H3('Interactive Lattice Selector'),
            html.P('Click on a point in the lattice graph to view its structure.'),
            dcc.Graph(id='lattices-plot', className='graph'),
        ]),
        html.Div(className='graph-column', children=[
            html.H3('2D Lattice Visualization'),
            html.P('View the lattice structure from the selected point.'),
            dcc.Graph(id='coordinates-plot', className='graph'),
        ]),
    ]),
    html.Div(className='input-container', children=[
        html.Div(className='input-group', children=[
            html.Label('Sigma:', htmlFor='sigma-input'),
            dcc.Input(id='sigma-input', type='number', value=1, step=0.1, style={'marginTop': '20px'}),
        ]),
        html.Div(className='input-group', children=[
            html.Label('Range of Lattice Points:', htmlFor='range-input'),
            dcc.Input(id='range-input', type='number', value=4, min=1, max=20, step=1, style={'margin': '10px'}),
        ]),
        html.Button('Animation', id='animation-button', n_clicks=0)
    ]),
    html.Footer(className='footer', children=[
        html.P(['©2024 Made by Team 5. All rights reserved.']),
    ]),
])

#Callback function to update the 'coordinates-plot' when hover data on 'lattices-plot' or when sigma-input changes
@app.callback(
    Output('coordinates-plot', 'figure'),
    [Input('lattices-plot', 'hoverData'),
     Input('sigma-input', 'value'),
     Input('range-input', 'value')]
)

#generate Plotly graphs base on hover data from another graph
# def update_plot(hoverData, sigma):
def update_plot(hoverData, sigma, range_val):
    fig = go.Figure() #initialise empty plotly figure
    origin = np.array([0, 0]) # Use a 2D origin point

    hover_x = hover_y = 0  
    r12 = r01 = r02 = 0  # Default values if hoverData is not available

    # # Check if the hoverData is None (this can be used to detect if animation should be initiated)
    # if not hoverData:
    #     return fig  # Return an empty figure to clear the plot

    if hoverData:
        hover_x = hoverData['points'][0]['x']
        hover_y = hoverData['points'][0]['y']
        # Define mirror coordinates
        mirror_x = 1 - hover_y
        mirror_y = 1 - hover_x

        # Handle mirror transformation for oblique lattices
        if hover_x + hover_y > 1: # checks if the mirrored transformation needs to be applied
            hover_x, hover_y = mirror_x, mirror_y

        # Calculate vector components based on the hover data and sigma
        r12 = (sigma / 3) * hover_y
        r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
        r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)
        
        v1 = np.sqrt(r12**2 + r01**2)
        v2 = np.sqrt(r12**2 + r02**2)
        ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))

        # Define direction vectors
        dir1 = np.array([v1, 0])
        dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang)])

        # Generate lattice points
        for i in range(-range_val, range_val + 1):  # Using range_val to control the extent of the lattice
            for j in range(-range_val, range_val + 1):
                point = origin + i * dir1 + j * dir2
                fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers', marker=dict(size=5, color='black')))

        # Draw the square
        square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])

        # Plot the vectors from the origin
        fig.add_trace(go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] - dir1[0] - dir2[0]], y=[origin[1], origin[1] - dir1[1] - dir2[1]], mode='lines+markers', line=dict(color='red', width=2)))

    # ANIMATION
    # Calculate the initial and final vectors based on hover data
    final_v1 = np.sqrt(r12**2 + r01**2)
    final_v2 = np.sqrt(r12**2 + r02**2)
    final_ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))

    final_dir1 = np.array([final_v1, 0])
    final_dir2 = np.array([final_v2 * np.cos(final_ang), final_v2 * np.sin(final_ang)])

    # Initialize the vectors for animation from the origin (0,0)
    initial_dir1 = np.array([0, 0])
    initial_dir2 = np.array([0, 0])

    # Determine the number of frames for the animation
    num_frames = 40  # Adjust as needed for smoother or faster animation

    # Calculate the steps for each frame
    dir1_steps = np.linspace(initial_dir1, final_dir1, num_frames)
    dir2_steps = np.linspace(initial_dir2, final_dir2, num_frames)

    frames = []  # List to hold all the frames for the animation

    for i in range(num_frames):
        frame_data = []
        current_dir1 = dir1_steps[i]
        current_dir2 = dir2_steps[i]

        # Define initial and final values for the demonstration
        initial_hover_x = 0.5  # Example starting x-coordinate
        initial_hover_y = 0.5  # Example starting y-coordinate
        final_hover_x = 1.0    # Example ending x-coordinate
        final_hover_y = 1.0    # Example ending y-coordinate
        hover_x = np.interp(i, [0, num_frames-1], [initial_hover_x, final_hover_x])
        hover_y = np.interp(i, [0, num_frames-1], [initial_hover_y, final_hover_y])

        # Define mirror coordinates
        mirror_x = 1 - hover_y
        mirror_y = 1 - hover_x

        # Handle mirror transformation for oblique lattices
        if hover_x + hover_y > 1:
            hover_x, hover_y = mirror_x, mirror_y

        # Calculate vector components based on the hover data and sigma
        r12 = (sigma / 3) * hover_y
        r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
        r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)

        v1 = np.sqrt(r12**2 + r01**2)
        v2 = np.sqrt(r12**2 + r02**2)
        ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))

        current_dir1 = np.array([v1, 0])
        current_dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang)])

        # Calculate the corners of the square for the current frame
        square_corners = np.array([origin, origin + current_dir1, origin + current_dir1 + current_dir2, origin + current_dir2, origin])
        frame_data.append(go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', line=dict(color='cyan')))

        # Add the vectors for the current frame
        frame_data.append(go.Scatter(x=[origin[0], origin[0] + current_dir1[0]], y=[origin[1], origin[1] + current_dir1[1]], mode='lines+markers', line=dict(color='green', width=2)))
        frame_data.append(go.Scatter(x=[origin[0], origin[0] + current_dir2[0]], y=[origin[1], origin[1] + current_dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)))
        frame_data.append(go.Scatter(x=[origin[0], origin[0] - current_dir1[0] - current_dir2[0]], y=[origin[1], origin[1] - current_dir1[1] - current_dir2[1]], mode='lines+markers', line=dict(color='red', width=2)))

        # Generate lattice points for each frame
        lattice_points = []
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                point = origin + j * current_dir1 + k * current_dir2
                lattice_points.append([point[0], point[1]])
        lattice_points = np.array(lattice_points)
        frame_data.append(go.Scatter(x=lattice_points[:, 0], y=lattice_points[:, 1], mode='markers', marker=dict(size=5, color='black')))

        frames.append(go.Frame(data=frame_data, name=f'frame{i}'))

    fig.frames = frames  # Add the frames to the figure

    # Define a standard size for all graphs
    graph_width = 600
    graph_height = 400
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                    'label': 'Pause',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                    'label': 'Structure',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                    'label': 'Settings',
                    'method': 'animate'
                }
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

    fig.update_layout(
        title='2D Lattice from Hovered Point',
        xaxis=dict(range=[-1, 1], autorange=False),
        yaxis=dict(range=[-1, 1], autorange=False),
        showlegend=False
    )
    # print(hoverData)
    print(f"Hovered coordinates: ({hover_x}, {hover_y})")

    return fig

# another callback function to update the 'lattices-plot' when sigma-input changes
@app.callback(
    Output('lattices-plot', 'figure'),
    [Input('sigma-input', 'value')]
)

# generate plotly graph bases on hover data
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

# Main execution
if __name__ == '__main__':
    app.run_server(debug=True)