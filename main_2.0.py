#Dash Core Components (graphs and sliders), INPUT OUTPUT for callbacks, plotly for visualisation
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Data for plots - coordinates for x y
x = np.array([1, 0.5, 0, 0, 1])
y = np.array([0, 0.5, 1, 0, 0])

# Create Dash App
app = dash.Dash(__name__, external_stylesheets=['/style.css'])

#html layout
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
            dcc.Input(id='sigma-input', type='number', value=1, step=0.1),
        ]),
        html.Div(className='input-group', children=[
            html.Label('Range of Lattice Points:', htmlFor='range-input'),
            dcc.Input(id='range-input', type='number', value=4, min=1, max=20, step=1),
        ]),
            html.Div(className='animation', children=[
            html.Button('Animation ON/OFF', id='animation-button', n_clicks=0),
        ])
    ]),
    html.Footer(className='footer', children=[
        html.P(['©2024 Made by Team 5. All rights reserved.']),
    ]),
])

def calculate_vectors(hover_x, hover_y, sigma):
    """Calculate vectors and angles based on input coordinates and sigma."""
    mirror_x, mirror_y = 1 - hover_y, 1 - hover_x # Define mirror coordinates
    if hover_x + hover_y > 1:
        hover_x, hover_y = mirror_x, mirror_y

    r12 = (sigma / 3) * hover_y
    r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
    r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)
    
    v1 = np.sqrt(r12**2 + r01**2)
    v2 = np.sqrt(r12**2 + r02**2)
    ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))
    dir1 = np.array([v1, 0])
    dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang)])

    return dir1, dir2

#Callback function to update the 'coordinates-plot' when hover data on 'lattices-plot' or when sigma-input changes
@app.callback(
    Output('coordinates-plot', 'figure'),
    [Input('lattices-plot', 'hoverData'),
     Input('sigma-input', 'value'),
     Input('range-input', 'value'),
     Input('animation-button', 'n_clicks')],  # Add the animation button as an input
    [State('coordinates-plot', 'figure')]  # Use the current state of the figure
)

#generate Plotly graphs base on hover data from another graph
def update_plot(hoverData, sigma, range_val, n_clicks, current_figure):
    fig = go.Figure() #initialise empty plotly figure
    origin = np.array([0, 0]) # Use a 2D origin point

    hover_x = hover_y = 0  
    # r12 = r01 = r02 = 0  # Default values if hoverData is not available
    animate = n_clicks % 2 != 0 # Check the number of clicks to determine whether to animate or not
    
    if animate:
        # If the button has been clicked an odd number of times, run the animation
        path_points = [
            (0, 0), (0.000001, 0.999999),    # From (0, 0) to (0, 1)
            (0.000002, 1), (0.999999, 0), # From (0, 1) to (1, 0)
            (1, 0.000001), (0.000002, 0.000002),     # From (1, 0) to (0, 0)
            (0.000003, 0.000003), (0.25, 0.25) # From (0, 0) to (0.25, 0.25)
        ]

        total_steps = 160 # Adjust the number of steps in the animation as desired
        steps_per_segment = total_steps // (len(path_points) // 2)

        # Create frames for the animation
        frames = []
        for i in range(0, len(path_points), 2):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            for step in np.linspace(0, 1, steps_per_segment):
                hover_x = (1 - step) * start_point[0] + step * end_point[0]
                hover_y = (1 - step) * start_point[1] + step * end_point[1]
                dir1, dir2 = calculate_vectors(hover_x, hover_y, sigma)
        
                i = np.arange(-range_val, range_val + 1)
                j = np.arange(-range_val, range_val + 1)
                ix, jx = np.meshgrid(i, j, indexing='ij')
                points = origin + ix.reshape(-1, 1) * dir1 + jx.reshape(-1, 1) * dir2
                square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])
                
                frame = go.Frame(
                    data=[
                        go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(size=5, color='black')),
                        go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', line=dict(color='cyan')),
                        go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', line=dict(color='green', width=2)),
                        go.Scatter(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)),
                        go.Scatter(x=[origin[0], origin[0] - dir1[0] - dir2[0]], y=[origin[1], origin[1] - dir1[1] - dir2[1]], mode='lines+markers', line=dict(color='red', width=2))
                    ],
                    name=str(hover_y)
                )
                frames.append(frame)

        # Add the initial state of the figure
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=5, color='black')))
        # Draw the initial square
        square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])
        fig.add_trace(go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', line=dict(color='cyan')))
        # Plot the initial vectors from the origin
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[origin[0], origin[0] - dir1[0] - dir2[0]], y=[origin[1], origin[1] - dir1[1] - dir2[1]], mode='lines+markers', line=dict(color='red', width=2)))

        # Set up layout for the animation
        fig.update_layout(
            title='2D Lattice from Hovered Point',
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            showlegend=False,
            updatemenus=[{
                'buttons': [{
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },{
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }],
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

        # Set up the initial frame
        fig['frames'] = frames
        fig['layout']['sliders'] = [{
            'steps': [{'args': [[f.name], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': round(float(f.name), 2), 'method': 'animate'} for f in frames],
            'transition': {'duration': 300},
            'x': 0.1,
            'y': 0,
            'currentvalue': {'font': {'size': 12}, 'prefix': 'Y-value: ', 'visible': True, 'xanchor': 'right'},
            'len': 0.9,
            'xanchor': 'left',
            'yanchor': 'top'
        }]

    else:
        # If the button has been clicked an even number of times, handle hover data
        if hoverData:
            hover_x = hoverData['points'][0]['x']
            hover_y = hoverData['points'][0]['y']
            dir1, dir2 = calculate_vectors(hover_x, hover_y, sigma)
            
            # Generate lattice points using efficient numpy operations
            i = np.arange(-range_val, range_val + 1)
            j = np.arange(-range_val, range_val + 1)
            ix, jx = np.meshgrid(i, j, indexing='ij')
            points = origin + ix.reshape(-1, 1) * dir1 + jx.reshape(-1, 1) * dir2
            fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(size=5, color='black')))
            
            # Draw the square
            square_corners = np.array([origin, origin + dir1, origin + dir1 + dir2, origin + dir2, origin])

            # Plot the vectors from the origin
            fig.add_trace(go.Scatter(x=square_corners[:, 0], y=square_corners[:, 1], mode='lines', fill='toself', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=[origin[0], origin[0] - dir1[0] - dir2[0]], y=[origin[1], origin[1] - dir1[1] - dir2[1]], mode='lines+markers', line=dict(color='red', width=2)))

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