import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np

# Create Dash App
app = dash.Dash(__name__)

# HTML layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='lattices-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='coordinates-plot', style={'width': '50%', 'display': 'inline-block'})
    ]),
    dcc.Input(id='sigma-input', type='number', value=1, step=0.1, style={'marginTop': '20px'}),
    html.Label('Range of Lattice Points:'),
    dcc.Input(id='range-input', type='number', value=4, min=1, max=20, step=1, style={'margin': '10px'})
])

# Function to calculate directional vectors based on hover data and sigma
def calculate_vectors(hover_x, hover_y, sigma):
    r12 = (sigma / 3) * hover_y
    r01 = (sigma / 6) * (3 - 3 * hover_x - hover_y)
    r02 = (sigma / 6) * (3 + 3 * hover_x - hover_y)

    v1 = np.sqrt(r12**2 + r01**2)
    v2 = np.sqrt(r12**2 + r02**2)
    ang = np.arccos((-4 * hover_y**2) / np.sqrt((9 * hover_x**2 + 5 * hover_y**2 - 6 * hover_y + 9)**2 - 36 * hover_x**2 * (3 - hover_y)**2))
    dir1 = np.array([v1, 0])
    dir2 = np.array([v2 * np.cos(ang), v2 * np.sin(ang)])
    
    return dir1, dir2

# Callback function to update the 'coordinates-plot' with animation
@app.callback(
    Output('coordinates-plot', 'figure'),
    [Input('lattices-plot', 'hoverData'),
     Input('sigma-input', 'value'),
     Input('range-input', 'value')]
)
def update_plot(hoverData, sigma, range_val):
    fig = go.Figure()
    origin = np.array([0, 0])

    if hoverData:
        hover_x = hoverData['points'][0]['x']
        hover_y = hoverData['points'][0]['y']

        if hover_x + hover_y > 1:
            hover_x = 1 - hover_y
            hover_y = 1 - hover_x

        dir1, dir2 = calculate_vectors(hover_x, hover_y, sigma)

        num_frames = 40
        for i in range(num_frames):
            frame_hover_x = np.interp(i, [0, num_frames-1], [0.5, hover_x])
            frame_hover_y = np.interp(i, [0, num_frames-1], [0.5, hover_y])
            frame_dir1, frame_dir2 = calculate_vectors(frame_hover_x, frame_hover_y, sigma)

            frame_data = [
                go.Scatter(x=[origin[0], origin[0] + frame_dir1[0]], y=[origin[1], origin[1] + frame_dir1[1]], mode='lines+markers', line=dict(color='green', width=2)),
                go.Scatter(x=[origin[0], origin[0] + frame_dir2[0]], y=[origin[1], origin[1] + frame_dir2[1]], mode='lines+markers', line=dict(color='blue', width=2))
            ]

            fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir1[0]], y=[origin[1], origin[1] + dir1[1]], mode='lines+markers', line=dict(color='green', width=2)))
            fig.add_trace(go.Scatter(x=[origin[0], origin[0] + dir2[0]], y=[origin[1], origin[1] + dir2[1]], mode='lines+markers', line=dict(color='blue', width=2)))

            fig.add_frame(go.Frame(data=frame_data, name=f'frame{i}'))

    fig.update_layout(
        title='2D Lattice from Hovered Point',
        xaxis=dict(range=[-1, 1], autorange=False),
        yaxis=dict(range=[-1, 1], autorange=False),
        showlegend=False,
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}], 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}], 'label': 'Pause', 'method': 'animate'}
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

    return fig

# Main execution
if __name__ == '__main__':
    app.run_server(debug=True)
