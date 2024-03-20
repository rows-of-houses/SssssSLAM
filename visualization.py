import numpy as np
import plotly.graph_objects as go

def plot_data(poses, landmarks, marker_points, arrows_length=0.05):

    positions = []
    orientations = []

    for i in range(len(poses)):
        positions.append(poses[i][:3, 3])
        orientations.append(poses[i][:3, :3])

    positions = np.array(positions)

    x_unit = np.array([[1],[0],[0]]) * arrows_length
    y_unit = np.array([[0],[1],[0]]) * arrows_length
    z_unit = np.array([[0],[0],[1]]) * arrows_length

    x_ends = []
    y_ends = []
    z_ends = []

    for i in range(len(orientations)):
        x_ends.append(orientations[i] @ x_unit)
        y_ends.append(orientations[i] @ y_unit)
        z_ends.append(orientations[i] @ z_unit)

    layout = go.Layout(scene=dict(aspectmode='data'))

    arrows = []

    for i in range(len(x_ends)):
        arrows.append(go.Scatter3d(x=[positions[i][0], positions[i][0] + x_ends[i][0][0]],
                                   y=[positions[i][1], positions[i][1] + x_ends[i][1][0]],
                                   z=[positions[i][2], positions[i][2] + x_ends[i][2][0]],
                                   marker_size=1, line=dict(color='red',width=2)))

    for i in range(len(y_ends)):
        arrows.append(go.Scatter3d(x=[positions[i][0], positions[i][0] + y_ends[i][0][0]],
                                   y=[positions[i][1], positions[i][1] + y_ends[i][1][0]],
                                   z=[positions[i][2], positions[i][2] + y_ends[i][2][0]],
                                   marker_size=1, line=dict(color='green',width=2)))

    for i in range(len(z_ends)):
        arrows.append(go.Scatter3d(x=[positions[i][0], positions[i][0] + z_ends[i][0][0]],
                                   y=[positions[i][1], positions[i][1] + z_ends[i][1][0]],
                                   z=[positions[i][2], positions[i][2] + z_ends[i][2][0]],
                                   marker_size=1, line=dict(color='blue',width=2)))

    fig = go.Figure(data=[go.Scatter3d(x=positions.T[0],
                                       y=positions.T[1],
                                       z=positions.T[2],
                                       marker_size=3,
                                       text=np.arange(0, len(positions)),
                                       line=dict(color='black',width=2))] + \
                         [go.Scatter3d(x=landmarks.T[0],
                                       y=landmarks.T[1],
                                       z=landmarks.T[2],
                                       mode='markers',
                                       marker=dict(color='blue', size=1))] + \
                         [go.Scatter3d(x=marker_points.T[0],
                                       y=marker_points.T[1],
                                       z=marker_points.T[2],
                                       mode='markers',
                                       marker=dict(color='orange', size=3))] + \
                        arrows,
                        layout=layout)

    fig.show()