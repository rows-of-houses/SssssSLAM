import numpy as np
import plotly.graph_objects as go
import cv2 as cv


def video_to_nparray(path):
    cap = cv.VideoCapture(path)

    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return buf


def plot_data(poses, landmarks, marker_points, arrows_length=0.05, window=3):

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
                          arrows)
    
    width = np.max(positions, axis=1) - np.min(positions, axis=1)
    range_ = np.vstack((np.mean(positions, axis=1) - window * width, np.mean(positions, axis=1) + window * width))

    range_ = range_.T

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=range_[0]),
            yaxis = dict(nticks=4, range=range_[1]),
            zaxis = dict(nticks=4, range=range_[2]),
            aspectmode='data'))
    
    fig.show()

    return fig