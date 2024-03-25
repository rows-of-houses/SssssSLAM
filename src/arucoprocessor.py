import cv2 as cv
import numpy as np
import cv2.aruco

class ArucoProcessor:
    def __init__(self):
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(arucoDict)

        self.mtx = np.array([[1.23472227e+03, 0.00000000e+00, 6.38814655e+02],
                             [0.00000000e+00, 1.23122213e+03, 3.37257425e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.dist = np.zeros((1, 5))

        self.marker_size = 0.144
        # self.marker_points = np.array([[-self.marker_size / 2,  self.marker_size / 2, 0],
        #                                [ self.marker_size / 2,  self.marker_size / 2, 0],
        #                                [ self.marker_size / 2, -self.marker_size / 2, 0],
        #                                [-self.marker_size / 2, -self.marker_size / 2, 0]])
        self.marker_points = np.array([[-self.marker_size / 2, -self.marker_size / 2, 0],
                                       [ self.marker_size / 2, -self.marker_size / 2, 0],
                                       [ self.marker_size / 2,  self.marker_size / 2, 0],
                                       [-self.marker_size / 2,  self.marker_size / 2, 0]])

        self.curr_aruco_pose = np.eye(4)
        self.prev_aruco_pose = np.eye(4)
        self.prev_prev_aruco_pose = np.eye(4)

        self.corners = None

        self.iteration = 0

        self.reliable_tracking = False
        self.prev_speed = 100

        self.prev_reliable_pose = np.eye(4)
        self.filtered_aruco_pose = np.eye(4)

        
    def filter_aruco_pose(self):
        disp = np.linalg.norm(self.curr_aruco_pose[:3, 3] - self.prev_aruco_pose[:3, 3])
        # prev_disp = np.linalg.norm(self.prev_aruco_pose[:3, 3] - self.prev_prev_aruco_pose[:3, 3])

        if not self.reliable_tracking and disp < 0.6:
            self.reliable_tracking = True
            self.prev_reliable_pose = self.prev_aruco_pose
            print('Reliable')
        if not self.reliable_tracking:
            return None
        if disp < 0.6 and self.reliable_tracking: # meters
            self.prev_aruco_pose = self.curr_aruco_pose
            return self.curr_aruco_pose
        else:
            return None
    
    def process_frame(self, frame):
        
        corners, ids, rejected_img_points = self.detector.detectMarkers(frame)
        self.corners = corners
        if len(corners) > 0:
            for i in range(0, len(ids)):
                _, rvec, tvec = cv2.solvePnP(self.marker_points,
                                             corners[i][0],
                                             self.mtx,
                                             self.dist,
                                             False,
                                             cv2.SOLVEPNP_IPPE_SQUARE)

                aruco_pose = np.eye(4)
                aruco_rot_matrix, _ = cv2.Rodrigues(rvec)
                aruco_pose[:3, :3] = aruco_rot_matrix
                aruco_pose[:3, 3:] = np.array(tvec)

                self.prev_prev_aruco_pose = self.prev_aruco_pose
                self.prev_aruco_pose = self.curr_aruco_pose
                self.curr_aruco_pose = aruco_pose

                self.filtered_aruco_pose = self.filter_aruco_pose()

                self.iteration += 1

                if self.filtered_aruco_pose is not None:
                    return True
        return False
    
    def get_corners(self):
        return self.corners[0][0]
    
    def get_pose_wrt_aruco(self):
        if self.filtered_aruco_pose is None:
            return None
        return self.invert_se3(self.filtered_aruco_pose)
    
    def get_aruco_pose(self):
        return self.filtered_aruco_pose
    
    def invert_se3(self, se3):
        inv = np.eye(4, dtype=float)
        inv[:3, :3] = se3[:3, :3].T
        inv[:3, 3:] = -se3[:3, :3].T @ se3[:3, 3:]

        return inv