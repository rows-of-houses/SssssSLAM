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
        self.marker_points = np.array([[-self.marker_size / 2,  self.marker_size / 2, 0],
                                       [ self.marker_size / 2,  self.marker_size / 2, 0],
                                       [ self.marker_size / 2, -self.marker_size / 2, 0],
                                       [-self.marker_size / 2, -self.marker_size / 2, 0]])

        self.curr_aruco_pose = np.eye(4)
        self.prev_aruco_pose = np.eye(4)
        self.prev_prev_aruco_pose = np.eye(4)

        self.corners = None

        self.iteration = 0
    
    def filter_pose(self, pose):
        speed = np.linalg.norm(pose[:3, 3] - self.prev_aruco_pose[:3, 3]) / 0.033
        # print(speed)
        if speed > 1.2: # real linear speed in m/s!
            filtered_pose = self.prev_aruco_pose @ \
                            self.invert_se3(self.prev_prev_aruco_pose) @ \
                            self.prev_aruco_pose  # extrapolation
            return filtered_pose
        else:
            return pose
    
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

                if self.iteration > 6:
                    aruco_pose = self.filter_pose(aruco_pose)

                self.prev_prev_aruco_pose = self.prev_aruco_pose
                self.prev_aruco_pose = self.curr_aruco_pose
                self.curr_aruco_pose = aruco_pose

                # basically, it is just inverse of SE3
                pose_wrt_aruco = self.invert_se3(self.curr_aruco_pose)

                pose_wrt_aruco = np.array([[1,  0,  0, 0],
                                           [0, -1,  0, 0],
                                           [0,  0, -1, 0],
                                           [0,  0,  0, 1]]) @ pose_wrt_aruco # need to define, if it is needed or not
                
                self.pose_wrt_aruco = pose_wrt_aruco
                self.iteration += 1
            return True
        return False
    
    def get_corners(self):
        return self.corners[0][0]
    
    def get_pose_wrt_aruco(self):
        return self.pose_wrt_aruco
    
    def get_aruco_pose(self):
        return self.curr_aruco_pose
    
    def invert_se3(self, se3):
        inv = np.eye(4, dtype=float)
        inv[:3, :3] = se3[:3, :3].T
        inv[:3, 3:] = -se3[:3, :3].T @ se3[:3, 3:]

        return inv