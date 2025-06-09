"""
Helper function to read Custom dataset.

Code is for .ply point clouds and corresponding .json file 
label (labeled by labelCloud)
"""

import numpy as np
import cv2
import os
import json

type2class = {'box': 0, 'person': 1, 'stacker': 2, 'cart': 3, 'others': 4}
class2type = {type2class[t]:t for t in type2class}

# 將 深度坐標系 轉換為 相機坐標系
def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

# 將從 相機坐標系 的點雲轉換為 深度坐標系
def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

class Object3d(object):
    def __init__(self, obj_data):
        self.classname = obj_data['name']  # 物件名稱（如 person、box）
        
        # 解析 centroid (重心)
        centroid = obj_data['centroid']
        self.centroid = np.array([centroid['x'], centroid['y'], centroid['z']])
        
        # 解析 dimensions (尺寸)
        dimensions = obj_data['dimensions']
        self.l = dimensions['length']
        self.w = dimensions['width']
        self.h = dimensions['height']
        
        # 解析 rotations (旋轉角度)
        rotations = obj_data['rotations']
        self.orientation = np.array([rotations['x'], rotations['y'], rotations['z']])
        
        # 計算 heading_angle（這裡以 z 軸旋轉角度為基準）
        # 假設XY翻轉方向不同(加負號)
        self.heading_angle = -self.orientation[2]  # z 軸的旋轉角度轉換為弧度

    def __str__(self): # object
        return f"Class: {self.classname}, {self.centroid}, {self.l}, {self.w}, {self.h}, {self.orientation}, {self.heading_angle}"


# Coordinate Transformation
class Custom_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    # Original camera calibration matrix   
    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3,3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3,3), order='F')
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]

    def __str__(self):
        info = f"Camera Calibration Parameters:\n"
        info += f"Rotation Matrix (Rtilt):\n{self.Rtilt}\n"
        info += f"Intrinsic Matrix (K):\n{self.K}\n"
        info += f"Focal Length (f_u, f_v): ({self.f_u}, {self.f_v})\n"
        info += f"Principal Point (c_u, c_v): ({self.c_u}, {self.c_v})\n"
        return info
    
    def project_camera_to_upright_depth(self, pc):
        ''' Convert points from camera coordinate to upright depth coordinate '''
        points_upright = np.dot(pc[:, 0:3], self.Rtilt.T)
        return points_upright

    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,0:3])) # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)
        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]
        return uv[:,0:2], pc2[:,2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)
    
    ####### Added #######
    def project_camera_to_image(self, pc):
        ''' Convert points from camera coordinate to image coordinate '''
        # Project the 3D points in camera coordinates to image coordinates
        uv = np.dot(pc, np.transpose(self.K))  # (n,3)
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]
        return uv[:, 0:2]

    def project_upright_camera_to_image(self, pc):
        ''' 
        Input: (N,3) array of points in upright camera coordinates
            Output: (N,2) array of 2D points (UV coordinates) in the image plane
        '''
        pc = self.project_upright_camera_to_upright_depth(pc)
        uv, _ = self.project_upright_depth_to_image(pc)
        
        return uv[:, 0:2]


def rotate_corners(corners, angles_deg):
    """
    Rotate the 3D corners around the X, Y, and Z axes by specified angles.
    
    Parameters:
    - corners (np.array): An array of shape (8, 3) containing the 3D coordinates of the corners.
    - angles_deg (tuple): A tuple containing the rotation angles (in degrees) for X, Y, and Z axes: (angle_x, angle_y, angle_z).
    
    Returns:
    - rotated_corners (np.array): The rotated 3D corners.
    """
    # Convert angles to radians
    angles_rad = np.radians(angles_deg)
    
    # Rotation matrix around X axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
        [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]
    ])
    
    # Rotation matrix around Y axis
    R_y = np.array([
        [np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
        [0, 1, 0],
        [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]
    ])
    
    # Rotation matrix around Z axis
    R_z = np.array([
        [np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
        [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
        [0, 0, 1]
    ])
    
    # Combine the three rotation matrices (Z * Y * X order)
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # Rotate the corners by applying the rotation matrix
    rotated_corners = np.dot(corners, R.T)
    
    return rotated_corners

##### 執行三維空間中繞各軸旋轉的操作 #####
# 繞X軸旋轉，所以只改變Y、Z座標

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


# 繞Y軸旋轉，所以只改變X、Z座標
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


# 繞Z軸旋轉，所以只改變X、Y座標
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# Extrinstic matrix
def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """ 
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

""" Label format
SUN RGBD: (K, 15) --> *(K, 13) / (K, 8)
Classname: dining_table, Box2D: [180.0523 136.6551 618.6412 380.4182],
Centroid: [ 0.67      2.76     -1.136364], Dimensions (w, l, h): (1.520546, 0.605098, 0.377273), 
Orientation: [ 0.754443 -0.656365  0.      ], Heading Angle: 0.72
--------------------------------------------------------------------------------
Ours: (K, 8)
Class ID: box, Centroid: (-2.70986008644104, -1.6982731819152832, 1.2134801149368286), 
Dimensions: (0.6242964863777161, 0.3764145076274872, 0.7315064668655396), 
Rotation Z: 1.5184364318847656
"""

def read_custom_label(label_filename): # Label extract
    try:
        with open(label_filename, 'r', encoding='utf-8', errors='ignore') as f:
            label_data = json.load(f)
    except Exception as e:
        print(f"Error loading label file {label_filename}: {e}")
        return []

    objects = []
    for obj_data in label_data['objects']:
        obj = Object3d(obj_data)
        objects.append(obj)
    
    return objects

def load_image(img_filename):
    img = cv2.imread(img_filename)
    height, width = img.shape[:2]
    
    ##### Pre-processing here #####
    img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)))
    # img = cv2.flip(img, 1)
    ###############################

    return img

def load_depth_points_txt(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height 
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])

# 判斷給定的一組點 𝑝 是否在一個 凸包
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

# 從點雲中提取出位於 3D box 內的點
def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)


def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotz(obj.heading_angle) # tmp

    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj.l # along heading arrow
    w = obj.w # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    return np.transpose(corners_3d)

def compute_orientation_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in image coord.
            orientation_3d: (2,3) array in depth coord.
    '''


    # orientation in object coordinate system
    ori = obj.orientation
    orientation_3d = np.array([[0, ori[0]],[0, ori[1]],[0,0]])
    center = obj.centroid
    orientation_3d[0,:] = orientation_3d[0,:] + center[0]
    orientation_3d[1,:] = orientation_3d[1,:] + center[1]
    orientation_3d[2,:] = orientation_3d[2,:] + center[2]
    
    # project orientation into the image plane
    return np.transpose(orientation_3d)

################## Add for projection ##################
def project_to_2d(points_3d, K):
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (8, 4)
    Rt = np.array([
        [  0.54168675,  -0.02536605,  -0.84019761,   1.84421923],
        [  0.53870453,  -0.75682214,   0.37015898,   0.89555374],
        [ -0.64526963,  -0.65312848,  -0.39629572, -12.99710068]
    ])
    point_2ds = []
    for point in points_3d_homogeneous:
        point_2d = K @ Rt @ point
        point_2d = point_2d[:2] / point_2d[2]
        point_2ds.append(point_2d)

    return np.array(point_2ds).astype(int)

def project_3d_box_to_2d_image(box, K, image):
    box_2d = project_to_2d(box, K)
    draw_3d_box_projection(image, box_2d)
    return image

def draw_3d_box_projection(image, box_2d, color=(0, 255, 0), thickness=2):
    """
    在圖像上畫出 3D 邊界框的投影
    :param image: 圖像數據
    :param box_2d: 投影到 2D 平面的 3D 邊界框頂點，形狀為 (8, 2)
    :param color: 邊框顏色
    :param thickness: 邊框厚度
    """
    # 畫出底面框 (0-1-2-3)
    for i in range(4):
        p1 = tuple(box_2d[i])
        p2 = tuple(box_2d[(i + 1) % 4])
        cv2.line(image, p1, p2, color, thickness)

    # 畫出頂面框 (4-5-6-7)
    for i in range(4, 8):
        p1 = tuple(box_2d[i])
        p2 = tuple(box_2d[(i + 1) % 4 + 4])
        cv2.line(image, p1, p2, color, thickness)

    # 畫出連接底面和頂面的線
    for i in range(4):
        p1 = tuple(box_2d[i])
        p2 = tuple(box_2d[i + 4])
        cv2.line(image, p1, p2, color, thickness)

################## Add for projection ##################

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image


import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
