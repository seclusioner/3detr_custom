import os
import numpy as np
from torch.utils.data import Dataset

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)

import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = "datasets/votenet/custom/" ## Replace with path to dataset
DATA_PATH_V2 = "" 
# Label_Dir = "dataloader_test/labels"
# PC_Dir = "dataloader_test/pointclouds"

CLASS = ['box', 'person', 'stacker', 'cart', 'others']

def load_ply_point_cloud(file_path, use_color=True):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError(f"Point cloud at {file_path} has no points")
        
        points = np.asarray(pcd.points)  # 點雲的空間位置

        # 顏色資訊，如果沒有顏色，設為灰色
        if use_color and pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            z_values = points[:, 2]
            z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
            colors = np.stack([z_normalized, z_normalized, z_normalized], axis=1)

        return points, colors
    
    except Exception as e:
        print(f"Error loading point cloud {file_path}: {e}")
        return np.array([]), np.array([])  # 若無法讀取，返回空陣列

def load_json_labels(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            label_data = json.load(f)
        return label_data
        
    except Exception as e:
        print(f"Error loading label file {file_path}: {e}")
        return []  # 若無法讀取，返回空列表
    
class PointCloudConfig(object):
    def __init__(self):
        self.num_semcls = len(CLASS)
        self.num_angle_bin = 12
        self.max_num_obj = 64
        
        # type -> class index
        self.type2class = {cls: idx for idx, cls in enumerate(CLASS)}
        
        # class index -> type
        self.class2type = {idx: cls for idx, cls in enumerate(CLASS)}
        
        # type -> onehot index (本質上跟 type2class 一樣)
        self.type2onehotclass = {cls: idx for idx, cls in enumerate(CLASS)}

    def angle2class(self, angle):
        """將連續角度轉換為離散角度類別"""
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """將離散角度類別和殘差轉換回角度"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

class PointCloudDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        
        self.dataset_config = dataset_config
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )

        if root_dir is None:
            root_dir = DATA_PATH_V1  # dataset path

        self.data_path = os.path.join(root_dir, f"custom_pc_bbox_votes_50k_{split_set}")
        
        if split_set in ["train", "val"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)]))
            )
        elif split_set in ["trainval"]:
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths
        
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return len(self.scan_names)

    def load_point_cloud(self, scan_path, use_color=True):
        """從Ply檔案加載點雲"""
        ply_file = os.path.join(scan_path, f"{scan_path}.ply")  # 假設點雲存儲在與掃描名稱相同的ply檔案中
        return load_ply_point_cloud(ply_file, use_color=use_color)

    def load_labels(self, scan_path):
        """從JSON檔案加載標籤"""
        label_file = os.path.join(scan_path, f"{scan_path}.json")  # 假設標籤存儲在與掃描名稱相同的json檔案中
        return load_json_labels(label_file)
        
    def convert_labels_to_bbox(self, labels):
        class_mapping = {name: idx for idx, name in enumerate(self.classes)}
        bboxes = []
        for label in labels:
            ########################### Extract label from .json ###########################
            centroid = label['centroid']
            dimensions = label['dimensions']
            rotations = label['rotation']
            
            rotation_matrix = R.from_euler('z', rotation_z).as_matrix()
            centroid = self.rotate_points(centroid.reshape(1, 3), rotation_matrix).flatten()

            x, y, z = centroid[0], centroid[1], centroid[2]

            length, width, height = dimensions[0], dimensions[1], dimensions[2]
            rotation_z = rotations[2]
            
            class_id = class_mapping.get(label['name'], -1)  # -1 代表未知類別
            ########################### bbox ###########################
            # SUN RGBD dataset's lwh is half length
            # 推測：SUN RGB-D 格式中，heading_angle 是由 +X 軸逆時針旋轉到 -Y 軸的角度，而 rotation_z 可能是 +Z 軸的正方向來表示的（ +X 到 +Y ），因此取-
            heading_angle = -rotation_z
            bbox = [class_id, x, y, z, length / 2, width / 2, height / 2, heading_angle]
            bboxes.append(bbox)

        return np.array(bboxes, dtype=np.float32) # (K, 8) K: 物體的數量
    
    def rotate_points(self, points, rotation_matrix):
        """
        旋轉點雲
        :param points: 點雲 (N x 3)
        :param rotation_matrix: 旋轉矩陣 (3 x 3)
        :return: 旋轉後的點雲
        """
        return np.dot(points, rotation_matrix.T)
    
    def __getitem__(self, idx):
        """
        根據索引加載點雲圖和標籤
        :param idx: 資料集的索引
        :return: 加載的點雲圖和標籤
        """
        ############## Loading data ##############
        # Load point clouds
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
        
        # Loading label
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6 --> (50000, 6)
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8

        # label_file = os.path.join(self.label_dir, self.label_files[idx])
        # labels = self.load_labels(label_file)

        # point_cloud, colors = self.load_point_cloud(scan_path, self.use_color)  # (N x 3) [x, y, z], (N x 3) [r, g, b]
        # point_cloud = np.concatenate([point_cloud, colors], axis=1)  # (N x 6) [x, y, z, r, g, b]
        # bboxes = self.convert_labels_to_bbox(labels) # (K, 8)

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6, "Points Cloud dimension is incorrect."
            point_cloud = point_cloud[:, 0:6]
            # point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        if self.use_height: # Preset: False
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1
            )  # (N,4) or (N,7)


        ############## DATA AUGMENTATION ##############
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (
                    1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                rgb_color += (
                    0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )

        ############## Label processing ##############
        max_num_obj = self.max_num_obj
        angle_classes = np.zeros((max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((max_num_obj))
        max_bboxes = np.zeros((max_num_obj, 8))
        
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes[0:len(bboxes), :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((max_num_obj, 6))  # 存儲目標框信息
        
        """ [class_id, x, y, z, length, width, height, rotation_z] """
        for i, bbox in enumerate(bboxes):
            class_id = int(bbox[0])  # 類別 ID
            raw_angles[i] = bbox[6] % (2 * np.pi)  # 計算角度
            box3d_size = bbox[3:6] * 2  # 計算 3D 尺寸
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])  # 角度編碼
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual

            # 計算 3D 邊界框的 8 個角點
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # 計算軸對齊的邊界框
            xmin, ymin, zmin = np.min(corners_3d, axis=0)
            xmax, ymax, zmax = np.max(corners_3d, axis=0)
            target_bboxes[i, :] = np.array([
                (xmin + xmax) / 2,  # 中心位置
                (ymin + ymax) / 2,
                (zmin + zmax) / 2,
                xmax - xmin,  # 邊界框的尺寸
                ymax - ymin,
                zmax - zmin
            ])

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max

        return ret_dict

def test():
    print("Testing ...")


if __name__ == "__main__":
    test()
