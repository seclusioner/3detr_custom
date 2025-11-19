import argparse
import os
import sys

import numpy as np
import torch

# 3DETR codebase specific imports
from datasets import build_dataset
from models import build_model
from utils.nms import *

##### Added #####
import open3d as o3d 
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import datasets.votenet.custom.custom_utils as custom_utils

PATH = "datasets/votenet/custom/custom_trainval/"
OUTPUT_DIR = "outputs/demo/"

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")


    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", default="custom" ,type=str, choices=["scannet", "sunrgbd", "custom"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )

    ##### Testing #####
    parser.add_argument("--test_only", default=True, action="store_true")  # 僅進行推理
    parser.add_argument("--test_ckpt", required=True, type=str, help="Path to the checkpoint model for testing")
    parser.add_argument("--output", default="3D", required=True, type=str, help="Decide output result is 2D or 3D")

    ##### I/O #####
    parser.add_argument("--log_dir", default='log', help='Log directory for saving results')
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory to save checkpoints (if needed)")

    return parser

######## Input file ########
def load_ply_file(ply_path, color=True):
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            raise ValueError(f"Point cloud at {ply_path} has no points")

        points = np.asarray(pcd.points)  # 點雲的空間位置

        if color and pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            z_values = points[:, 2]
            z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
            colors = np.stack([z_normalized, z_normalized, z_normalized], axis=1)  # 三個通道一樣，形成灰度顏色
        return points, colors
    except Exception as e:
        print(f"Error loading point cloud {ply_path}: {e}")
        return np.array([]), np.array([])  # 若無法讀取，返回空陣列

def convert_ply_to_npz(ply_path, output_npz_path, num_points=50000):
    point_cloud_np, color_np = load_ply_file(ply_path)
    print("Loading .ply successfully.")

    if point_cloud_np.shape[0] > num_points:
        indices = np.random.choice(point_cloud_np.shape[0], num_points, replace=False)
        point_cloud_np = point_cloud_np[indices]
        color_np = color_np[indices]
    
    point_cloud_data = np.concatenate((point_cloud_np, color_np), axis=-1)

    if not os.path.exists(output_npz_path):
        os.makedirs(output_npz_path)
        print(f"Directory {output_npz_path} created.")
    
    output_file_path = os.path.join(output_npz_path, "test.npz")
    np.savez_compressed(output_file_path, pc=point_cloud_data)
    
    print(f"Saved converted .npz file to {output_file_path}")

def load_npz_file(npz_dir):
    if not os.path.isdir(npz_dir):
        raise ValueError(f"The provided directory '{npz_dir}' does not exist.")
    
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    if not npz_files:
        raise ValueError(f"No .npz files found in the directory '{npz_dir}'.")
    
    point_clouds = []
    for npz_file in npz_files:
        npz_file_path = os.path.join(npz_dir, npz_file)
        data = np.load(npz_file_path)
        print(f"Loading .npz file: {npz_file_path}")
        
        if 'pc' in data:
            point_cloud_np = data['pc']
            point_clouds.append(point_cloud_np)
        else:
            print(f"Warning: {npz_file} does not contain 'pc' key, skipping.")
    
    return point_clouds

def compute_point_cloud_dims(point_cloud_np):
    # 計算每個維度（x, y, z）的最小值和最大值
    point_cloud_dims_min = np.min(point_cloud_np[:, :, :3], axis=1)  # 最小值
    point_cloud_dims_max = np.max(point_cloud_np[:, :, :3], axis=1)  # 最大值
    return point_cloud_dims_min, point_cloud_dims_max

######## Model ########
def load_model(args):
    if args.model_name != "3detr":
        raise ValueError(f"Unsupported model name {args.model_name}. Only '3detr' is supported.")
    
    print(f"Initializing dataset: {args.dataset_name}...")
    datasets, dataset_config = build_dataset(args)
    
    print(f"Initializing model: {args.model_name}...")
    model, _ = build_model(args, dataset_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.isfile(args.test_ckpt):
        raise FileNotFoundError(f"Checkpoint file {args.test_ckpt} not found.")
    
    print(f"Loading model from checkpoint: {args.test_ckpt}...")
    
    checkpoint = torch.load(args.test_ckpt, map_location=device)
    
    try:
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {args.test_ckpt}.")
    except KeyError as e:
        print(f"Error: The checkpoint file does not contain the expected key 'model'. KeyError: {e}")
        raise
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}. This could be due to layer mismatches or missing weights.")
        raise

    return model, datasets, dataset_config


######## Calibration ########
def rotate_3d_boxes(boxes, angle_deg=(0, 0, 0)):
    # 角度轉換為弳度
    angle_rad = np.radians(angle_deg)

    # 繞 X 軸的旋轉矩陣
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad[0]), -np.sin(angle_rad[0])],
        [0, np.sin(angle_rad[0]), np.cos(angle_rad[0])]
    ])

    # 繞 Y 軸的旋轉矩陣
    rotation_matrix_y = np.array([
        [np.cos(angle_rad[1]), 0, np.sin(angle_rad[1])],
        [0, 1, 0],
        [-np.sin(angle_rad[1]), 0, np.cos(angle_rad[1])]
    ])

    # 繞 Z 軸的旋轉矩陣
    rotation_matrix_z = np.array([
        [np.cos(angle_rad[2]), -np.sin(angle_rad[2]), 0],
        [np.sin(angle_rad[2]), np.cos(angle_rad[2]), 0],
        [0, 0, 1]
    ])

    # 合併三個旋轉矩陣，先繞 X 軸，再繞 Y 軸，最後繞 Z 軸
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    # 遍歷每個框並應用旋轉
    rotated_boxes = []
    for box in boxes:
        # 應用旋轉
        rotated_box = np.dot(box, rotation_matrix.T)  # 點與旋轉矩陣相乘
        rotated_boxes.append(rotated_box)

    return np.array(rotated_boxes)

######## Projection ########
def project_to_2d(points_3d, K):
    """
    將 3D 點投影到 2D 圖像平面
    :param points_3d: 形狀為 (8, 3) 的 3D 點
    :param K: 相機內參矩陣
    :return: 形狀為 (8, 2) 的 2D 點
    """
    # 3D 點轉換為齊次坐標
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (8, 4)
    K_extended = np.hstack([K, np.array([[0],[0],[1]])])  # (3, 4)
    
    # 投影計算
    points_2d_homogeneous = np.dot(K_extended, points_3d_homogeneous.T).T  # (8, 3)

    # 歸一化為 2D 坐標
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]  # (8, 2)
    
    return points_2d.astype(int)

def project_3d_box_to_2d_image(boxes, K, image):
    """
    將 3D 邊界框投影到 2D 圖像上並繪製
    :param boxes: 形狀為 (K, 8, 3) 的 3D 邊界框
    :param K: 相機內參矩陣
    :param image: 圖像數據
    :return: 帶有 3D 投影的圖像
    """
    for i in range(boxes.shape[0]):
        box_3d = boxes[i]  # 每個 3D 邊界框的 8 個點
        box_2d = project_to_2d(box_3d, K)  # 投影到 2D
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

############# Inference #############
def inference(args, model, point_cloud_np, idx): 
    assert point_cloud_np.shape[1] == 6, f"Point Cloud data dimension is {point_cloud_np.shape}"

    # 變為 (1, 6, N)，即 (batch_size, channels, num_points)，這裡 channels = 6 (xyz + rgb)
    point_cloud_np = np.expand_dims(point_cloud_np, axis=0)  # shape: (1, N, 6)
    
    point_cloud_xyz = point_cloud_np[:, :, :3]  # shape: (1, N, 3) (xyz)
    point_cloud_rgb = point_cloud_np[:, :, 3:]  # shape: (1, N, 3) (rgb)

    point_cloud_input = np.concatenate((point_cloud_xyz, point_cloud_rgb), axis=1)  # shape: (1, 6, N)
    point_cloud_dims_min, point_cloud_dims_max = compute_point_cloud_dims(point_cloud_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 傳入模型的 inputs 應該是字典，並且包含 'point_clouds' 鍵
    inputs = {
        "point_clouds": torch.tensor(point_cloud_input, dtype=torch.float32).to(device),
        "point_cloud_dims_min": torch.tensor(point_cloud_dims_min, dtype=torch.float32).to(device),
        "point_cloud_dims_max": torch.tensor(point_cloud_dims_max, dtype=torch.float32).to(device),
    }

    model.eval()
    with torch.no_grad(): # ~ detach
        outputs = model(inputs)

    #####################
    final_output = outputs['outputs']  # 來自最後一層的預測結果

    boxes = final_output['box_corners']  # torch.Size([1, 256, 8, 3])
    labels = final_output['sem_cls_prob']  # torch.Size([1, 256, 5])
    scores = final_output['objectness_prob']  # torch.Size([1, 256])

    ########## Post-process ##########
    boxes = boxes.squeeze(0)  # shape: (256, 8, 3)
    labels = labels.squeeze(0)  # shape: (256, 5)
    scores = scores.squeeze(0)  # shape: (256)

    score_threshold = 0.5  # 設定得分閾值
    mask = scores > score_threshold  # 過濾條件

    filtered_boxes = boxes[mask]  # shape: (num_valid_boxes, 8, 3)
    filtered_labels = labels[mask]  # shape: (num_valid_boxes, num_classes)
    filtered_scores = scores[mask]  # shape: (num_valid_boxes)

    nms_threshold = 0.5  # 設定 NMS 重疊閾值

    # (num_valid_boxes, 8, 3) -> (num_valid_boxes, 24)
    filtered_boxes_2d = filtered_boxes.view(filtered_boxes.size(0), -1)  # shape: (num_valid_boxes, 24)

    nms_indices = nms_3d_faster(filtered_boxes_2d.cpu().numpy(), overlap_threshold=nms_threshold)

    final_boxes = filtered_boxes[nms_indices]  # shape: (num_nms_boxes, 8, 3)
    final_labels = filtered_labels[nms_indices]  # shape: (num_nms_boxes, num_classes)
    final_scores = filtered_scores[nms_indices]  # shape: (num_nms_boxes)
    ##################################
    if args.output=="3D":
        visualize_3d_boxes(final_boxes, final_labels, final_scores, point_cloud_np[0]) # 3D
    elif args.output=="2D":
        proj_on_img(final_boxes, final_labels, final_scores, idx) # 2D

# point cloud visualization
def visualize_3d_boxes(boxes, labels, scores, points, output_pcd_path=f"{OUTPUT_DIR}output_pcd.ply", output_box_path=f"{OUTPUT_DIR}output_boxes.ply"):
    # (N, 6)，前 3 個元素是 xyz，後 3 個元素是 rgb  50000, 6)
    points_xyz = points[:, :3]
    points_rgb = points[:, 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)
    
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Point cloud saved to {output_pcd_path}")
    
    # 構建 LineSet 來儲存邊界框
    line_set = o3d.geometry.LineSet()
    
    box_points = []
    box_lines = []
    
    boxes = boxes.cpu().numpy()
    calib_boxes = boxes # rotate_3d_boxes(boxes, angle_deg=(-90, 0, 0))

    for i in range(calib_boxes.shape[0]):  # [num_boxes, 8, 3]
        # box = calib_boxes[i].cpu().numpy()  # shape: (8, 3)
        box = calib_boxes[i]  # shape: (8, 3)
        box_points.extend(box)

        # 定義邊界框的 12 條邊
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 頂面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 連接底面與頂面的線
        ])
        box_lines.extend(lines + len(box_points) - 8)  # 更新線索引

    # 將 box_points 和 box_lines 轉換成 Open3D 支援的格式
    box_points = np.array(box_points, dtype=np.float64)
    box_lines = np.array(box_lines, dtype=np.int64)

    line_set.points = o3d.utility.Vector3dVector(box_points)
    line_set.lines = o3d.utility.Vector2iVector(box_lines)

    # 儲存邊界框
    o3d.io.write_line_set(output_box_path, line_set)
    print(f"3D boxes saved to {output_box_path}")

def proj_on_img(boxes, labels, scores, idx, output_path=f"{OUTPUT_DIR}output.png"):
    calib = custom_utils.Custom_Calibration(f"{PATH}calib/calib.txt")
    image_path = f"datasets/votenet/custom/custom_trainval/image/{idx}.png" # images = [f for f in os.listdir("Inputs/image/") if f.endswith('.png')]

    image = cv2.imread(image_path)
    w, h = image.shape[:2]
    image = cv2.resize(image, (h, w))
    image = cv2.rotate(image, cv2.ROTATE_180)

    boxes = boxes.cpu().numpy()

    image_with_3d_box_projection = project_3d_box_to_2d_image(boxes, calib.K, image)

    image = cv2.flip(image_with_3d_box_projection, 0)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    print("Saving image ...")
    cv2.imwrite(output_path, image)

def main(args):
    try:
        model, datasets, dataset_config = load_model(args)
        print("Model has been successfully loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    index = np.random.randint(401, 485)
    index = 484
    print(f"----- data index: {index}")
    npz_path = "datasets/votenet/custom/custom_pc_bbox_votes_50k_val/" + "%06d_pc.npz"%(index) # 401 - 484

    # ply_path = "datasets/votenet/custom/..."
    # convert_ply_to_npz(ply_path, npz_path)

    data = np.load(npz_path)
    input_data = data['pc'] # input_data = load_npz_file(npz_path)

    print("Start infernece ...")
    inference(args, model, input_data, idx=index)
    
if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    print("Good Luck!")
    main(args)
    print("Finished!")
