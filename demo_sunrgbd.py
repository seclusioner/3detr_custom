'''
Demo code for SUN RGBD
'''

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

import datasets.votenet.sunrgbd.sunrgbd_utils as sunrgbd_utils

DEFAULT_TYPE_WHITELIST = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

PATH = "datasets/votenet/sunrgbd/sunrgbd_trainval/"
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
        "--dataset_name", default="sunrgbd" ,type=str, choices=["scannet", "sunrgbd", "custom"]
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

    ##### I/O #####
    parser.add_argument("--log_dir", default='log', help='Log directory for saving results')
    parser.add_argument("--checkpoint_dir", default=None, type=str, help="Directory to save checkpoints (if needed)")

    return parser

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

def compute_point_cloud_dims(point_cloud_np):
    # 計算每個維度（x, y, z）的最小值和最大值
    point_cloud_dims_min = np.min(point_cloud_np[:, :, :3], axis=1)  # 最小值
    point_cloud_dims_max = np.max(point_cloud_np[:, :, :3], axis=1)  # 最大值
    return point_cloud_dims_min, point_cloud_dims_max

######## Calibration ########
def rotate_point(corner, heading_angle):
    R = np.array([
        [np.cos(heading_angle), -np.sin(heading_angle), 0],
        [np.sin(heading_angle), np.cos(heading_angle), 0],
        [0, 0, 1]
    ])
    # 旋轉頂點
    rotated_corner = np.dot(R, corner)
    return rotated_corner

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

############# Inference #############
def inference(args, model, point_cloud_np, idx):
    assert point_cloud_np.shape[1] == 6, f"Point Cloud data dimension is {point_cloud_np.shape}"

    # 變為 (1, 6, N)，即 (batch_size, channels, num_points)，這裡 channels = 6 (xyz + rgb)
    point_cloud_np = np.expand_dims(point_cloud_np, axis=0)  # shape: (1, N, 6)
    
    # 分別提取 xyz 和 rgb
    point_cloud_xyz = point_cloud_np[:, :, :3]  # shape: (1, N, 3) (xyz)
    point_cloud_rgb = point_cloud_np[:, :, 3:]  # shape: (1, N, 3) (rgb)

    point_cloud_input = np.concatenate((point_cloud_xyz, point_cloud_rgb), axis=1)  # shape: (1, 6, N)
    point_cloud_dims_min, point_cloud_dims_max = compute_point_cloud_dims(point_cloud_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = {
        "point_clouds": torch.tensor(point_cloud_input, dtype=torch.float32).to(device),
        "point_cloud_dims_min": torch.tensor(point_cloud_dims_min, dtype=torch.float32).to(device),
        "point_cloud_dims_max": torch.tensor(point_cloud_dims_max, dtype=torch.float32).to(device),
    }
    
    model.eval()
    with torch.no_grad(): # ~ detach
        outputs = model(inputs)

    # ----------------------------------------------------------------------------------------------------
    # print(outputs["outputs"].keys())
    # dict_keys(['sem_cls_logits', 'center_normalized', 'center_unnormalized', 
    # 'size_normalized', 'size_unnormalized', 'angle_logits', 'angle_residual', 
    # 'angle_residual_normalized', 'angle_continuous', 'objectness_prob', 'sem_cls_prob', 
    # 'box_corners'])
    # print((outputs["aux_outputs"])) # list


    #####################
    final_output = outputs['outputs']  # 來自最後一層的預測結果

    ################## Trace code ref. ##################
    # batchsize = outputs['outputs']["sem_cls_prob"].shape[0]
    # nqueries = outputs['outputs']["sem_cls_prob"].shape[1]
    # outputs loss
    
    # original
    center = final_output['center_unnormalized']
    boxes = final_output['box_corners']  # torch.Size([1, 256, 8, 3])
    labels = final_output['sem_cls_prob']  # torch.Size([1, 256, 5]) -> pred_cls_prob
    scores = final_output['objectness_prob']  # torch.Size([1, 256])
    heading_angles = final_output['angle_residual'] #
    
    ########## Post-process ##########
    # 去除 batch 維度，將形狀轉換為 (256, 8, 3)，(256, num_classes)，(256)
    boxes = boxes.squeeze(0)  # shape: (256, 8, 3)
    labels = labels.squeeze(0)  # shape: (256, 5)
    scores = scores.squeeze(0)  # shape: (256)
    heading_angles = heading_angles.squeeze(0)  # shape: (256)
    
    center = center.squeeze(0)

    score_threshold = 0.9  # 0.9
    mask = scores > score_threshold  # 過濾條件

    filtered_center = center[mask]    
    
    filtered_boxes = boxes[mask]    # shape: (num_valid_boxes, 8, 3)
    filtered_labels = labels[mask]  # shape: (num_valid_boxes, num_classes) -> OK
    filtered_scores = scores[mask]  # shape: (num_valid_boxes)

    nms_threshold = 0.6  # 設定 NMS 重疊閾值

    # 將 boxes 的維度展開為 (num_valid_boxes, 24)，其中 24 來自於 8 個頂點，每個頂點有 3 個座標
    # (num_valid_boxes, 8, 3) -> (num_valid_boxes, 24)
    filtered_boxes_2d = filtered_boxes.view(filtered_boxes.size(0), -1)  # shape: (num_valid_boxes, 24)

    # 在這裡傳遞展開後的 boxes 給 nms_3d_faster 函數
    nms_indices = nms_3d_faster(filtered_boxes_2d.cpu().numpy(), overlap_threshold=nms_threshold)

    filtered_centers = filtered_center[nms_indices]
    final_boxes = filtered_boxes[nms_indices]  # shape: (num_nms_boxes, 8, 3)
    final_labels = filtered_labels[nms_indices]  # shape: (num_nms_boxes, num_classes)
    

    final_scores = filtered_scores[nms_indices]  # shape: (num_nms_boxes)
    final_heading_angles = heading_angles[nms_indices]  # shape: (num_nms_boxes)
    
    predicted_indices  = torch.argmax(final_labels, dim=1)
    final_labels = [DEFAULT_TYPE_WHITELIST[idx] for idx in predicted_indices.cpu().numpy()]

    ##################################
    # visualize_3d_boxes(final_boxes, final_labels, final_scores, point_cloud_np[0], idx=idx) # 3D
    demo(final_boxes, final_labels, final_scores, final_heading_angles, filtered_centers, idx=idx) # 2D
    # demo2(final_boxes, final_labels, final_scores, final_heading_angles, filtered_centers, idx=idx) # 2D

def visualize_3d_boxes(boxes, labels, scores, points, idx, output_pcd_path=f"{OUTPUT_DIR}output_pcd.ply", output_box_path=f"{OUTPUT_DIR}output_boxes.ply"):
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
    
    # 創建一個空的 list 來儲存邊界框的點
    box_points = []
    box_lines = []
    boxes = boxes.cpu().numpy()
    
    # 座標校正
    calib = sunrgbd_utils.SUNRGBD_Calibration(f"{PATH}calib/" + "%06d.txt"%(idx))
    calib_boxes = rotate_3d_boxes(boxes, angle_deg=(-90, 0, 0))
    # calib_boxes = calib.project_camera_to_upright_depth(np.expand_dims(boxes, axis=0).astype(np.float32))

    # 對每個框進行處理
    for i in range(calib_boxes.shape[0]):  # 假設 boxes 的形狀是 [num_boxes, 8, 3]
        # box = calib_boxes[i].cpu().numpy()  # shape: (8, 3)
        box = calib_boxes[i]  # shape: (8, 3)
        box_points.extend(box)  # 將框的頂點加入列表

        # 定義邊界框的 12 條邊
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 頂面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 連接底面與頂面的線
        ])
        box_lines.extend(lines + len(box_points) - 8)  # 更新線索引

    # 將 box_points 和 box_lines 轉換成 Open3D 支援的格式
    box_points = np.array(box_points, dtype=np.float64)  # 確保數據類型是 np.float64
    box_lines = np.array(box_lines, dtype=np.int64)  # 確保線索引是 int64

    line_set.points = o3d.utility.Vector3dVector(box_points)
    line_set.lines = o3d.utility.Vector2iVector(box_lines)

    # 儲存邊界框
    o3d.io.write_line_set(output_box_path, line_set)
    print(f"3D boxes saved to {output_box_path}")
    
def demo(boxes, labels, scores, heading_angles, centers, idx, output_path="output.jpg"):
    """
    Sol: project_upright_camera_to_upright_depth -> project_upright_depth_to_image

    """
    calib = sunrgbd_utils.SUNRGBD_Calibration(f"{PATH}calib/" + "%06d.txt"%(idx))
    img = cv2.imread(f"{PATH}image/"+"%06d.jpg"%(idx))

    boxes = boxes.cpu().numpy()
    classname = 0
    for i, corners_3d in enumerate(boxes):
        corners_2d = []
        for corner in corners_3d:
            pc = calib.project_upright_camera_to_upright_depth(np.expand_dims(corner, axis=0).astype(np.float32))
            uv, _ = calib.project_upright_depth_to_image(pc)
            corners_2d.append([int(np.round(uv[0, 0])), int(np.round(uv[0, 1]))])
            
        corners_2d = np.array(corners_2d)

        # 頂面 (前四個點)
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i + 1) % 4])  # 連接頂面邊
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # 顯示為綠色線條

        # 底面 (後四個點)
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i + 1) % 4 + 4])  # 連接底面邊
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # 顯示為綠色線條

        # 垂直邊 (頂面到底面的連接)
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[i + 4])  # 連接頂面到底面的邊
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)  # 顯示為綠色線條

        
        # 在頂左角顯示物體標籤
        label = labels[classname]
        classname+=1
        label_position = (corners_2d[6][0], corners_2d[6][1] - 10)
        cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("Saving image ...")
    cv2.imwrite(output_path, img)

def demo2(boxes, labels, scores, heading_angles, centers, idx, output_path="output.jpg"):
    """
    Sol: project_upright_camera_to_upright_depth -> project_upright_depth_to_image

    """
    calib = sunrgbd_utils.SUNRGBD_Calibration(f"{PATH}calib/" + "%06d.txt"%(idx))
    img = cv2.imread(f"{PATH}image/"+"%06d.jpg"%(idx))

    boxes = boxes.cpu().numpy()
    for _, corners_3d in enumerate(boxes):
        corners_2d = []
        for corner in corners_3d:
            pc = calib.project_upright_camera_to_upright_depth(np.expand_dims(corner, axis=0).astype(np.float32))
            uv, _ = calib.project_upright_depth_to_image(pc)
            corners_2d.append([int(np.round(uv[0, 0])), int(np.round(uv[0, 1]))])
            
        corners_2d = np.array(corners_2d)
        img = sunrgbd_utils.draw_projected_box3d(img, corners_2d, labels, (0, 255, 0))

    print("Saving image ...")
    cv2.imwrite(output_path, img)

def main(args):
    """
    train: 5051 - 10335
    val: 1 - 5050
    """
    idxs = np.array(range(1, 10336))
    np.random.seed(0)
    np.random.shuffle(idxs)
    for idx in range(10336):
        idx = idxs[idx]
        print('-' * 10, 'data index: ', idx)
        if idx > 0 and idx < 5051:
            npz_path = "datasets/votenet/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val/" + "%06d_pc.npz"%(idx)
        elif idx > 5050 and idx < 10335:
            npz_path = "datasets/votenet/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_train/" + "%06d_pc.npz"%(idx)
        else:
            print("Index out of range!")
            exit()

        data = np.load(npz_path)
        input_data = data['pc']
        try:
            model, datasets, dataset_config = load_model(args)
            print("Model has been successfully loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
        print("Start infernece ...")
        inference(args, model, input_data, idx=idx)
        print('-'*30)
        print('Type anything to continue to the next sample...')
        input()

if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    main(args)
    print("Finished!")
