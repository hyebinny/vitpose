import os
import json
import numpy as np
import mmcv
import cv2
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer

# 경로 설정
image_path = '/mnt/d/DL-proj/vitpose/input/00004_image_000001_visible.png'
json_path = '/mnt/d/DL-proj/vitpose/output/00004_image_000001_visible.json'

# 이미지 불러오기
image = mmcv.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# keypoints 불러오기
with open(json_path, 'r') as f:
    data = json.load(f)

keypoints = np.array(data['keypoints'], dtype=np.float32).reshape(1, 17, 2)  # (1, 17, 2)

# PoseDataSample 객체 구성
pose_data = PoseDataSample()
pose_data.pred_instances = InstanceData()
pose_data.pred_instances.keypoints = keypoints 
pose_data.gt_instances = InstanceData()

dataset_meta = {
    'keypoint_name': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ],
    'keypoint_id2name': {
        i: name for i, name in enumerate([
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ])
    },
    'skeleton_links': [
        [15, 13], [13, 11], [16, 14], [14, 12],
        [11, 12], [5, 11], [6, 12], [5, 6],
        [5, 7], [6, 8], [7, 9], [8, 10],
        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
        [3, 5], [4, 6]
    ],
    'keypoint_colors': [(255, 0, 0)] * 17,
    'link_colors': [(0, 255, 0)] * 19,
}

# Visualizer 초기화
visualizer = PoseLocalVisualizer(kpt_color = dataset_meta['keypoint_colors'], link_color = dataset_meta['link_colors'])
visualizer.set_dataset_meta(dataset_meta)

# 시각화 수행
visualizer.add_datasample(
    name='vis_result',
    image=image,
    data_sample=pose_data,
    draw_bbox=False,
    draw_gt = False,
    show=False,  
    out_file='/mnt/d/DL-proj/vitpose/output/pose_result.jpg'
)
