import os
import json
import mmcv
from mmpose.apis import init_model, inference_topdown

# 입력 경로
image_path = '/mnt/d/DL-proj/vitpose/input/00004_image_000001_visible.png'
output_path = '/mnt/d/DL-proj/vitpose/output/00004_image_000001_visible.json'

# 모델 경로
# classic decoder
config_file = '/mnt/d/DL-proj/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py'
checkpoint_file = '/mnt/d/DL-proj/vitpose/ckpt/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth'

# # simple decoder
# config_file = '/mnt/d/DL-proj/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192.py'
# checkpoint_file = '/mnt/d/DL-proj/vitpose/ckpt/td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192-3a7ee9e1_20230314.pth'

# 모델 초기화
model = init_model(config_file, checkpoint_file, device='cuda')

# 이미지 로드 및 resize
image = mmcv.imread(image_path)
orig_h, orig_w = image.shape[:2]
image = mmcv.imresize(image, (256, 192))

# keypoint 검출
pose_result = inference_topdown(model, image)
keypoints = pose_result[0].pred_instances.keypoints

# keypoint (256x192 기준) (17, 2)
keypoints = pose_result[0].pred_instances.keypoints[0]

# 보정: resize → 원본 이미지 좌표
inference_w, inference_h = 256, 192
scale_x = orig_w / inference_w
scale_y = orig_h / inference_h
keypoints[:, 0] *= scale_x
keypoints[:, 1] *= scale_y

# 저장 (shape = (17, 2))
json_data = {
    'keypoints': keypoints.tolist()
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(json_data, f, indent=4)

print(f'Pose keypoints (17x2) saved to: {output_path}')
